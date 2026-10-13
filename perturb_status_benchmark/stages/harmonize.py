"""S00 cohort harmonization and validation."""

from __future__ import annotations

import glob
from pathlib import Path
import re
import subprocess
import shutil
from typing import Any

import numpy as np
import pandas as pd

from ..io import atomic_write_json, atomic_write_parquet, atomic_write_tsv, read_h5ad_obs
from ..context import StageContext


METHOD_ORDER = ["mixscape", "mixscale", "ps", "target_expression"]


def run(context: StageContext) -> tuple[list[Path], dict[str, Any]]:
    config = context.config
    stage_dir = context.stage_dir
    harmonized_root = stage_dir / "harmonized" / "cell_scores"
    availability_frames: list[pd.DataFrame] = []
    cohort_records: list[dict[str, Any]] = []
    validation: dict[str, Any] = {"status": "pass", "datasets": {}}
    artifacts: list[Path] = []

    for dataset_id in context.dataset_ids:
        dataset = config["datasets"][dataset_id]
        base = _read_cohort(dataset_id, dataset, config["cohort"])
        cohort_records.extend(_cohort_records(dataset_id, dataset, base))
        method_frames = []
        dataset_validation = {
            "eligible_cells": int(len(base)),
            "targeted_cells": int((~base["is_ntc"]).sum()),
            "ntc_cells": int(base["is_ntc"].sum()),
            "methods": {},
        }
        for method_id in METHOD_ORDER:
            frame, method_validation = _read_method(
                config, dataset_id, dataset, method_id, base, stage_dir
            )
            method_frames.append(frame)
            dataset_validation["methods"][method_id] = method_validation
            availability_frames.append(_availability(frame))

        combined = pd.concat(method_frames, ignore_index=True)
        combined.sort_values(["cell_id", "method_id"], kind="mergesort", inplace=True)
        output = harmonized_root / f"dataset_id={dataset_id}" / "part-000.parquet"
        atomic_write_parquet(combined, output)
        artifacts.append(output)
        validation["datasets"][dataset_id] = dataset_validation

    cohort_path = stage_dir / "cohort_summary.tsv"
    availability_path = stage_dir / "method_availability.tsv"
    validation_path = stage_dir / "validation.json"
    atomic_write_tsv(pd.DataFrame(cohort_records), cohort_path)
    availability = pd.concat(availability_frames, ignore_index=True)
    availability.sort_values(["dataset_id", "method_id", "target_id"], inplace=True)
    atomic_write_tsv(availability, availability_path)
    atomic_write_json(validation_path, validation)
    artifacts.extend(sorted((stage_dir / "availability_sources").glob("*.tsv")))
    artifacts.extend([cohort_path, availability_path, validation_path])
    return artifacts, {"datasets": len(context.dataset_ids), "rows": int(sum(_parquet_rows(p) for p in artifacts if p.suffix == ".parquet"))}


def _read_cohort(dataset_id: str, dataset: dict[str, Any], cohort: dict[str, Any]) -> pd.DataFrame:
    mapping = dataset["metadata"]
    fields = ["cell_id", "target", "control", "assignment", "batch", "reagent"]
    source_columns = [mapping[field] for field in fields]
    if mapping.get("construct"):
        source_columns.append(mapping["construct"])
    frame = read_h5ad_obs(dataset["h5ad"], source_columns)
    frame.rename(columns={mapping[field]: field for field in fields}, inplace=True)
    if mapping.get("construct"):
        frame.rename(columns={mapping["construct"]: "construct_id"}, inplace=True)
    else:
        frame["construct_id"] = frame["reagent"]
    eligible = set(cohort["eligible_assignment_structures"])
    frame = frame.loc[frame["assignment"].isin(eligible)].copy()
    frame.rename(columns={"target": "target_id", "control": "is_ntc", "batch": "batch_id", "reagent": "reagent_id"}, inplace=True)
    frame["dataset_id"] = dataset_id
    frame["is_ntc"] = frame["is_ntc"].astype(bool)
    frame["perturbation_order"] = 1
    delimiter = dataset.get("perturbation_delimiter")
    if delimiter:
        targeted = ~frame["is_ntc"]
        frame.loc[targeted, "perturbation_order"] = (
            frame.loc[targeted, "target_id"].astype(str).str.count(re.escape(delimiter)).add(1)
        )
    if frame["cell_id"].duplicated().any():
        raise ValueError(f"{dataset_id}: duplicate eligible cell_id in M01")
    if frame["construct_id"].isna().any():
        raise ValueError(f"{dataset_id}: eligible cells require construct_id")
    return frame[["dataset_id", "cell_id", "target_id", "is_ntc", "batch_id", "reagent_id", "construct_id", "assignment", "perturbation_order"]]


def _read_method(
    config: dict[str, Any],
    dataset_id: str,
    dataset: dict[str, Any],
    method_id: str,
    base: pd.DataFrame,
    stage_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = dataset["methods"][method_id]
    columns = ["cell_id", spec["target_column"], spec["score_column"]]
    if spec.get("status_column"):
        columns.append(spec["status_column"])
    result = pd.read_csv(spec["path"], sep="\t", usecols=list(dict.fromkeys(columns)), low_memory=False)
    result.rename(columns={spec["target_column"]: "result_target", spec["score_column"]: "score"}, inplace=True)
    if result["cell_id"].duplicated().any():
        raise ValueError(f"{dataset_id}/{method_id}: duplicate cell_id")
    base_ids = pd.Index(base["cell_id"])
    result_ids = pd.Index(result["cell_id"])
    missing = base_ids.difference(result_ids)
    unexpected = result_ids.difference(base_ids)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"{dataset_id}/{method_id}: cohort mismatch; missing={len(missing)}, unexpected={len(unexpected)}"
        )
    merged = base.merge(result, on="cell_id", how="left", validate="one_to_one", sort=False)
    target_mismatch = (~merged["is_ntc"]) & (
        merged["target_id"].astype(str) != merged["result_target"].astype(str)
    )
    if target_mismatch.any():
        raise ValueError(f"{dataset_id}/{method_id}: {int(target_mismatch.sum())} target labels disagree")
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
    finite_score = pd.Series(np.isfinite(merged["score"]), index=merged.index)
    merged["method_id"] = method_id
    merged["native_status"] = pd.Series(pd.NA, index=merged.index, dtype="string")

    if method_id == "mixscape":
        raw = merged[spec["status_column"]].astype("string")
        merged.loc[raw == spec["active_status"], "native_status"] = "perturbed"
        merged.loc[raw == spec["inactive_status"], "native_status"] = "np"
        merged["scorable"] = (~merged["is_ntc"]) & finite_score & merged["native_status"].notna()
        reason = "invalid_probability_or_class"
    elif method_id == "mixscale":
        valid_targets = _mixscale_targets(config, dataset_id, spec, stage_dir)
        merged["scorable"] = (~merged["is_ntc"]) & merged["target_id"].isin(valid_targets) & finite_score
        reason = "target_direction_unavailable"
    elif method_id == "ps":
        raw = merged[spec["status_column"]].astype("string")
        allowed = [spec["scorable_status"]]
        merged["scorable"] = (~merged["is_ntc"]) & raw.isin(allowed) & finite_score
        reason = "method_unscorable"
    else:
        raw = merged[spec["status_column"]].astype("string")
        allowed = spec["scorable_status"]
        if not isinstance(allowed, list):
            allowed = [allowed]
        merged["native_status"] = raw
        merged["scorable"] = (~merged["is_ntc"]) & raw.isin(allowed) & finite_score
        reason = "method_unscorable"

    merged["unscorable_reason"] = pd.Series(pd.NA, index=merged.index, dtype="string")
    merged.loc[merged["is_ntc"], "unscorable_reason"] = "reference_control"
    merged.loc[(~merged["is_ntc"]) & (~merged["scorable"]), "unscorable_reason"] = reason
    if method_id in {"ps", "target_expression"}:
        failed = (~merged["is_ntc"]) & (~merged["scorable"]) & raw.notna()
        merged.loc[failed, "unscorable_reason"] = raw.loc[failed]
    output_columns = [
        "dataset_id", "cell_id", "target_id", "is_ntc", "batch_id", "reagent_id", "construct_id",
        "assignment", "perturbation_order", "method_id", "score", "native_status",
        "scorable", "unscorable_reason",
    ]
    out = merged[output_columns].copy()
    details = {
        "rows": int(len(out)),
        "targeted_rows": int((~out["is_ntc"]).sum()),
        "scorable_targeted_rows": int(out["scorable"].sum()),
        "scorable_targets": int(out.loc[out["scorable"], "target_id"].nunique()),
    }
    return out, details


def _mixscale_targets(
    config: dict[str, Any], dataset_id: str, spec: dict[str, Any], stage_dir: Path
) -> set[str]:
    availability = spec["availability"]
    if availability["mode"] != "rds_shards":
        raise ValueError(f"Unsupported Mixscale availability mode: {availability['mode']}")
    pattern = str(config["project_root"] / availability["glob"])
    inputs = [Path(path) for path in sorted(glob.glob(pattern))]
    if not inputs:
        raise FileNotFoundError(f"No Mixscale shard Tool files match: {pattern}")
    output = stage_dir / "availability_sources" / f"{dataset_id}_mixscale_targets.tsv"
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        script = Path(__file__).parents[1] / "r" / "export_mixscale_targets.R"
        rscript = shutil.which("Rscript")
        if rscript is None:
            raise FileNotFoundError("Required executable is not on PATH: Rscript")
        subprocess.run([rscript, str(script), str(output), *map(str, inputs)], check=True)
    targets = pd.read_csv(output, sep="\t", dtype=str)["target_id"]
    return set(targets)


def _availability(frame: pd.DataFrame) -> pd.DataFrame:
    targeted = frame.loc[~frame["is_ntc"]]
    grouped = targeted.groupby(
        ["dataset_id", "method_id", "target_id", "perturbation_order"],
        observed=True,
        sort=False,
    )
    result = grouped.agg(targeted_cells=("cell_id", "size"), scorable_cells=("scorable", "sum")).reset_index()
    result["scorable_target"] = result["scorable_cells"] > 0
    return result


def _cohort_records(dataset_id: str, dataset: dict[str, Any], base: pd.DataFrame) -> list[dict[str, Any]]:
    targeted = base.loc[~base["is_ntc"]]
    records = [{
        "dataset_id": dataset_id,
        "modality": dataset["modality"],
        "stratum": "all",
        "eligible_cells": len(base),
        "ntc_cells": int(base["is_ntc"].sum()),
        "targeted_cells": len(targeted),
        "represented_perturbations": targeted["target_id"].nunique(),
    }]
    if dataset.get("perturbation_delimiter"):
        for label, subset in (("single_gene", targeted[targeted["perturbation_order"] == 1]), ("multi_gene", targeted[targeted["perturbation_order"] > 1])):
            records.append({
                "dataset_id": dataset_id,
                "modality": dataset["modality"],
                "stratum": label,
                "eligible_cells": pd.NA,
                "ntc_cells": pd.NA,
                "targeted_cells": len(subset),
                "represented_perturbations": subset["target_id"].nunique(),
            })
    return records


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows
