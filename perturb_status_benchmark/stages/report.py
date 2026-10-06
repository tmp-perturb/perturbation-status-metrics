"""Final benchmark-table assembly and global validation gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..context import StageContext
from ..io import atomic_write_tsv
from .harmonize import METHOD_ORDER


def run(context: StageContext) -> tuple[list[Path], dict[str, Any]]:
    coverage = _read(context, "coverage", "coverage_summary.tsv")
    engagement = _read(context, "target_expression", "direct_expression_summary.tsv")
    effect = _read(context, "effect", "effect_association.tsv")
    reproducibility = _read(context, "reproducibility", "reproducibility_summary.tsv")
    resources = _read(context, "resources", "resource_summary.tsv")

    index = pd.MultiIndex.from_product(
        [context.dataset_ids, METHOD_ORDER], names=["dataset_id", "method_id"]
    ).to_frame(index=False)
    table = index.merge(
        coverage.loc[coverage["stratum"].eq("all"), [
            "dataset_id", "method_id", "scorable_perturbations", "perturbation_coverage",
            "cells_in_scorable_perturbations", "targeted_cell_coverage",
        ]], on=["dataset_id", "method_id"], how="left", validate="one_to_one"
    )
    table = table.merge(
        engagement.loc[:, [
            "dataset_id", "method_id", "scorable_perturbations", "median_rho", "ci_low", "ci_high",
        ]].rename(columns={
            "scorable_perturbations": "engagement_perturbations", "median_rho": "engagement_median_rho",
            "ci_low": "engagement_ci_low", "ci_high": "engagement_ci_high",
        }), on=["dataset_id", "method_id"], how="left", validate="one_to_one"
    )
    table["engagement_status"] = np.where(table["method_id"].eq("target_expression"), "reference_not_evaluated", np.where(table["engagement_median_rho"].notna(), "reported", "not_estimable"))
    table = table.merge(
        effect.loc[effect["stratum"].eq("all"), ["dataset_id", "method_id", "perturbations", "spearman_rho", "ci_low", "ci_high"]].rename(columns={
            "perturbations": "effect_perturbations", "spearman_rho": "effect_spearman_rho",
            "ci_low": "effect_ci_low", "ci_high": "effect_ci_high",
        }), on=["dataset_id", "method_id"], how="left", validate="one_to_one"
    )
    table = table.merge(
        reproducibility.loc[reproducibility["stratum"].eq("all"), ["dataset_id", "method_id", "status", "perturbations", "median_delta_rho", "ci_low", "ci_high"]].rename(columns={
            "perturbations": "reproducibility_perturbations", "ci_low": "reproducibility_ci_low", "ci_high": "reproducibility_ci_high",
            "status": "reproducibility_status",
        }), on=["dataset_id", "method_id"], how="left", validate="one_to_one"
    )
    table = table.merge(resources, on=["dataset_id", "method_id"], how="left", validate="one_to_one")
    _validate(table, len(context.dataset_ids))

    table_path = context.stage_dir / "benchmark_summary.tsv"
    atomic_write_tsv(table, table_path)
    report_path = context.stage_dir / "benchmark_report.md"
    _write_report(table, report_path, context.config["protocol"])
    return [table_path, report_path], {
        "summary_rows": len(table),
        "global_validation": "pass",
        "standalone_runtime_missing": int(table["standalone_wall_seconds"].isna().sum()),
    }


def _read(context: StageContext, stage: str, name: str) -> pd.DataFrame:
    path = context.upstream(stage, name)
    if not path.exists():
        raise RuntimeError(f"report requires {stage}/{name}")
    return pd.read_csv(path, sep="\t")


def _validate(table: pd.DataFrame, datasets: int) -> None:
    if len(table) != datasets * len(METHOD_ORDER):
        raise ValueError("summary does not contain one row per dataset-method pair")
    required = ["perturbation_coverage", "targeted_cell_coverage", "effect_spearman_rho", "incremental_wall_seconds"]
    if table[required].isna().any().any():
        missing = table.loc[table[required].isna().any(axis=1), ["dataset_id", "method_id"]]
        raise ValueError(f"missing required primary results: {missing.to_dict('records')}")
    for column in ["perturbation_coverage", "targeted_cell_coverage"]:
        if not table[column].between(0, 1).all():
            raise ValueError(f"invalid coverage values in {column}")
    for column in ["effect_spearman_rho", "median_delta_rho"]:
        if not table[column].dropna().between(-2, 2).all():
            raise ValueError(f"invalid correlation-derived values in {column}")
    missing_repro = table["median_delta_rho"].isna()
    if not table.loc[missing_repro, "reproducibility_status"].astype(str).str.startswith("not_applicable").all():
        raise ValueError("missing reproducibility values require explicit not_applicable status")


def _write_report(table: pd.DataFrame, path: Path, protocol: str) -> None:
    lines = [
        f"# Perturbation-status benchmark ({protocol})", "",
        "This report is generated from the frozen target-level benchmark tables.",
        "Coverage is reported separately from performance; no method is treated as ground truth and no composite score is calculated.", "",
        "## Completion", "",
        f"- Dataset-method rows: {len(table)}",
        f"- Complete standalone runtime records: {table['standalone_wall_seconds'].notna().sum()} / {len(table)}",
        "- Detailed confidence intervals, missingness reasons and supporting agreement metrics remain in their stage tables.", "",
        "## Primary table", "",
        "See `benchmark_summary.tsv`. CRISPRko, CRISPRa and CRISPRi remain separate dataset rows.", "",
        "## Required disclosures", "",
        "- PS uses the validated `pooled_global_union_sharded` scalable approximation.",
        "- Within-perturbation target-expression correlation uses NTC-free direction-adjusted direct target-gene expression.",
        "- The target-expression baseline reuses Mixscape matched-control assignments and is not independent ground truth.",
        "- Cross-method agreement is descriptive concordance, not correctness.",
        "- PSE-weighting effects compare different guide implementations with disjoint targeted cells and symmetrized, disjoint batch-matched NTC references; Norman is not applicable.",
        "- Missing resource measurements are retained as missing.", "",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
