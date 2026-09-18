#!/usr/bin/env python3
"""Collect all method branches and run the complete v0.2 benchmark."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination)


def serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items() if key not in {"fingerprint", "config_path"}}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def by_parent(paths: list[str]) -> dict[Path, Path]:
    return {Path(path).resolve().parent: Path(path).resolve() for path in paths}


def select_paths(groups: list[list[str]], predicate) -> list[str]:
    """Filter and de-duplicate OB 0.6 collector inputs.

    OB 0.6 passes every output from a referenced stage to each input argument;
    selection by the declared output id is therefore enforced by filename here.
    """
    selected = {str(Path(path).resolve()) for group in groups for path in group if predicate(Path(path))}
    return sorted(selected)


def write_resource_inputs(config: dict[str, Any], root: Path, method_records: dict[tuple[str, str], dict], shared: dict[str, tuple[Path, dict]]) -> None:
    resources = root / "resource_inputs"
    resources.mkdir(parents=True)
    mixscale_rows, ps_rows, baseline_rows = [], [], []
    dataset_resources = {}
    for dataset in config["datasets"]:
        mixscape = method_records[(dataset, "mixscape")]
        mixscale = method_records[(dataset, "mixscale")]
        ps = method_records[(dataset, "ps")]
        baseline = method_records[(dataset, "target_expression")]
        shared_root, shared_manifest = shared[dataset]
        mixscale_rows.append({
            "dataset": dataset, "wall_time_seconds": mixscale["wall_seconds"],
            "peak_worker_rss_bytes": int(mixscale.get("peak_child_rss_kib", 0)) * 1024,
            "shards": mixscale.get("workers", 1), "max_workers": mixscale.get("workers", 1),
        })
        ps_rows.append({
            "dataset": dataset, "input_build_wall_seconds": np.nan, "input_build_peak_rss_gib": np.nan,
            "successful_attempt_wall_seconds": ps["wall_seconds"],
            "analysis_peak_single_process_rss_gib": ps.get("peak_child_rss_kib", 0) / 2**20,
            "analysis_concurrent_rss_upper_bound_gib": ps.get("peak_child_rss_kib", 0) / 2**20,
            "shards": ps.get("workers", 1), "max_workers": ps.get("workers", 1),
            "execution_mode": "pooled_global_union_sharded", "failed_attempt_wall_seconds": 0,
        })
        baseline_rows.append({
            "dataset": dataset, "wall_seconds": baseline["wall_seconds"],
            "peak_worker_rss_gib": baseline.get("peak_child_rss_kib", 0) / 2**20,
            "concurrent_rss_upper_bound_gib": baseline.get("peak_child_rss_kib", 0) / 2**20,
            "shards": baseline.get("workers", 1), "max_workers": baseline.get("workers", 1),
        })
        dataset_resources[dataset] = {
            "shared_preprocessing_events": str(shared_root / "logs" / "stage_resources.jsonl"),
            "shard_build_time": str(shared_root / "shard_build.time.txt"),
            "mixscape_wall_seconds": mixscape["wall_seconds"],
            "mixscape_peak_worker_rss_gib": mixscape.get("peak_child_rss_kib", 0) / 2**20,
            "mixscape_shards": mixscape.get("workers", 1), "mixscape_workers": mixscape.get("workers", 1),
        }
    mixscale_path = resources / "mixscale.tsv"
    ps_path = resources / "ps.tsv"
    baseline_path = resources / "baseline.tsv"
    pd.DataFrame(mixscale_rows).to_csv(mixscale_path, sep="\t", index=False)
    pd.DataFrame(ps_rows).to_csv(ps_path, sep="\t", index=False)
    pd.DataFrame(baseline_rows).to_csv(baseline_path, sep="\t", index=False)
    config["resource_inputs"] = {
        "mixscale_manifest": str(mixscale_path), "ps_manifest": str(ps_path),
        "target_expression_manifest": str(baseline_path), "datasets": dataset_resources,
    }


def compare_tables(observed_root: Path, reference_root: Path) -> dict[str, Any]:
    comparisons = []
    patterns = ["harmonized/*.parquet", "metrics/**/*.tsv", "tables/*.tsv"]
    for pattern in patterns:
        for observed in sorted(observed_root.glob(pattern)):
            relative = observed.relative_to(observed_root)
            if relative.parts[:2] == ("metrics", "resources"):
                continue
            reference = reference_root / relative
            row = {"path": str(relative), "reference_exists": reference.is_file(), "equal": False}
            if not reference.is_file():
                comparisons.append(row)
                continue
            try:
                left = pd.read_parquet(observed) if observed.suffix == ".parquet" else pd.read_csv(observed, sep="\t")
                right = pd.read_parquet(reference) if reference.suffix == ".parquet" else pd.read_csv(reference, sep="\t")
                row["shape_observed"] = list(left.shape)
                row["shape_reference"] = list(right.shape)
                if left.shape != right.shape or list(left.columns) != list(right.columns):
                    comparisons.append(row)
                    continue
                numeric = [column for column in left.columns if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column])]
                max_diff = 0.0
                if numeric:
                    difference = (left[numeric].astype(float) - right[numeric].astype(float)).abs().to_numpy()
                    max_diff = float(np.nanmax(difference)) if difference.size and not np.isnan(difference).all() else 0.0
                nonnumeric = [column for column in left.columns if column not in numeric]
                text_equal = all(left[column].fillna("<NA>").astype(str).equals(right[column].fillna("<NA>").astype(str)) for column in nonnumeric)
                row.update({"max_abs_numeric_difference": max_diff, "nonnumeric_equal": text_equal, "equal": text_equal and max_diff <= 1e-10})
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            comparisons.append(row)
    return {"status": "PASS" if comparisons and all(item["equal"] for item in comparisons) else "FAIL", "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", default="benchmark")
    parser.add_argument("--data.h5ad", nargs="+", required=True)
    parser.add_argument("--data.spec", nargs="+", required=True)
    parser.add_argument("--shared_signature.bundle", nargs="+", required=True)
    parser.add_argument("--shared_signature.manifest", nargs="+", required=True)
    parser.add_argument("--status_estimation.result", nargs="+", required=True)
    parser.add_argument("--status_estimation.manifest", nargs="+", required=True)
    parser.add_argument("--status_estimation.aux", nargs="+", required=True)
    parser.add_argument("--reference_results", default="")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data_groups = [getattr(args, "data.h5ad"), getattr(args, "data.spec")]
    shared_groups = [getattr(args, "shared_signature.bundle"), getattr(args, "shared_signature.manifest")]
    method_groups = [getattr(args, "status_estimation.result"), getattr(args, "status_estimation.manifest"), getattr(args, "status_estimation.aux")]
    h5ads = by_parent(select_paths(data_groups, lambda path: path.suffix == ".h5ad"))
    specs = [Path(path) for path in select_paths(data_groups, lambda path: path.name.endswith(".spec.json"))]
    shared_bundles = by_parent(select_paths(shared_groups, lambda path: path.name == "shared_signature.tar.gz"))
    shared_manifests = by_parent(select_paths(shared_groups, lambda path: path.name == "shared_manifest.json"))
    results = by_parent(select_paths(method_groups, lambda path: path.name == "status_result.tsv.gz"))
    manifests = by_parent(select_paths(method_groups, lambda path: path.name == "method_manifest.json"))
    auxiliaries = by_parent(select_paths(method_groups, lambda path: path.name == "method_aux.tar.gz"))

    with tempfile.TemporaryDirectory(prefix="pse-benchmark-") as temp_text:
        root = Path(temp_text)
        datasets: dict[str, tuple[dict, Path]] = {}
        for spec_path in specs:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            datasets[spec["dataset"]] = (spec, h5ads[spec_path.parent])
        method_records: dict[tuple[str, str], dict] = {}
        method_files: dict[tuple[str, str], tuple[Path, Path]] = {}
        for parent, manifest_path in manifests.items():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            key = (manifest["dataset"], manifest["method"])
            method_records[key] = manifest
            method_files[key] = (results[parent], auxiliaries[parent])
        shared: dict[str, tuple[Path, dict]] = {}
        for parent, manifest_path in shared_manifests.items():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            destination = root / "shared" / manifest["dataset"]
            destination.mkdir(parents=True)
            safe_extract(shared_bundles[parent], destination)
            shared[manifest["dataset"]] = (destination / "shared", manifest)

        first_spec = next(iter(datasets.values()))[0]
        project = Path(first_spec["project_root"])
        sys.path.insert(0, str(project / "07_benchmark" / "src"))
        from perturb_status_benchmark.config import load_config
        config = serializable(load_config(Path(first_spec["benchmark_config"])))
        config["project_root"] = str(root)
        config["output_root"] = str(root / "published")
        config["work_root"] = str(root / "work")
        config["log_root"] = str(root / "logs")
        config["datasets"] = {dataset: config["datasets"][dataset] for dataset in datasets}
        for dataset, (_, h5ad) in datasets.items():
            config["datasets"][dataset]["h5ad"] = str(h5ad)
            for method in ("mixscape", "mixscale", "ps", "target_expression"):
                result, auxiliary = method_files[(dataset, method)]
                config["datasets"][dataset]["methods"][method]["path"] = str(result)
                aux_dir = root / "method_aux" / dataset / method
                aux_dir.mkdir(parents=True)
                safe_extract(auxiliary, aux_dir)
                if method == "mixscale":
                    config["datasets"][dataset]["methods"][method]["availability"] = {
                        "mode": "rds_shards", "glob": str(aux_dir / "mixscale" / "shards" / "shard_*" / "mixscale_tool.rds")
                    }
        write_resource_inputs(config, root, method_records, shared)
        config_path = root / "benchmark.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = [sys.executable, str(project / "07_benchmark" / "run_benchmark.py"), "--config", str(config_path), "--stage", "all", "--run-id", "omni", "--publish"]
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
        published = root / "published" / "omni"
        shutil.copy2(published / "tables" / "benchmark_summary.tsv", output / "benchmark_summary.tsv")
        shutil.copy2(published / "manifest" / "run_manifest.json", output / "benchmark_manifest.json")
        with tarfile.open(output / "benchmark_results.tar.gz", "w:gz") as archive:
            archive.add(published, arcname="perturbation_status_benchmark")
        if args.reference_results:
            parity = compare_tables(published, Path(args.reference_results))
        else:
            parity = {"status": "NOT_REQUESTED", "comparisons": []}
        (output / "parity_report.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
        if parity["status"] == "FAIL":
            raise RuntimeError("Scientific parity comparison failed; see parity_report.json")


if __name__ == "__main__":
    main()
