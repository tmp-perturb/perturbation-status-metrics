#!/usr/bin/env python3
"""Compute the benchmark-owned target-expression engagement reference."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def execute(command: list[str], log: Path) -> dict[str, object]:
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[key] = "1"
    started = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(["/usr/bin/time", "-v", *command], stdout=stream, stderr=subprocess.STDOUT, env=environment)
    text = log.read_text(encoding="utf-8")
    peak = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", text)
    if process.returncode:
        raise RuntimeError(f"Worker failed: {log}")
    return {"wall_seconds": time.time() - started, "peak_rss_kib": int(peak.group(1)) if peak else None, "log": str(log)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    environment_prefix = args.project_root / "environments/mixscape/env"
    os.environ["R_LIBS_USER"] = str(environment_prefix / "lib/R/library")
    os.environ["PATH"] = f"{environment_prefix / 'bin'}:{os.environ.get('PATH', '')}"
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    commands = [([
        str(args.project_root / "environments/mixscape/env/bin/Rscript"),
        str(ROOT / "reference_expression/run_shard.R"),
        "--config", str(args.config.resolve()), "--shard", str(shard)
    ], output / "logs" / f"shard_{shard:03d}.log") for shard in range(1, int(config["shards"]) + 1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(config["max_workers"])) as pool:
        metrics = list(pool.map(lambda item: execute(*item), commands))

    pieces = [pd.read_csv(path, sep="\t") for path in sorted((output / "shards").glob("shard_*/target_expression_reference.tsv"))]
    targeted = pd.concat(pieces, ignore_index=True)
    if targeted["cell_id"].duplicated().any():
        raise ValueError("Duplicated target cells")
    metadata = pd.read_csv(config["metadata"], sep="\t")
    cell_column = "cell_id" if "cell_id" in metadata else metadata.columns[0]
    canonical = metadata[[cell_column, config["label_field"], "is_ntc"]].rename(
        columns={cell_column: "cell_id", config["label_field"]: "target_label"})
    canonical = canonical.merge(targeted, on=["cell_id", "target_label", "is_ntc"], how="left")
    canonical.loc[canonical["is_ntc"], "reference_status"] = "control"
    missing = (~canonical["is_ntc"]) & canonical["reference_status"].isna()
    if missing.any():
        raise ValueError(f"Incomplete target-cell coverage: {int(missing.sum())}")
    columns = ["cell_id", "target_label", "is_ntc", "reference_status", "reference_score",
               "target_features_found", "target_features_total", "reference_threshold"]
    temporary = output / "target_expression_reference.tsv.partial"
    canonical[columns].to_csv(temporary, sep="\t", index=False)
    temporary.replace(output / "target_expression_reference.tsv")
    audit = {
        "cells": len(canonical), "ntc_cells": int(canonical["is_ntc"].sum()),
        "target_cells": int((~canonical["is_ntc"]).sum()),
        "scorable_target_cells": int(canonical.loc[~canonical["is_ntc"], "reference_score"].notna().sum()),
        "unscorable_target_cells": int(canonical.loc[~canonical["is_ntc"], "reference_score"].isna().sum()),
        "engaged_target_cells": int((canonical["reference_status"] == "engaged").sum()),
        "status": "PASS",
    }
    (output / "result_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    manifest = {"module": "metric_reference", "method": "Target-expression engagement reference",
                "dataset": config["dataset"], "shards": config["shards"], "max_workers": config["max_workers"],
                "numerical_threads_per_worker": 1, "wall_seconds": time.time() - started,
                "worker_metrics": metrics, "status": "PASS"}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
