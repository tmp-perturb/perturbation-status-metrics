"""Shared benchmark runner and stage registry."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/perturb_status_benchmark_matplotlib")

from .config import load_config
from .context import StageContext
from .manifest import RunManifest
from .publish import publish
from .resources import PeakMemoryMonitor
from .stages import agreement, coverage, effect, embedding, harmonize, pseudobulk, report, reproducibility, resource_metrics, target_expression


STAGES = {
    "validate": harmonize.run,
    "coverage": coverage.run,
    "target_expression": target_expression.run,
    "embedding": embedding.run,
    "effect": effect.run,
    "pseudobulk": pseudobulk.run,
    "reproducibility": reproducibility.run,
    "agreement": agreement.run,
    "resources": resource_metrics.run,
    "report": report.run,
}
DEPENDENCIES = {
    "validate": [],
    "coverage": ["validate"],
    "target_expression": ["validate"],
    "embedding": ["validate"],
    "effect": ["validate", "embedding"],
    "pseudobulk": ["validate", "embedding"],
    "reproducibility": ["pseudobulk"],
    "agreement": ["validate"],
    "resources": ["validate"],
    "report": ["coverage", "target_expression", "effect", "reproducibility", "agreement", "resources"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal perturbation-status benchmark")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=[*STAGES, "all"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--datasets", nargs="+", help="optional dataset subset for a pilot run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="recompute the selected stage")
    parser.add_argument("--publish", action="store_true", help="atomically publish only after every stage passes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_run_id(args.run_id)
    config = load_config(args.config)
    dataset_ids = tuple(args.datasets or config["datasets"].keys())
    unknown = set(dataset_ids) - set(config["datasets"])
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    _set_numeric_threads(config["resources"]["numeric_threads_per_worker"])
    run_root = config["work_root"] / args.run_id
    manifest = RunManifest(run_root / "run_manifest.json", args.run_id, config, dataset_ids)
    selected = list(STAGES) if args.stage == "all" else [args.stage]
    for stage in selected:
        for dependency in DEPENDENCIES[stage]:
            if not manifest.complete(dependency):
                raise RuntimeError(f"Stage {stage} requires completed stage: {dependency}")
        if args.resume and manifest.complete(stage) and not args.force:
            print(f"[{stage}] already complete; reusing validated artifacts", flush=True)
            continue
        stage_dir = run_root / "stages" / stage
        if stage_dir.exists():
            if not args.force and not args.resume:
                raise RuntimeError(
                    f"Stage directory already exists: {stage_dir}; use --resume or --force"
                )
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True)
        manifest.start(stage)
        started = time.monotonic()
        memory_monitor = PeakMemoryMonitor()
        memory_monitor.start()
        try:
            context = StageContext(
                config=config,
                run_id=args.run_id,
                run_root=run_root,
                stage=stage,
                dataset_ids=dataset_ids,
            )
            artifacts, details = STAGES[stage](context)
            elapsed = time.monotonic() - started
            peak_rss_gib = memory_monitor.stop()
            manifest.finish(stage, artifacts, {
                **details,
                "wall_seconds": elapsed,
                "peak_process_tree_rss_gib": peak_rss_gib,
            })
            print(f"[{stage}] complete in {elapsed:.1f}s", flush=True)
        except Exception as error:
            memory_monitor.stop()
            manifest.fail(stage, f"{type(error).__name__}: {error}")
            raise
    if args.publish:
        destination = publish(config, args.run_id, run_root, list(STAGES))
        manifest.set_status("published", published_output=str(destination))
        shutil.copy2(manifest.path, destination / "manifest" / "run_manifest.json")
        print(f"[publish] complete: {destination}", flush=True)


def _validate_run_id(run_id: str) -> None:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if not run_id or any(character not in allowed for character in run_id):
        raise ValueError("run-id may contain only letters, numbers, dot, underscore and hyphen")


def _set_numeric_threads(threads: int) -> None:
    value = str(threads)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value
