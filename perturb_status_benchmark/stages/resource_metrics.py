"""R1 resource provenance and comparable execution-cost summaries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..context import StageContext
from ..io import atomic_savefig, atomic_write_tsv


METHODS = ["mixscape", "mixscale", "ps"]


def run(context: StageContext) -> tuple[list[Path], dict[str, Any]]:
    spec = context.config.get("resource_inputs")
    if not spec:
        raise ValueError("resources stage requires resource_inputs in config")
    root = context.config["project_root"]
    mixscale = pd.read_csv(_path(root, spec["mixscale_manifest"]), sep="\t").set_index("dataset")
    ps = pd.read_csv(_path(root, spec["ps_manifest"]), sep="\t").set_index("dataset")
    events: list[dict[str, Any]] = []

    for dataset_id in context.dataset_ids:
        dataset_spec = spec["datasets"][dataset_id]
        shared = _shared_preprocessing(_path(root, dataset_spec["shared_preprocessing_events"]))
        build = _gnu_time(_path(root, dataset_spec["shard_build_time"]))
        events.extend([
            _event(dataset_id, "shared", "shared_preprocessing", **shared, source_path=dataset_spec["shared_preprocessing_events"]),
            _event(dataset_id, "shared", "shard_materialization", **build, source_path=dataset_spec["shard_build_time"]),
            _event(dataset_id, "mixscape", "scoring_success", wall_seconds=dataset_spec["mixscape_wall_seconds"], peak_rss_gib=dataset_spec["mixscape_peak_worker_rss_gib"], rss_type="peak_worker", shards=dataset_spec["mixscape_shards"], workers=dataset_spec["mixscape_workers"], source_path="handoff.md#M03-recorded-resources"),
        ])
        m = mixscale.loc[dataset_id]
        events.append(_event(dataset_id, "mixscale", "scoring_success", wall_seconds=m.wall_time_seconds, peak_rss_gib=m.peak_worker_rss_bytes / 2**30, rss_type="peak_worker", shards=m.shards, workers=m.max_workers, source_path=spec["mixscale_manifest"]))
        p = ps.loc[dataset_id]
        if pd.notna(p.input_build_wall_seconds):
            events.append(_event(dataset_id, "ps", "input_build", wall_seconds=p.input_build_wall_seconds, peak_rss_gib=p.input_build_peak_rss_gib, rss_type="peak_process", source_path=spec["ps_manifest"]))
        else:
            events.append(_event(dataset_id, "ps", "input_build", source_path=spec["ps_manifest"], note="not recorded; standalone total unavailable"))
        events.append(_event(dataset_id, "ps", "scoring_success", wall_seconds=p.successful_attempt_wall_seconds, peak_rss_gib=p.analysis_peak_single_process_rss_gib, rss_type="peak_process", concurrent_rss_upper_bound_gib=p.analysis_concurrent_rss_upper_bound_gib, shards=p.shards, workers=p.max_workers, source_path=spec["ps_manifest"], execution_mode=p.execution_mode))
        if float(p.failed_attempt_wall_seconds) > 0:
            events.append(_event(dataset_id, "ps", "scoring_failed", wall_seconds=p.failed_attempt_wall_seconds, source_path=spec["ps_manifest"], note="terminated attempt retained in operational cost"))

    event_frame = pd.DataFrame(events)
    summary = _summarize(event_frame, context)
    event_path = context.stage_dir / "runtime_events.tsv"
    summary_path = context.stage_dir / "resource_summary.tsv"
    atomic_write_tsv(event_frame, event_path)
    atomic_write_tsv(summary, summary_path)
    figures = _plot(summary, context.stage_dir)
    return [event_path, summary_path, *figures], {
        "runtime_events": len(event_frame),
        "dataset_method_summaries": len(summary),
        "complete_standalone_wall_times": int(summary["standalone_wall_seconds"].notna().sum()),
    }


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _event(dataset_id: str, owner: str, event: str, wall_seconds: float = np.nan, peak_rss_gib: float = np.nan, rss_type: str = "unavailable", concurrent_rss_upper_bound_gib: float = np.nan, shards: float = np.nan, workers: float = np.nan, source_path: str = "", execution_mode: str = "", note: str = "") -> dict[str, Any]:
    return {"dataset_id": dataset_id, "owner": owner, "event": event, "wall_seconds": wall_seconds, "peak_rss_gib": peak_rss_gib, "rss_type": rss_type, "concurrent_rss_upper_bound_gib": concurrent_rss_upper_bound_gib, "shards": shards, "workers": workers, "numeric_threads_per_worker": 1, "execution_mode": execution_mode, "source_path": source_path, "note": note}


def _shared_preprocessing(path: Path) -> dict[str, float | str]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    start = next(r for r in records if r["stage"] == "input_read" and r["event"] == "start")
    end = next(r for r in records if r["stage"] == "checkpoint_B" and r["event"] == "end")
    selected = [r for r in records if start["epoch"] <= r["epoch"] <= end["epoch"]]
    peak = max(float(r.get("process", {}).get("VmHWM", np.nan)) for r in selected) / 2**30
    return {"wall_seconds": end["epoch"] - start["epoch"], "peak_rss_gib": peak, "rss_type": "peak_process"}


def _gnu_time(path: Path) -> dict[str, float | str]:
    text = path.read_text()
    elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    if not elapsed or not rss:
        raise ValueError(f"Incomplete GNU time record: {path}")
    parts = [float(value) for value in elapsed.group(1).split(":")]
    seconds = parts[-1] + (parts[-2] * 60 if len(parts) >= 2 else 0) + (parts[-3] * 3600 if len(parts) == 3 else 0)
    return {"wall_seconds": seconds, "peak_rss_gib": int(rss.group(1)) / 2**20, "rss_type": "peak_process"}


def _summarize(events: pd.DataFrame, context: StageContext) -> pd.DataFrame:
    rows = []
    for dataset_id in context.dataset_ids:
        data = events[events["dataset_id"].eq(dataset_id)]
        shared = data[data["owner"].eq("shared")]
        shared_complete = shared["wall_seconds"].notna().all()
        for method in METHODS:
            own = data[data["owner"].eq(method)]
            success = own[own["event"].eq("scoring_success")]
            incremental = success["wall_seconds"].sum(min_count=1)
            if method == "ps":
                input_build = own[own["event"].eq("input_build")]
                standalone = own.loc[own["event"].isin(["input_build", "scoring_success"]), "wall_seconds"].sum(min_count=2)
                components = own[own["event"].isin(["input_build", "scoring_success"])]
            else:
                standalone = shared["wall_seconds"].sum() + incremental if shared_complete and pd.notna(incremental) else np.nan
                components = pd.concat([shared, success])
            failed = own.loc[own["event"].eq("scoring_failed"), "wall_seconds"].sum(min_count=1)
            operational = standalone + failed if pd.notna(standalone) and pd.notna(failed) else standalone
            result_path = context.config["datasets"][dataset_id]["methods"][method]["path"]
            rows.append({
                "dataset_id": dataset_id, "method_id": method,
                "incremental_wall_seconds": incremental,
                "standalone_wall_seconds": standalone,
                "operational_wall_seconds_all_attempts": operational,
                "standalone_peak_component_rss_gib": components["peak_rss_gib"].max(),
                "peak_rss_interpretation": "maximum recorded component; not an observed whole-workflow peak",
                "concurrent_rss_upper_bound_gib": own["concurrent_rss_upper_bound_gib"].max(),
                "canonical_result_bytes": result_path.stat().st_size,
                "standalone_complete": bool(pd.notna(standalone)),
            })
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, output: Path) -> list[Path]:
    datasets = list(dict.fromkeys(summary["dataset_id"]))
    figure, axes = plt.subplots(1, len(datasets), figsize=(4.1 * len(datasets), 4.1), sharey=True)
    axes = np.atleast_1d(axes)
    colors = ["#0072B2", "#CC79A7", "#009E73"]
    for axis, dataset_id in zip(axes, datasets):
        selected = summary[summary["dataset_id"].eq(dataset_id)].set_index("method_id").reindex(METHODS)
        values = selected["standalone_wall_seconds"].to_numpy() / 60
        axis.bar(range(len(METHODS)), values, color=colors)
        axis.set_yscale("log")
        axis.set_xticks(range(len(METHODS)), METHODS, rotation=35, ha="right")
        axis.set_title(dataset_id)
    axes[0].set_ylabel("Standalone wall time (minutes, log scale)")
    figure.suptitle("Recorded end-to-end execution cost")
    figure.tight_layout()
    png, pdf = output / "resources.png", output / "resources.pdf"
    atomic_savefig(figure, png, dpi=200)
    atomic_savefig(figure, pdf)
    plt.close(figure)
    return [png, pdf]
