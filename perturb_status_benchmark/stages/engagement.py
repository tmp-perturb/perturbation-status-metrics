"""B2 within-perturbation target-expression correlation benchmark."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..context import StageContext
from ..io import atomic_savefig, atomic_write_tsv
from ..statistics import bootstrap_median_ci, stable_seed


EVALUATED_METHODS = ["mixscape", "mixscale", "ps"]


def run(context: StageContext) -> tuple[list[Path], dict[str, Any]]:
    root = context.upstream("validate", "harmonized", "cell_scores")
    if not root.exists():
        raise RuntimeError("engagement requires a completed validate stage")
    minimum = int(context.config["cohort"]["min_cells_per_perturbation"])
    bootstrap = int(context.config["statistics"]["bootstrap_resamples"])
    base_seed = int(context.config["statistics"]["seed"])
    records: list[dict[str, Any]] = []

    for dataset_id in context.dataset_ids:
        path = root / f"dataset_id={dataset_id}" / "part-000.parquet"
        frame = pd.read_parquet(
            path,
            columns=["cell_id", "target_id", "is_ntc", "perturbation_order", "method_id", "score", "scorable"],
        )
        frame = frame.loc[~frame["is_ntc"]]
        reference_spec = context.config["datasets"][dataset_id]["target_expression_reference"]
        reference = pd.read_csv(reference_spec["path"], sep="\t", usecols=[
            "cell_id", reference_spec["target_column"], reference_spec["score_column"],
            reference_spec["status_column"],
        ]).rename(columns={
            reference_spec["target_column"]: "target_id",
            reference_spec["score_column"]: "reference_score",
        })
        allowed = reference_spec["scorable_status"]
        if not isinstance(allowed, list):
            allowed = [allowed]
        reference = reference.loc[reference[reference_spec["status_column"]].isin(allowed), [
            "cell_id", "target_id", "reference_score",
        ]]
        order = frame[["cell_id", "target_id", "perturbation_order"]].drop_duplicates()
        reference = reference.merge(order, on=["cell_id", "target_id"], validate="one_to_one")
        for method_id in EVALUATED_METHODS:
            method = frame.loc[
                (frame["method_id"] == method_id) & frame["scorable"],
                ["cell_id", "target_id", "score"],
            ].rename(columns={"score": "method_score"})
            common = reference.merge(
                method, on=["cell_id", "target_id"], how="inner", validate="one_to_one"
            )
            for (target_id, order), group in common.groupby(
                ["target_id", "perturbation_order"], sort=False, observed=True
            ):
                n_cells = len(group)
                method_unique = group["method_score"].nunique(dropna=True)
                reference_unique = group["reference_score"].nunique(dropna=True)
                if n_cells < minimum:
                    status, rho = "insufficient_common_cells", np.nan
                elif method_unique < 2:
                    status, rho = "constant_method_score", np.nan
                elif reference_unique < 2:
                    status, rho = "constant_target_expression_score", np.nan
                else:
                    rho = float(spearmanr(group["method_score"], group["reference_score"]).statistic)
                    status = "scorable" if np.isfinite(rho) else "undefined_correlation"
                records.append({
                    "dataset_id": dataset_id,
                    "method_id": method_id,
                    "target_id": target_id,
                    "perturbation_order": int(order),
                    "common_cells": n_cells,
                    "method_unique_scores": method_unique,
                    "reference_unique_scores": reference_unique,
                    "rho_engagement": rho,
                    "status": status,
                })

    per_target = pd.DataFrame(records)
    summaries = _summaries(per_target, context.config, context.dataset_ids, bootstrap, base_seed)
    pairwise = _pairwise(per_target, context.config, context.dataset_ids, bootstrap, base_seed)
    per_target_path = context.stage_dir / "engagement_per_perturbation.tsv"
    summary_path = context.stage_dir / "engagement_summary.tsv"
    pairwise_path = context.stage_dir / "engagement_pairwise.tsv"
    atomic_write_tsv(per_target, per_target_path)
    atomic_write_tsv(summaries, summary_path)
    atomic_write_tsv(pairwise, pairwise_path)
    figure_paths = _plot(per_target, context.stage_dir)
    artifacts = [per_target_path, summary_path, pairwise_path, *figure_paths]
    return artifacts, {
        "perturbation_method_rows": len(per_target),
        "scorable_correlations": int(per_target["rho_engagement"].notna().sum()),
    }


def _strata(dataset: dict[str, Any]) -> list[tuple[str, int | None]]:
    result: list[tuple[str, int | None]] = [("all", None)]
    if dataset.get("perturbation_delimiter"):
        result.extend([("single_gene", 1), ("multi_gene", 2)])
    return result


def _select_stratum(frame: pd.DataFrame, order: int | None) -> pd.DataFrame:
    if order is None:
        return frame
    return frame[frame["perturbation_order"].eq(1) if order == 1 else frame["perturbation_order"].gt(1)]


def _summaries(
    frame: pd.DataFrame, config: dict[str, Any], dataset_ids: tuple[str, ...], resamples: int, base_seed: int
) -> pd.DataFrame:
    rows = []
    for dataset_id in dataset_ids:
        dataset = config["datasets"][dataset_id]
        dataset_rows = frame[frame["dataset_id"] == dataset_id]
        for stratum, order in _strata(dataset):
            selected = _select_stratum(dataset_rows, order)
            for method_id in EVALUATED_METHODS:
                method = selected[selected["method_id"] == method_id]
                valid = method["rho_engagement"].dropna().to_numpy()
                estimate, low, high = bootstrap_median_ci(
                    valid, resamples, stable_seed(base_seed, "engagement", dataset_id, stratum, method_id)
                )
                rows.append({
                    "dataset_id": dataset_id,
                    "stratum": stratum,
                    "method_id": method_id,
                    "evaluated_perturbations": len(method),
                    "scorable_correlations": len(valid),
                    "median_rho": estimate,
                    "ci_low": low,
                    "ci_high": high,
                })
    return pd.DataFrame(rows)


def _pairwise(
    frame: pd.DataFrame, config: dict[str, Any], dataset_ids: tuple[str, ...], resamples: int, base_seed: int
) -> pd.DataFrame:
    rows = []
    for dataset_id in dataset_ids:
        dataset = config["datasets"][dataset_id]
        dataset_rows = frame[frame["dataset_id"] == dataset_id]
        for stratum, order in _strata(dataset):
            selected = _select_stratum(dataset_rows, order)
            wide = selected.pivot(index="target_id", columns="method_id", values="rho_engagement")
            for left, right in combinations(EVALUATED_METHODS, 2):
                common = wide[[left, right]].dropna()
                difference = (common[left] - common[right]).to_numpy()
                estimate, low, high = bootstrap_median_ci(
                    difference,
                    resamples,
                    stable_seed(base_seed, "engagement_difference", dataset_id, stratum, left, right),
                )
                rows.append({
                    "dataset_id": dataset_id,
                    "stratum": stratum,
                    "method_left": left,
                    "method_right": right,
                    "common_perturbations": len(common),
                    "median_rho_difference": estimate,
                    "ci_low": low,
                    "ci_high": high,
                })
    return pd.DataFrame(rows)


def _plot(frame: pd.DataFrame, output: Path) -> list[Path]:
    datasets = list(dict.fromkeys(frame["dataset_id"]))
    colors = {"mixscape": "#0072B2", "mixscale": "#CC79A7", "ps": "#009E73"}
    figure, axes = plt.subplots(1, len(datasets), figsize=(11.5, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(20260913)
    for axis, dataset_id in zip(axes, datasets):
        dataset = frame[(frame["dataset_id"] == dataset_id) & frame["rho_engagement"].notna()]
        values = [dataset.loc[dataset["method_id"] == method, "rho_engagement"].to_numpy() for method in EVALUATED_METHODS]
        box = axis.boxplot(values, positions=np.arange(3), widths=0.5, patch_artist=True, showfliers=False, medianprops={"color": "black"})
        for patch, method in zip(box["boxes"], EVALUATED_METHODS):
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.55)
        for index, (method, method_values) in enumerate(zip(EVALUATED_METHODS, values)):
            if len(method_values) <= 300:
                axis.scatter(index + rng.uniform(-0.12, 0.12, len(method_values)), method_values, s=5, color=colors[method], alpha=0.22, linewidths=0)
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.set_xticks(np.arange(3), EVALUATED_METHODS, rotation=20, ha="right")
        axis.set_title(dataset_id)
    axes[0].set_ylabel("Within-perturbation Spearman rho")
    figure.suptitle("Within-perturbation target-expression correlation")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    png = output / "engagement.png"
    pdf = output / "engagement.pdf"
    atomic_savefig(figure, png, dpi=200)
    atomic_savefig(figure, pdf)
    plt.close(figure)
    return [png, pdf]
