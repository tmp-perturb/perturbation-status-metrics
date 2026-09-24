"""D1 descriptive cross-method agreement (not a performance benchmark)."""

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
from ..statistics import bootstrap_median_ci, bootstrap_spearman_ci, stable_seed


METHODS = ["mixscape", "mixscale", "ps"]
PAIRS = list(combinations(METHODS, 2))


def run(context: StageContext) -> tuple[list[Path], dict[str, Any]]:
    root = context.upstream("validate", "harmonized", "cell_scores")
    minimum = int(context.config["cohort"]["min_cells_per_perturbation"])
    resamples = int(context.config["statistics"]["bootstrap_resamples"])
    base_seed = int(context.config["statistics"]["seed"])
    cell_rows: list[dict[str, Any]] = []
    median_rows: list[dict[str, Any]] = []

    for dataset_id in context.dataset_ids:
        frame = pd.read_parquet(
            root / f"dataset_id={dataset_id}" / "part-000.parquet",
            columns=["cell_id", "target_id", "is_ntc", "perturbation_order", "method_id", "score", "native_status", "scorable"],
        )
        frame = frame.loc[~frame["is_ntc"]]
        score = frame.loc[frame["scorable"], ["cell_id", "target_id", "perturbation_order", "method_id", "score"]]
        for left, right in PAIRS:
            left_frame = score.loc[score["method_id"].eq(left), ["cell_id", "target_id", "perturbation_order", "score"]].rename(columns={"score": "score_left"})
            right_frame = score.loc[score["method_id"].eq(right), ["cell_id", "target_id", "score"]].rename(columns={"score": "score_right"})
            common = left_frame.merge(right_frame, on=["cell_id", "target_id"], validate="one_to_one")
            for (target_id, order), group in common.groupby(["target_id", "perturbation_order"], sort=False, observed=True):
                unique_left = group["score_left"].nunique()
                unique_right = group["score_right"].nunique()
                if len(group) < minimum:
                    status, rho = "insufficient_common_cells", np.nan
                elif unique_left < 2 or unique_right < 2:
                    status, rho = "constant_score", np.nan
                else:
                    rho = float(spearmanr(group["score_left"], group["score_right"]).statistic)
                    status = "scorable" if np.isfinite(rho) else "undefined_correlation"
                cell_rows.append({
                    "dataset_id": dataset_id, "target_id": target_id,
                    "perturbation_order": int(order), "method_left": left,
                    "method_right": right, "common_cells": len(group),
                    "unique_scores_left": unique_left, "unique_scores_right": unique_right,
                    "rho_cell": rho, "status": status,
                })

            medians = common.groupby(["target_id", "perturbation_order"], observed=True).agg(
                score_left=("score_left", "median"),
                score_right=("score_right", "median"),
                common_cells=("cell_id", "size"),
            ).reset_index()
            medians = medians[medians["common_cells"] >= minimum]
            for stratum, order in _strata(context.config["datasets"][dataset_id]):
                selected = _select_stratum(medians, order)
                estimate, low, high = bootstrap_spearman_ci(
                    selected["score_left"].to_numpy(), selected["score_right"].to_numpy(), resamples,
                    stable_seed(base_seed, "agreement_medians", dataset_id, stratum, left, right),
                )
                median_rows.append({
                    "dataset_id": dataset_id, "stratum": stratum,
                    "method_left": left, "method_right": right,
                    "common_perturbations": len(selected), "rho_target_median": estimate,
                    "ci_low": low, "ci_high": high,
                })

    per_target = pd.DataFrame(cell_rows)
    summary = _cell_summaries(per_target, context, resamples, base_seed)
    target_medians = pd.DataFrame(median_rows)
    outputs = [
        (per_target, "agreement_per_perturbation.tsv"),
        (summary, "agreement_cell_summary.tsv"),
        (target_medians, "agreement_target_medians.tsv"),
    ]
    artifacts = []
    for table, name in outputs:
        path = context.stage_dir / name
        atomic_write_tsv(table, path)
        artifacts.append(path)
    artifacts.extend(_plot(summary, target_medians, context.stage_dir))
    return artifacts, {
        "pair_perturbation_rows": len(per_target),
        "scorable_cell_correlations": int(per_target["rho_cell"].notna().sum()),
        "target_median_comparisons": len(target_medians),
    }


def _strata(dataset: dict[str, Any]) -> list[tuple[str, int | None]]:
    result = [("all", None)]
    if dataset.get("perturbation_delimiter"):
        result.extend([("single_gene", 1), ("multi_gene", 2)])
    return result


def _select_stratum(frame: pd.DataFrame, order: int | None) -> pd.DataFrame:
    if order is None:
        return frame
    return frame[frame["perturbation_order"].eq(1) if order == 1 else frame["perturbation_order"].gt(1)]


def _cell_summaries(frame: pd.DataFrame, context: StageContext, resamples: int, base_seed: int) -> pd.DataFrame:
    rows = []
    for dataset_id in context.dataset_ids:
        dataset_rows = frame[frame["dataset_id"].eq(dataset_id)]
        for stratum, order in _strata(context.config["datasets"][dataset_id]):
            selected = _select_stratum(dataset_rows, order)
            for left, right in PAIRS:
                pair = selected[selected["method_left"].eq(left) & selected["method_right"].eq(right)]
                valid = pair["rho_cell"].dropna().to_numpy()
                estimate, low, high = bootstrap_median_ci(valid, resamples, stable_seed(base_seed, "agreement_cells", dataset_id, stratum, left, right))
                rows.append({
                    "dataset_id": dataset_id, "stratum": stratum,
                    "method_left": left, "method_right": right,
                    "evaluated_perturbations": len(pair), "scorable_correlations": len(valid),
                    "median_rho_cell": estimate, "ci_low": low, "ci_high": high,
                })
    return pd.DataFrame(rows)


def _plot(cell_summary: pd.DataFrame, target_medians: pd.DataFrame, output: Path) -> list[Path]:
    datasets = list(dict.fromkeys(cell_summary["dataset_id"]))
    figure, axes = plt.subplots(2, len(datasets), figsize=(4.0 * len(datasets), 7.0), squeeze=False)
    for column, dataset_id in enumerate(datasets):
        for row, (table, value, title) in enumerate([
            (cell_summary, "median_rho_cell", "Median within-target cell rho"),
            (target_medians, "rho_target_median", "Target-median score rho"),
        ]):
            selected = table[table["dataset_id"].eq(dataset_id) & table["stratum"].eq("all")]
            matrix = np.eye(len(METHODS))
            for record in selected.itertuples():
                i, j = METHODS.index(record.method_left), METHODS.index(record.method_right)
                matrix[i, j] = matrix[j, i] = getattr(record, value)
            axis = axes[row, column]
            image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
            axis.set_xticks(range(len(METHODS)), METHODS, rotation=35, ha="right")
            axis.set_yticks(range(len(METHODS)), METHODS)
            axis.set_title(f"{dataset_id}\n{title}")
            for i in range(len(METHODS)):
                for j in range(len(METHODS)):
                    axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axes, fraction=0.02, pad=0.03, label="Spearman rho")
    png, pdf = output / "agreement.png", output / "agreement.pdf"
    atomic_savefig(figure, png, dpi=200, bbox_inches="tight")
    atomic_savefig(figure, pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]
