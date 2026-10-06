"""B3 transcriptomic effect magnitude and method association."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr

from ..context import StageContext
from ..io import atomic_write_tsv
from ..statistics import bootstrap_spearman_ci, stable_seed
from .harmonize import METHOD_ORDER


def run(context: StageContext) -> tuple[list[Path], dict[str, Any]]:
    settings = context.config.get("energy_distance", {})
    max_target = int(settings.get("max_target_cells", 256))
    max_ntc = int(settings.get("ntc_cells", 512))
    seed = int(settings.get("seed", context.config["statistics"]["seed"]))
    resamples = int(context.config["statistics"]["bootstrap_resamples"])
    minimum = int(context.config["cohort"]["min_cells_per_perturbation"])
    effect_records: list[dict[str, Any]] = []
    score_records: list[pd.DataFrame] = []

    for dataset_id in context.dataset_ids:
        embedding_path = context.upstream(
            "embedding", "expression", f"dataset_id={dataset_id}", "embedding.parquet"
        )
        if not embedding_path.exists():
            raise RuntimeError(f"effect requires embedding for {dataset_id}")
        embedding = pd.read_parquet(embedding_path)
        pc_columns = [column for column in embedding if column.startswith("pc_")]
        ntc = embedding[embedding["is_ntc"]]
        ntc_indices = _stratified_sample(
            ntc,
            max_ntc,
            stable_seed(seed, "energy_ntc", dataset_id),
        )
        ntc_coordinates = embedding.loc[ntc_indices, pc_columns].to_numpy(dtype=np.float32)
        ntc_within = _mean_within_distance(ntc_coordinates)
        targeted = embedding[~embedding["is_ntc"]]
        for target_id, indices in targeted.groupby("target_id", sort=True).groups.items():
            target_frame = embedding.loc[np.asarray(indices)]
            sampled_indices = _stratified_sample(
                target_frame,
                max_target,
                stable_seed(seed, "energy_target", dataset_id, str(target_id)),
            )
            coordinates = embedding.loc[sampled_indices, pc_columns].to_numpy(dtype=np.float32)
            cross = float(cdist(coordinates, ntc_coordinates, metric="euclidean").mean())
            energy = 2 * cross - _mean_within_distance(coordinates) - ntc_within
            effect_records.append({
                "dataset_id": dataset_id,
                "target_id": target_id,
                "targeted_cells": len(target_frame),
                "sampled_targeted_cells": len(sampled_indices),
                "sampled_ntc_cells": len(ntc_indices),
                "energy_distance": max(float(energy), 0.0),
            })

        scores_path = context.upstream(
            "validate", "harmonized", "cell_scores", f"dataset_id={dataset_id}", "part-000.parquet"
        )
        scores = pd.read_parquet(
            scores_path,
            columns=["target_id", "is_ntc", "perturbation_order", "method_id", "score", "scorable"],
        )
        scores = scores[(~scores["is_ntc"]) & scores["scorable"]]
        medians = (
            scores.groupby(["target_id", "perturbation_order", "method_id"], observed=True)["score"]
            .median()
            .rename("median_score")
            .reset_index()
        )
        medians["dataset_id"] = dataset_id
        score_records.append(medians)

    effects = pd.DataFrame(effect_records)
    score_medians = pd.concat(score_records, ignore_index=True)
    combined = score_medians.merge(effects, on=["dataset_id", "target_id"], how="left", validate="many_to_one")
    summaries = _summaries(combined, context, minimum, resamples, seed)
    pairwise = _paired_differences(combined, context, minimum, resamples, seed)

    effect_path = context.stage_dir / "transcriptomic_effect.tsv"
    median_path = context.stage_dir / "perturbation_score_medians.tsv"
    association_path = context.stage_dir / "effect_association.tsv"
    pairwise_path = context.stage_dir / "effect_association_pairwise.tsv"
    atomic_write_tsv(effects, effect_path)
    atomic_write_tsv(score_medians, median_path)
    atomic_write_tsv(summaries, association_path)
    atomic_write_tsv(pairwise, pairwise_path)
    artifacts = [effect_path, median_path, association_path, pairwise_path]
    return artifacts, {
        "perturbations": len(effects),
        "max_target_sample": max_target,
        "ntc_sample": max_ntc,
    }


def _stratified_sample(frame: pd.DataFrame, maximum: int, seed: int) -> np.ndarray:
    if len(frame) <= maximum:
        return frame.index.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    groups = [(batch, np.asarray(indices, dtype=np.int64)) for batch, indices in frame.groupby("batch_id", sort=True).groups.items()]
    sizes = np.asarray([len(indices) for _, indices in groups], dtype=float)
    raw = sizes / sizes.sum() * maximum
    allocation = np.floor(raw).astype(int)
    allocation[(allocation == 0) & (sizes > 0)] = 1
    while allocation.sum() > maximum:
        candidates = np.flatnonzero(allocation > 1)
        allocation[candidates[np.argmax(allocation[candidates] - raw[candidates])]] -= 1
    while allocation.sum() < maximum:
        capacity = sizes - allocation
        candidates = np.flatnonzero(capacity > 0)
        allocation[candidates[np.argmax(raw[candidates] - allocation[candidates])]] += 1
    selected = [rng.choice(indices, size=count, replace=False) for (_, indices), count in zip(groups, allocation)]
    return np.sort(np.concatenate(selected)).astype(np.int64)


def _mean_within_distance(coordinates: np.ndarray) -> float:
    count = len(coordinates)
    if count < 2:
        return 0.0
    return float(2 * pdist(coordinates, metric="euclidean").sum() / (count * count))


def _strata(dataset: dict[str, Any]) -> list[tuple[str, int | None]]:
    result: list[tuple[str, int | None]] = [("all", None)]
    if dataset.get("perturbation_delimiter"):
        result.extend([("single_gene", 1), ("multi_gene", 2)])
    return result


def _select(frame: pd.DataFrame, order: int | None) -> pd.DataFrame:
    if order is None:
        return frame
    return frame[frame["perturbation_order"].eq(1) if order == 1 else frame["perturbation_order"].gt(1)]


def _summaries(
    frame: pd.DataFrame,
    context: StageContext,
    minimum: int,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for dataset_id in context.dataset_ids:
        dataset = context.config["datasets"][dataset_id]
        base = frame[(frame["dataset_id"] == dataset_id) & (frame["targeted_cells"] >= minimum)]
        for stratum, order in _strata(dataset):
            selected = _select(base, order)
            for method_id in METHOD_ORDER:
                method = selected[selected["method_id"] == method_id].dropna(subset=["median_score", "energy_distance"])
                estimate, low, high = bootstrap_spearman_ci(
                    method["median_score"].to_numpy(),
                    method["energy_distance"].to_numpy(),
                    resamples,
                    stable_seed(seed, "effect", dataset_id, stratum, method_id),
                )
                rows.append({
                    "dataset_id": dataset_id,
                    "stratum": stratum,
                    "method_id": method_id,
                    "perturbations": len(method),
                    "spearman_rho": estimate,
                    "ci_low": low,
                    "ci_high": high,
                })
    return pd.DataFrame(rows)


def _paired_differences(
    frame: pd.DataFrame,
    context: StageContext,
    minimum: int,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng_seed = int(seed)
    for dataset_id in context.dataset_ids:
        dataset = context.config["datasets"][dataset_id]
        base = frame[(frame["dataset_id"] == dataset_id) & (frame["targeted_cells"] >= minimum)]
        for stratum, order in _strata(dataset):
            selected = _select(base, order)
            scores = selected.pivot(index="target_id", columns="method_id", values="median_score")
            energy = selected.drop_duplicates("target_id").set_index("target_id")["energy_distance"]
            for left, right in combinations(METHOD_ORDER, 2):
                common = scores[[left, right]].dropna().join(energy, how="inner").dropna()
                if len(common) < 3:
                    estimate = low = high = np.nan
                else:
                    left_rho = float(spearmanr(common[left], common["energy_distance"]).statistic)
                    right_rho = float(spearmanr(common[right], common["energy_distance"]).statistic)
                    estimate = left_rho - right_rho
                    rng = np.random.default_rng(stable_seed(rng_seed, "effect_difference", dataset_id, stratum, left, right))
                    boot = []
                    for _ in range(resamples):
                        indices = rng.integers(0, len(common), len(common))
                        sample = common.iloc[indices]
                        lr = float(spearmanr(sample[left], sample["energy_distance"]).statistic)
                        rr = float(spearmanr(sample[right], sample["energy_distance"]).statistic)
                        if np.isfinite(lr) and np.isfinite(rr):
                            boot.append(lr - rr)
                    low, high = np.quantile(boot, [0.025, 0.975]) if boot else (np.nan, np.nan)
                rows.append({
                    "dataset_id": dataset_id,
                    "stratum": stratum,
                    "method_left": left,
                    "method_right": right,
                    "common_perturbations": len(common),
                    "rho_difference": estimate,
                    "ci_low": low,
                    "ci_high": high,
                })
    return pd.DataFrame(rows)
