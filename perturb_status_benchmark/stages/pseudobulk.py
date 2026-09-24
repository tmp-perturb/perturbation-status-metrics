"""Build weighted and matched unweighted reagent-level perturbation profiles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from ..context import StageContext
from ..expression import ExpressionSource
from ..io import atomic_write_json, atomic_write_parquet
from .harmonize import METHOD_ORDER


def run(context: StageContext) -> tuple[list[Path], dict[str, Any]]:
    settings = context.config.get("pseudobulk")
    if not settings:
        raise ValueError("The selected config does not define pseudobulk settings")
    artifacts: list[Path] = []
    details: dict[str, Any] = {"datasets": {}}
    for dataset_id in context.dataset_ids:
        current, info = _build_dataset(context, dataset_id, settings)
        artifacts.extend(current)
        details["datasets"][dataset_id] = info
    return artifacts, details


def _build_dataset(
    context: StageContext, dataset_id: str, settings: dict[str, Any]
) -> tuple[list[Path], dict[str, Any]]:
    dataset = context.config["datasets"][dataset_id]
    reproducibility = dataset.get("reproducibility", {})
    if not reproducibility.get("enabled", True):
        dataset_dir = context.stage_dir / "pseudobulk" / f"dataset_id={dataset_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = dataset_dir / "pseudobulk_metadata.json"
        atomic_write_json(metadata_path, {
            "dataset_id": dataset_id,
            "status": reproducibility.get("status", "not_applicable"),
            "reason": reproducibility.get("reason", "Dataset excluded from construct reproducibility"),
            "profile_unit": "construct_id",
        })
        return [metadata_path], {"status": reproducibility.get("status", "not_applicable"), "groups": 0}
    source = ExpressionSource(
        path=dataset["h5ad"],
        counts_layer=context.config["expression"]["counts_layer"],
        feature_mode=dataset["expression"]["feature_mode"],
    )
    eligible = set(context.config["cohort"]["eligible_assignment_structures"])
    metadata, eligible_mask = source.metadata(dataset["metadata"], eligible)
    genes = source.feature_ids()
    feature_source = context.upstream(
        "embedding", "expression", f"dataset_id={dataset_id}", "feature_manifest.parquet"
    )
    feature_stats = pd.read_parquet(feature_source)
    if not np.array_equal(feature_stats["feature_id"].astype(str).to_numpy(), genes):
        raise ValueError(f"{dataset_id}: expression feature manifest does not match source")
    profile_mask = feature_stats["ntc_detection_fraction"].to_numpy() >= float(
        settings["min_ntc_detection_fraction"]
    )
    feature_indices = np.flatnonzero(profile_mask)
    profile_genes = genes[feature_indices]
    profile_symbols = _feature_symbols(dataset, profile_genes)

    score_path = context.upstream(
        "validate", "harmonized", "cell_scores", f"dataset_id={dataset_id}", "part-000.parquet"
    )
    scores = pd.read_parquet(
        score_path,
        columns=["cell_id", "target_id", "is_ntc", "perturbation_order", "method_id", "score", "scorable"],
    )
    score_index = {cell_id: index for index, cell_id in enumerate(metadata["cell_id"])}
    min_construct = int(context.config["cohort"].get(
        "min_cells_per_construct", context.config["cohort"]["min_cells_per_reagent"]
    ))
    guide_sets = (
        metadata.loc[~metadata["is_ntc"]]
        .groupby(["target_id", "construct_id"], observed=True)["reagent_id"]
        .agg(lambda values: tuple(sorted(set(map(str, values)))))
        .to_dict()
    )
    group_records: list[dict[str, Any]] = []
    cell_groups: dict[str, np.ndarray] = {}
    cell_weights: dict[str, np.ndarray] = {}

    for method_id in METHOD_ORDER:
        method = scores[(scores["method_id"] == method_id) & (~scores["is_ntc"]) & scores["scorable"]].copy()
        method["cell_position"] = method["cell_id"].map(score_index)
        if method["cell_position"].isna().any():
            raise ValueError(f"{dataset_id}/{method_id}: scored cell missing from expression cohort")
        method["cell_position"] = method["cell_position"].astype(np.int64)
        method["weight"] = method.groupby("target_id", observed=True)["score"].rank(method="average", pct=True)
        method = method.merge(
            metadata[["cell_id", "construct_id", "batch_id"]], on="cell_id", how="left", validate="one_to_one"
        )
        construct_counts = method.groupby(["target_id", "construct_id"], observed=True).size()
        qualifying = construct_counts[construct_counts >= min_construct].reset_index(name="cells")
        target_constructs = qualifying.groupby("target_id", observed=True).size()
        valid_targets = set(target_constructs[target_constructs >= 2].index)
        qualifying = qualifying[qualifying["target_id"].isin(valid_targets)]
        valid_pairs = set(zip(qualifying["target_id"], qualifying["construct_id"]))
        method = method[
            [(target, construct) in valid_pairs for target, construct in zip(method["target_id"], method["construct_id"])]
        ].copy()
        group_lookup = {}
        for (target_id, construct_id), group in method.groupby(["target_id", "construct_id"], sort=True, observed=True):
            group_id = len(group_records)
            group_lookup[(target_id, construct_id)] = group_id
            weights = group["weight"].to_numpy(dtype=float)
            guides = guide_sets[(target_id, construct_id)]
            group_records.append({
                "profile_row": group_id,
                "dataset_id": dataset_id,
                "method_id": method_id,
                "target_id": target_id,
                "construct_id": construct_id,
                "guide_set": "|".join(guides),
                "guide_count": len(guides),
                "perturbation_order": int(group["perturbation_order"].iloc[0]),
                "cells": len(group),
                "weight_sum": float(weights.sum()),
                "effective_weighted_cells": float(weights.sum() ** 2 / np.square(weights).sum()),
            })
        groups = np.full(len(metadata), -1, dtype=np.int32)
        weights = np.zeros(len(metadata), dtype=np.float32)
        if len(method):
            positions = method["cell_position"].to_numpy(dtype=np.int64)
            groups[positions] = [group_lookup[(target, construct)] for target, construct in zip(method["target_id"], method["construct_id"])]
            weights[positions] = method["weight"].to_numpy(dtype=np.float32)
        cell_groups[method_id] = groups
        cell_weights[method_id] = weights

    groups = pd.DataFrame(group_records)
    n_groups, n_genes = len(groups), len(feature_indices)
    weighted_counts = np.zeros((n_groups, n_genes), dtype=np.float32)
    unweighted_counts = np.zeros((n_groups, n_genes), dtype=np.float32)
    batch_values = sorted(metadata["batch_id"].astype(str).unique())
    batch_codes = pd.Categorical(metadata["batch_id"].astype(str), categories=batch_values).codes
    n_batches = len(batch_values)
    ntc_counts = np.zeros((n_batches, n_genes), dtype=np.float64)
    weighted_batch_mass = np.zeros((n_groups, n_batches), dtype=np.float32)
    unweighted_batch_mass = np.zeros((n_groups, n_batches), dtype=np.float32)
    eligible_lookup = np.full(len(eligible_mask), -1, dtype=np.int64)
    eligible_lookup[np.flatnonzero(eligible_mask)] = np.arange(len(metadata))

    for original_rows, matrix in source.iter_counts(
        eligible_mask, int(context.config["expression"]["row_chunk_size"])
    ):
        eligible_rows = eligible_lookup[original_rows]
        block = matrix[:, feature_indices]
        ntc_local = metadata.loc[eligible_rows, "is_ntc"].to_numpy()
        if ntc_local.any():
            _accumulate_rows(
                ntc_counts,
                batch_codes[eligible_rows][ntc_local],
                block[ntc_local],
                np.ones(int(ntc_local.sum()), dtype=np.float32),
            )
        for method_id in METHOD_ORDER:
            group_ids = cell_groups[method_id][eligible_rows]
            selected = group_ids >= 0
            if not selected.any():
                continue
            selected_groups = group_ids[selected]
            selected_block = block[selected]
            selected_weights = cell_weights[method_id][eligible_rows][selected]
            _accumulate_rows(weighted_counts, selected_groups, selected_block, selected_weights)
            _accumulate_rows(
                unweighted_counts,
                selected_groups,
                selected_block,
                np.ones(len(selected_groups), dtype=np.float32),
            )
            np.add.at(weighted_batch_mass, (selected_groups, batch_codes[eligible_rows][selected]), selected_weights)
            np.add.at(unweighted_batch_mass, (selected_groups, batch_codes[eligible_rows][selected]), 1.0)

    ntc_library = ntc_counts.sum(axis=1)
    if np.any(ntc_library <= 0):
        raise ValueError(f"{dataset_id}: at least one batch has zero NTC profile counts")
    ntc_logcpm = np.log1p(ntc_counts * (1e6 / ntc_library)[:, None]).astype(np.float32)
    weighted_profiles = _to_profiles(weighted_counts, weighted_batch_mass, ntc_logcpm)
    unweighted_profiles = _to_profiles(unweighted_counts, unweighted_batch_mass, ntc_logcpm)

    dataset_dir = context.stage_dir / "pseudobulk" / f"dataset_id={dataset_id}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    weighted_path = dataset_dir / "weighted_profiles.npy"
    unweighted_path = dataset_dir / "unweighted_profiles.npy"
    _atomic_save_array(weighted_path, weighted_profiles)
    _atomic_save_array(unweighted_path, unweighted_profiles)
    groups_path = dataset_dir / "group_manifest.parquet"
    atomic_write_parquet(groups, groups_path)
    features_path = dataset_dir / "profile_features.parquet"
    atomic_write_parquet(
        pd.DataFrame({
            "profile_column": np.arange(n_genes),
            "feature_id": profile_genes,
            "feature_symbol": profile_symbols,
            "ntc_detection_fraction": feature_stats.loc[profile_mask, "ntc_detection_fraction"].to_numpy(),
        }),
        features_path,
    )
    metadata_path = dataset_dir / "pseudobulk_metadata.json"
    atomic_write_json(metadata_path, {
        "dataset_id": dataset_id,
        "groups": n_groups,
        "features": n_genes,
        "batches": batch_values,
        "min_cells_per_construct": min_construct,
        "weighting": "within_target_average_rank_percentile",
        "reference": "batch_composition_matched_eligible_ntc_log1p_cpm",
        "profile_unit": "construct_id",
        "guide_set_source": settings.get("guide_set_source", "observed_eligible_cells"),
        "require_disjoint_guide_sets": bool(settings.get("require_disjoint_guide_sets", True)),
    })
    return [weighted_path, unweighted_path, groups_path, features_path, metadata_path], {
        "groups": n_groups,
        "features": n_genes,
        "weighted_profile_bytes": weighted_path.stat().st_size,
        "unweighted_profile_bytes": unweighted_path.stat().st_size,
    }


def _accumulate_rows(
    destination: np.ndarray,
    group_ids: np.ndarray,
    matrix: sparse.csr_matrix,
    weights: np.ndarray,
) -> None:
    unique, inverse = np.unique(group_ids, return_inverse=True)
    assignment = sparse.csr_matrix(
        (weights, (inverse, np.arange(len(group_ids)))),
        shape=(len(unique), len(group_ids)),
    )
    partial = assignment @ matrix
    destination[unique] += partial.toarray().astype(destination.dtype, copy=False)


def _to_profiles(
    counts: np.ndarray, batch_mass: np.ndarray, ntc_logcpm: np.ndarray
) -> np.ndarray:
    library = counts.sum(axis=1, dtype=np.float64)
    if np.any(library <= 0):
        raise ValueError("At least one pseudobulk profile has zero total counts")
    counts *= (1e6 / library).astype(np.float32)[:, None]
    np.log1p(counts, out=counts)
    mass_total = batch_mass.sum(axis=1, dtype=np.float64)
    proportions = np.divide(
        batch_mass,
        mass_total[:, None],
        out=np.zeros_like(batch_mass),
        where=mass_total[:, None] > 0,
    )
    counts -= proportions @ ntc_logcpm
    return counts


def _feature_symbols(dataset: dict[str, Any], feature_ids: np.ndarray) -> np.ndarray:
    mapping_path = dataset["expression"].get("target_feature_map")
    if not mapping_path:
        return feature_ids.astype(str)
    mapping = pd.read_csv(mapping_path, sep="\t", dtype=str).set_index("feature_id")["symbol"]
    return pd.Series(feature_ids).map(mapping).fillna(pd.Series(feature_ids)).astype(str).to_numpy()


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)
