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
    membership_records: list[pd.DataFrame] = []
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
            membership_records.append(pd.DataFrame({
                "profile_row": group_id,
                "cell_id": group["cell_id"].astype(str).to_numpy(),
            }))
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
    ntc_counts = np.zeros((2, n_batches, n_genes), dtype=np.float64)
    weighted_batch_mass = np.zeros((n_groups, n_batches), dtype=np.float32)
    unweighted_batch_mass = np.zeros((n_groups, n_batches), dtype=np.float32)
    eligible_lookup = np.full(len(eligible_mask), -1, dtype=np.int64)
    eligible_lookup[np.flatnonzero(eligible_mask)] = np.arange(len(metadata))
    ntc_split = np.full(len(metadata), -1, dtype=np.int8)
    ntc_pool_records = []
    for batch_id, batch in metadata[metadata["is_ntc"]].groupby("batch_id", sort=True, observed=True):
        ordered = batch.sort_values("cell_id", kind="mergesort")
        positions = ordered.index.to_numpy(dtype=np.int64)
        assignments = np.arange(len(positions), dtype=np.int64) % 2
        ntc_split[positions] = assignments
        ntc_pool_records.append({
            "batch_id": str(batch_id), "eligible_ntc_cells": len(positions),
            "ntc_a_cells": int((assignments == 0).sum()), "ntc_b_cells": int((assignments == 1).sum()),
        })

    for original_rows, matrix in source.iter_counts(
        eligible_mask, int(context.config["expression"]["row_chunk_size"])
    ):
        eligible_rows = eligible_lookup[original_rows]
        block = matrix[:, feature_indices]
        ntc_local = metadata.loc[eligible_rows, "is_ntc"].to_numpy()
        if ntc_local.any():
            for pool in (0, 1):
                selected_ntc = ntc_local & ntc_split[eligible_rows].eq(pool) if isinstance(ntc_local, pd.Series) else ntc_local & (ntc_split[eligible_rows] == pool)
                if selected_ntc.any():
                    _accumulate_rows(
                        ntc_counts[pool],
                        batch_codes[eligible_rows][selected_ntc],
                        block[selected_ntc],
                        np.ones(int(selected_ntc.sum()), dtype=np.float32),
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

    ntc_library = ntc_counts.sum(axis=2)
    supported_batches = np.all(ntc_library > 0, axis=0)
    ntc_logcpm = np.zeros_like(ntc_counts, dtype=np.float32)
    for pool in (0, 1):
        valid = ntc_library[pool] > 0
        ntc_logcpm[pool, valid] = np.log1p(
            ntc_counts[pool, valid] * (1e6 / ntc_library[pool, valid])[:, None]
        ).astype(np.float32)
    weighted_supported = ~np.any((weighted_batch_mass > 0) & (~supported_batches)[None, :], axis=1)
    unweighted_supported = ~np.any((unweighted_batch_mass > 0) & (~supported_batches)[None, :], axis=1)
    groups["ntc_batch_support"] = weighted_supported & unweighted_supported
    weighted_profiles_ntc_a = _to_profiles(weighted_counts.copy(), weighted_batch_mass, ntc_logcpm[0])
    weighted_profiles_ntc_b = _to_profiles(weighted_counts.copy(), weighted_batch_mass, ntc_logcpm[1])
    unweighted_profiles_ntc_a = _to_profiles(unweighted_counts.copy(), unweighted_batch_mass, ntc_logcpm[0])
    unweighted_profiles_ntc_b = _to_profiles(unweighted_counts.copy(), unweighted_batch_mass, ntc_logcpm[1])

    dataset_dir = context.stage_dir / "pseudobulk" / f"dataset_id={dataset_id}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    profile_paths = {
        "weighted_ntc_a": dataset_dir / "weighted_profiles_ntc_a.npy",
        "weighted_ntc_b": dataset_dir / "weighted_profiles_ntc_b.npy",
        "unweighted_ntc_a": dataset_dir / "unweighted_profiles_ntc_a.npy",
        "unweighted_ntc_b": dataset_dir / "unweighted_profiles_ntc_b.npy",
    }
    for key, array in {
        "weighted_ntc_a": weighted_profiles_ntc_a, "weighted_ntc_b": weighted_profiles_ntc_b,
        "unweighted_ntc_a": unweighted_profiles_ntc_a, "unweighted_ntc_b": unweighted_profiles_ntc_b,
    }.items():
        _atomic_save_array(profile_paths[key], array)
    groups_path = dataset_dir / "group_manifest.parquet"
    atomic_write_parquet(groups, groups_path)
    membership_path = dataset_dir / "group_cell_membership.parquet"
    memberships = pd.concat(membership_records, ignore_index=True) if membership_records else pd.DataFrame(columns=["profile_row", "cell_id"])
    atomic_write_parquet(memberships, membership_path)
    ntc_split_path = dataset_dir / "ntc_split_manifest.parquet"
    atomic_write_parquet(pd.DataFrame(ntc_pool_records), ntc_split_path)
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
        "reference": "batch_composition_matched_disjoint_ntc_halves_log1p_cpm",
        "ntc_split": "within_batch_cell_id_sort_then_alternating_50_50",
        "symmetric_reference_swap": True,
        "profile_unit": "construct_id",
        "guide_set_source": settings.get("guide_set_source", "observed_eligible_cells"),
        "require_disjoint_guide_sets": bool(settings.get("require_disjoint_guide_sets", True)),
    })
    return [*profile_paths.values(), groups_path, membership_path, ntc_split_path, features_path, metadata_path], {
        "groups": n_groups,
        "features": n_genes,
        "profiles": 4,
        "unsupported_profile_groups": int((~groups["ntc_batch_support"]).sum()),
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
