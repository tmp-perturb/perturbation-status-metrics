"""Build the benchmark-owned expression representation shared by B3/B4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..context import StageContext
from ..expression import ExpressionSource
from ..io import atomic_write_json, atomic_write_parquet
from ..statistics import stable_seed


def run(context: StageContext) -> tuple[list[Path], dict[str, Any]]:
    settings = context.config.get("expression")
    if not settings:
        raise ValueError("The selected config does not define expression settings")
    artifacts: list[Path] = []
    details: dict[str, Any] = {"datasets": {}}
    for dataset_id in context.dataset_ids:
        dataset = context.config["datasets"][dataset_id]
        if "expression" not in dataset:
            raise ValueError(f"{dataset_id} does not define an expression adapter")
        dataset_artifacts, dataset_details = _build_dataset(
            context, dataset_id, dataset, settings
        )
        artifacts.extend(dataset_artifacts)
        details["datasets"][dataset_id] = dataset_details
    return artifacts, details


def _build_dataset(
    context: StageContext,
    dataset_id: str,
    dataset: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[list[Path], dict[str, Any]]:
    source = ExpressionSource(
        path=dataset["h5ad"],
        counts_layer=settings["counts_layer"],
        feature_mode=dataset["expression"]["feature_mode"],
    )
    eligible = set(context.config["cohort"]["eligible_assignment_structures"])
    metadata, eligible_mask = source.metadata(dataset["metadata"], eligible)
    genes = source.feature_ids()
    row_chunk = int(settings["row_chunk_size"])

    ntc_sum = np.zeros(len(genes), dtype=np.float64)
    ntc_sum_squares = np.zeros(len(genes), dtype=np.float64)
    ntc_detected = np.zeros(len(genes), dtype=np.int64)
    ntc_count = 0
    eligible_lookup = np.full(len(eligible_mask), -1, dtype=np.int64)
    eligible_lookup[np.flatnonzero(eligible_mask)] = np.arange(len(metadata))
    for original_rows, matrix in source.iter_log_normalized(eligible_mask, row_chunk):
        eligible_rows = eligible_lookup[original_rows]
        ntc_local = metadata.loc[eligible_rows, "is_ntc"].to_numpy()
        if not ntc_local.any():
            continue
        ntc = matrix[ntc_local]
        ntc_sum += np.asarray(ntc.sum(axis=0)).ravel()
        ntc_sum_squares += np.asarray(ntc.power(2).sum(axis=0)).ravel()
        ntc_detected += np.asarray(ntc.getnnz(axis=0)).ravel()
        ntc_count += ntc.shape[0]
    if ntc_count != int(metadata["is_ntc"].sum()):
        raise RuntimeError(f"{dataset_id}: NTC scan count mismatch")
    mean = ntc_sum / ntc_count
    variance = np.maximum(ntc_sum_squares / ntc_count - mean**2, 0)
    detection = ntc_detected / ntc_count
    eligible_genes = np.flatnonzero(
        (detection >= float(settings["min_ntc_detection_fraction"])) & (variance > 0)
    )
    hvg_count = min(int(settings["hvg_count"]), len(eligible_genes))
    order = np.lexsort((genes[eligible_genes], -variance[eligible_genes]))
    hvg_indices = eligible_genes[order[:hvg_count]]

    dataset_dir = context.stage_dir / "expression" / f"dataset_id={dataset_id}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_memmap_path = dataset_dir / "hvg_expression.float32.tmp"
    expression = np.memmap(
        raw_memmap_path, mode="w+", dtype=np.float32, shape=(len(metadata), hvg_count)
    )
    written = 0
    for original_rows, matrix in source.iter_log_normalized(eligible_mask, row_chunk):
        eligible_rows = eligible_lookup[original_rows]
        expression[eligible_rows] = matrix[:, hvg_indices].toarray().astype(np.float32, copy=False)
        written += len(eligible_rows)
    expression.flush()
    if written != len(metadata):
        raise RuntimeError(f"{dataset_id}: expression scan count mismatch")

    sample_indices = _balanced_sample(
        metadata,
        int(settings["pca_max_cells_per_target"]),
        int(settings["pca_max_ntc_cells"]),
        stable_seed(int(settings["seed"]), "pca_sample", dataset_id),
    )
    hvg_mean = mean[hvg_indices].astype(np.float32)
    hvg_sd = np.sqrt(variance[hvg_indices]).astype(np.float32)
    sample = np.asarray(expression[sample_indices], dtype=np.float32)
    sample = (sample - hvg_mean) / hvg_sd
    n_components = min(int(settings["n_pcs"]), sample.shape[0] - 1, sample.shape[1])
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=stable_seed(int(settings["seed"]), "pca_fit", dataset_id),
    )
    pca.fit(sample)
    del sample

    embedding_memmap_path = dataset_dir / "embedding.float32.tmp"
    embedding = np.memmap(
        embedding_memmap_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(metadata), n_components),
    )
    transform_chunk = int(settings["transform_chunk_size"])
    for start in range(0, len(metadata), transform_chunk):
        stop = min(start + transform_chunk, len(metadata))
        block = np.asarray(expression[start:stop], dtype=np.float32)
        block = (block - hvg_mean) / hvg_sd
        embedding[start:stop] = pca.transform(block).astype(np.float32)
    embedding.flush()
    batch_centroids = {}
    for batch_id, indices in metadata.groupby("batch_id", sort=True).indices.items():
        indices = np.asarray(indices)
        ntc_indices = indices[metadata.loc[indices, "is_ntc"].to_numpy()]
        if len(ntc_indices) == 0:
            raise ValueError(f"{dataset_id}: batch {batch_id} has no eligible NTC cells")
        centroid = np.asarray(embedding[ntc_indices]).mean(axis=0, dtype=np.float64).astype(np.float32)
        embedding[indices] -= centroid
        batch_centroids[str(batch_id)] = centroid.tolist()
    embedding.flush()

    pc_columns = [f"pc_{index + 1:02d}" for index in range(n_components)]
    output_frame = metadata[["cell_id", "target_id", "is_ntc", "batch_id", "reagent_id"]].copy()
    for index, column in enumerate(pc_columns):
        output_frame[column] = np.asarray(embedding[:, index])
    embedding_path = dataset_dir / "embedding.parquet"
    atomic_write_parquet(output_frame, embedding_path)

    feature_frame = pd.DataFrame({
        "feature_id": genes,
        "ntc_mean_log1p_cp10k": mean,
        "ntc_variance_log1p_cp10k": variance,
        "ntc_detection_fraction": detection,
        "selected_hvg": False,
        "hvg_rank": pd.Series(pd.NA, index=np.arange(len(genes)), dtype="Int64"),
    })
    feature_frame.loc[hvg_indices, "selected_hvg"] = True
    feature_frame.loc[hvg_indices, "hvg_rank"] = np.arange(1, hvg_count + 1)
    feature_path = dataset_dir / "feature_manifest.parquet"
    atomic_write_parquet(feature_frame, feature_path)

    sample_path = dataset_dir / "pca_sample_cells.parquet"
    atomic_write_parquet(metadata.loc[sample_indices, ["cell_id", "target_id", "is_ntc", "batch_id"]], sample_path)
    metadata_path = dataset_dir / "pca_metadata.json"
    atomic_write_json(metadata_path, {
        "dataset_id": dataset_id,
        "eligible_cells": len(metadata),
        "ntc_cells": ntc_count,
        "features": len(genes),
        "selected_hvgs": hvg_count,
        "pca_sample_cells": len(sample_indices),
        "n_pcs": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "batch_ntc_centroids_before_residualization": batch_centroids,
        "settings": settings,
    })
    del expression, embedding
    raw_memmap_path.unlink()
    embedding_memmap_path.unlink()
    return [embedding_path, feature_path, sample_path, metadata_path], {
        "eligible_cells": len(metadata),
        "ntc_cells": ntc_count,
        "features": len(genes),
        "selected_hvgs": hvg_count,
        "pca_sample_cells": len(sample_indices),
        "n_pcs": n_components,
    }


def _balanced_sample(
    metadata: pd.DataFrame,
    max_cells_per_target: int,
    max_ntc_cells: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    targeted = metadata.loc[~metadata["is_ntc"]]
    for _, indices in targeted.groupby("target_id", sort=True).groups.items():
        indices = np.asarray(indices)
        take = min(max_cells_per_target, len(indices))
        selected.append(np.sort(rng.choice(indices, size=take, replace=False)))
    ntc = metadata.loc[metadata["is_ntc"]]
    batch_groups = list(ntc.groupby("batch_id", sort=True).groups.values())
    per_batch = max(1, int(np.ceil(max_ntc_cells / len(batch_groups))))
    ntc_selected = []
    for indices in batch_groups:
        indices = np.asarray(indices)
        take = min(per_batch, len(indices))
        ntc_selected.extend(rng.choice(indices, size=take, replace=False))
    if len(ntc_selected) > max_ntc_cells:
        ntc_selected = list(rng.choice(np.asarray(ntc_selected), size=max_ntc_cells, replace=False))
    selected.append(np.sort(np.asarray(ntc_selected, dtype=np.int64)))
    return np.sort(np.concatenate(selected)).astype(np.int64)
