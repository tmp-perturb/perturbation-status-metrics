"""Shared chunked expression access and benchmark-owned representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .io import _read_h5ad_array, read_h5ad_obs


@dataclass
class ExpressionSource:
    path: Path
    counts_layer: str
    feature_mode: str

    def metadata(self, mapping: dict[str, str], eligible: set[str]) -> tuple[pd.DataFrame, np.ndarray]:
        fields = ["cell_id", "target", "control", "assignment", "batch", "reagent"]
        columns = [mapping[field] for field in fields]
        if mapping.get("construct"):
            columns.append(mapping["construct"])
        frame = read_h5ad_obs(self.path, columns)
        frame.rename(columns={mapping[field]: field for field in fields}, inplace=True)
        if mapping.get("construct"):
            frame.rename(columns={mapping["construct"]: "construct_id"}, inplace=True)
        else:
            frame["construct_id"] = frame["reagent"]
        mask = frame["assignment"].isin(eligible).to_numpy()
        frame = frame.loc[mask].reset_index(drop=True)
        frame.rename(
            columns={"target": "target_id", "control": "is_ntc", "batch": "batch_id", "reagent": "reagent_id"},
            inplace=True,
        )
        frame["is_ntc"] = frame["is_ntc"].astype(bool)
        return frame, mask

    def feature_ids(self) -> np.ndarray:
        with h5py.File(self.path, "r") as handle:
            var = handle["var"]
            values = _read_h5ad_array(var[var.attrs["_index"]]).astype(str)
        if self.feature_mode == "direct":
            genes = values
        elif self.feature_mode == "usa_sa":
            if len(values) % 3:
                raise ValueError("USA feature count is not divisible into S/U/A blocks")
            size = len(values) // 3
            blocks = [values[offset * size : (offset + 1) * size] for offset in range(3)]
            suffixes = ("_S", "_U", "_A")
            base = []
            for block, suffix in zip(blocks, suffixes):
                if not np.all(np.char.endswith(block, suffix)):
                    raise ValueError(f"Invalid USA feature suffix block: {suffix}")
                base.append(np.asarray([value[: -len(suffix)] for value in block], dtype=str))
            if not (np.array_equal(base[0], base[1]) and np.array_equal(base[0], base[2])):
                raise ValueError("USA S/U/A feature blocks do not have identical gene order")
            genes = base[0]
        else:
            raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")
        if len(np.unique(genes)) != len(genes):
            raise ValueError("Benchmark expression feature IDs must be unique")
        return genes

    def iter_log_normalized(
        self, eligible_mask: np.ndarray, row_chunk_size: int
    ) -> Iterator[tuple[np.ndarray, sparse.csr_matrix]]:
        for original_rows, block in self.iter_counts(eligible_mask, row_chunk_size):
            library_size = np.asarray(block.sum(axis=1)).ravel()
            scale = np.divide(
                10000.0,
                library_size,
                out=np.zeros_like(library_size, dtype=np.float32),
                where=library_size > 0,
            )
            block = block.multiply(scale[:, None]).tocsr()
            np.log1p(block.data, out=block.data)
            yield original_rows, block

    def iter_counts(
        self, eligible_mask: np.ndarray, row_chunk_size: int
    ) -> Iterator[tuple[np.ndarray, sparse.csr_matrix]]:
        with h5py.File(self.path, "r") as handle:
            node = handle[f"layers/{self.counts_layer}"]
            indptr = node["indptr"][:]
            n_rows, n_features = map(int, node.attrs["shape"])
            if len(eligible_mask) != n_rows:
                raise ValueError("Eligible mask length does not match expression rows")
            gene_count = n_features if self.feature_mode == "direct" else n_features // 3
            for start in range(0, n_rows, row_chunk_size):
                stop = min(start + row_chunk_size, n_rows)
                selected = np.flatnonzero(eligible_mask[start:stop])
                if len(selected) == 0:
                    continue
                data_start, data_stop = int(indptr[start]), int(indptr[stop])
                block = sparse.csr_matrix(
                    (
                        node["data"][data_start:data_stop],
                        node["indices"][data_start:data_stop],
                        indptr[start : stop + 1] - data_start,
                    ),
                    shape=(stop - start, n_features),
                )[selected]
                if self.feature_mode == "usa_sa":
                    block = block[:, :gene_count] + block[:, 2 * gene_count : 3 * gene_count]
                block = block.astype(np.float32, copy=False).tocsr()
                yield selected + start, block
