"""Shared I/O helpers used by every benchmark stage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


def read_h5ad_obs(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read selected AnnData obs columns without opening the expression matrix."""
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        missing = [column for column in columns if column not in obs]
        if missing:
            raise ValueError(f"Missing obs columns in {path}: {missing}")
        result = {column: _read_h5ad_array(obs[column]) for column in columns}
    return pd.DataFrame(result)


def _read_h5ad_array(node: h5py.Dataset | h5py.Group) -> np.ndarray:
    encoding = node.attrs.get("encoding-type", "")
    if encoding == "categorical":
        categories = _decode(node["categories"][:])
        codes = node["codes"][:]
        values = np.empty(len(codes), dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        values[~valid] = None
        return values
    return _decode(node[:])


def _decode(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind != "O":
        return values
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else value for value in values],
        dtype=object,
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def atomic_savefig(figure: Any, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, format=path.suffix.lstrip("."), **kwargs)
    os.replace(temporary, path)
