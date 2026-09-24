"""Shared statistical estimators for benchmark stages."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.stats import spearmanr


def stable_seed(base_seed: int, *labels: str) -> int:
    payload = "\x1f".join((str(base_seed), *labels)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def bootstrap_median_ci(
    values: np.ndarray, resamples: int, seed: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    estimate = float(np.median(values))
    rng = np.random.default_rng(seed)
    medians = np.empty(resamples, dtype=float)
    chunk = 100
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        medians[start:stop] = np.median(values[indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return estimate, float(low), float(high)


def bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, resamples: int, seed: int
) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    estimate = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(resamples):
        indices = rng.integers(0, len(x), size=len(x))
        if len(np.unique(x[indices])) < 2 or len(np.unique(y[indices])) < 2:
            continue
        value = float(spearmanr(x[indices], y[indices]).statistic)
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return estimate, np.nan, np.nan
    low, high = np.quantile(estimates, [0.025, 0.975])
    return estimate, float(low), float(high)
