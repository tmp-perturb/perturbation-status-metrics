"""B1 coverage and assessability metrics."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..context import StageContext
from ..io import atomic_write_tsv
from .harmonize import METHOD_ORDER


def run(context: StageContext) -> tuple[list[Path], dict[str, Any]]:
    source = context.upstream("validate", "method_availability.tsv")
    if not source.exists():
        raise RuntimeError("coverage requires a completed validate stage")
    availability = pd.read_csv(source, sep="\t")
    minimum = int(context.config["cohort"]["min_cells_per_perturbation"])
    summaries: list[dict[str, Any]] = []
    common_records: list[dict[str, Any]] = []
    pattern_records: list[dict[str, Any]] = []

    for dataset_id in context.dataset_ids:
        dataset = context.config["datasets"][dataset_id]
        dataset_rows = availability[availability["dataset_id"] == dataset_id]
        strata = [("all", dataset_rows)]
        if dataset.get("perturbation_delimiter"):
            strata.extend([
                ("single_gene", dataset_rows[dataset_rows["perturbation_order"] == 1]),
                ("multi_gene", dataset_rows[dataset_rows["perturbation_order"] > 1]),
            ])
        for stratum, rows in strata:
            target_reference = rows[rows["method_id"] == METHOD_ORDER[0]][
                ["target_id", "targeted_cells"]
            ].drop_duplicates()
            represented = set(target_reference["target_id"])
            target_cells = target_reference.set_index("target_id")["targeted_cells"]
            scorable_sets: dict[str, set[str]] = {}
            for method_id in METHOD_ORDER:
                method_rows = rows[rows["method_id"] == method_id]
                scorable = set(method_rows.loc[method_rows["scorable_target"], "target_id"])
                scorable_sets[method_id] = scorable
                performance = set(target_cells[target_cells >= minimum].index)
                summaries.append({
                    "dataset_id": dataset_id,
                    "stratum": stratum,
                    "method_id": method_id,
                    "represented_perturbations": len(represented),
                    "scorable_perturbations": len(scorable),
                    "perturbation_coverage": len(scorable) / len(represented),
                    "eligible_targeted_cells": int(target_cells.sum()),
                    "cells_in_scorable_perturbations": int(target_cells.loc[list(scorable)].sum()) if scorable else 0,
                    "targeted_cell_coverage": float(target_cells.loc[list(scorable)].sum() / target_cells.sum()) if scorable else 0.0,
                    "performance_eligible_perturbations": len(performance),
                    "performance_scorable_perturbations": len(performance & scorable),
                })
            for left, right in combinations(METHOD_ORDER, 2):
                common_records.append({
                    "dataset_id": dataset_id,
                    "stratum": stratum,
                    "methods": f"{left}+{right}",
                    "common_scorable_perturbations": len(scorable_sets[left] & scorable_sets[right]),
                })
            all_common = set.intersection(*(scorable_sets[method] for method in METHOD_ORDER))
            common_records.append({
                "dataset_id": dataset_id,
                "stratum": stratum,
                "methods": "all_four",
                "common_scorable_perturbations": len(all_common),
            })
            target_matrix = (
                rows.pivot(index="target_id", columns="method_id", values="scorable_target")
                .reindex(columns=METHOD_ORDER)
                .fillna(False)
                .astype(bool)
            )
            patterns = target_matrix.groupby(METHOD_ORDER, observed=True).size()
            for pattern, count in patterns.items():
                pattern_records.append({
                    "dataset_id": dataset_id,
                    "stratum": stratum,
                    **{method: bool(value) for method, value in zip(METHOD_ORDER, pattern)},
                    "perturbations": int(count),
                })

    summary = pd.DataFrame(summaries)
    common = pd.DataFrame(common_records)
    patterns = pd.DataFrame(pattern_records)
    summary_path = context.stage_dir / "coverage_summary.tsv"
    common_path = context.stage_dir / "common_perturbations.tsv"
    pattern_path = context.stage_dir / "availability_patterns.tsv"
    atomic_write_tsv(summary, summary_path)
    atomic_write_tsv(common, common_path)
    atomic_write_tsv(patterns, pattern_path)
    artifacts = [summary_path, common_path, pattern_path]
    return artifacts, {"summary_rows": len(summary), "availability_patterns": len(patterns)}
