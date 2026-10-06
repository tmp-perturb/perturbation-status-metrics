"""NTC-free within-perturbation direct target-expression benchmark."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..context import StageContext
from ..io import atomic_write_tsv
from ..statistics import bootstrap_median_ci, stable_seed

METHODS = ["mixscape", "mixscale", "ps"]
DATASETS = ["replogle2022", "norman2019", "papalexi2021"]


def run(context: StageContext) -> tuple[list[Path], dict[str, Any]]:
    scores_root = context.upstream("validate", "harmonized", "cell_scores")
    expression_path = Path(context.config["direct_expression"]["cell_table"])
    if not scores_root.exists() or not expression_path.exists():
        raise RuntimeError("target_expression requires harmonized scores and the frozen direct-expression cell table")
    expression = pd.read_csv(expression_path, sep="\t")
    minimum = int(context.config["cohort"]["min_cells_per_perturbation"])
    resamples = int(context.config["statistics"]["bootstrap_resamples"])
    base_seed = int(context.config["statistics"]["seed"])
    correlation_rows, bulk_rows = [], []

    for dataset_id in [dataset for dataset in DATASETS if dataset in context.dataset_ids]:
        direct = expression[
            expression.dataset_id.eq(dataset_id) & expression.status.eq("scorable")
        ][["cell_id", "target_id", "direct_response", "components_found", "components_total"]]
        scores = pd.read_parquet(
            scores_root / f"dataset_id={dataset_id}" / "part-000.parquet",
            columns=["cell_id", "target_id", "is_ntc", "perturbation_order", "method_id", "score", "scorable"],
        )
        scores = scores[(~scores.is_ntc) & scores.method_id.isin(METHODS) & scores.scorable]
        for method_id in METHODS:
            method = scores[scores.method_id.eq(method_id)][
                ["cell_id", "target_id", "perturbation_order", "score"]
            ]
            common = method.merge(direct, on=["cell_id", "target_id"], validate="one_to_one")
            for (target_id, order), group in common.groupby(["target_id", "perturbation_order"], sort=True, observed=True):
                n = len(group); score_unique = group.score.nunique(); response_unique = group.direct_response.nunique()
                if n < minimum: status, rho = "insufficient_common_cells", np.nan
                elif score_unique < 2: status, rho = "constant_method_score", np.nan
                elif response_unique < 2: status, rho = "constant_direct_expression", np.nan
                else:
                    rho = float(spearmanr(group.score, group.direct_response).statistic)
                    status = "scorable" if np.isfinite(rho) else "undefined_correlation"
                correlation_rows.append({
                    "dataset_id": dataset_id, "target_id": target_id,
                    "method_id": method_id, "common_cells": n, "method_unique_scores": score_unique,
                    "direct_expression_unique_values": response_unique, "spearman_rho": rho, "status": status,
                })
            counts = common.groupby("target_id", observed=True).size()
            unique_score = common.groupby("target_id", observed=True).score.nunique()
            unique_response = common.groupby("target_id", observed=True).direct_response.nunique()
            keep = counts.index[(counts >= minimum) & (unique_score >= 2) & (unique_response >= 2)]
            q = common[common.target_id.isin(keep)].copy()
            q["score_percentile"] = q.groupby("target_id", observed=True).score.rank(method="average", pct=True)
            q["response_percentile"] = q.groupby("target_id", observed=True).direct_response.rank(method="average", pct=True)
            q["score_quantile"] = np.ceil(q.score_percentile * 5).clip(1, 5).astype(int)
            bulk = q.groupby(["target_id", "score_quantile"], observed=True, as_index=False).agg(
                direct_expression_pseudobulk=("response_percentile", "mean"), cells=("cell_id", "size")
            )
            bulk["dataset_id"] = dataset_id; bulk["method_id"] = method_id
            bulk_rows.append(bulk)

    per = pd.DataFrame(correlation_rows)
    bulks = pd.concat(bulk_rows, ignore_index=True)
    summary = _summaries(per, context, resamples, base_seed)
    pairwise = _pairwise(per, context, resamples, base_seed)
    trends = _trends(bulks, context, resamples, base_seed)
    outputs = {
        "direct_expression_per_perturbation.tsv": per,
        "direct_expression_summary.tsv": summary,
        "direct_expression_pairwise.tsv": pairwise,
        "direct_expression_quantile_pseudobulk.tsv": bulks,
        "direct_expression_quantile_summary.tsv": trends,
    }
    paths=[]
    for name, table in outputs.items():
        path=context.stage_dir/name; atomic_write_tsv(table,path); paths.append(path)
    return paths, {"perturbation_method_rows":len(per),"scorable_correlations":int(per.spearman_rho.notna().sum()),"quantile_pseudobulks":len(bulks),"readout":"ntc_free_direction_adjusted_direct_expression"}


def _strata(context: StageContext, dataset_id: str):
    result=[("all",None)]
    if context.config["datasets"][dataset_id].get("perturbation_delimiter"):
        result += [("single_gene",1),("multi_gene",2)]
    return result


def _select(frame, order):
    return frame if order is None else frame[frame.perturbation_order.eq(1) if order==1 else frame.perturbation_order.gt(1)]


def _summaries(per, context, resamples, seed):
    rows=[]
    for ds in sorted(dataset for dataset in DATASETS if dataset in context.dataset_ids):
        for method in sorted(METHODS):
            group=per[(per.dataset_id==ds)&(per.method_id==method)]; valid=group.spearman_rho.dropna().to_numpy()
            est,lo,hi=_bootstrap_median_exact(valid,resamples,seed + DATASETS.index(ds)*100 + METHODS.index(method))
            rows.append({"dataset_id":ds,"method_id":method,"readout_definition":"direct_expression","evaluated_perturbations":len(group),"scorable_perturbations":len(valid),"median_rho":est,"ci_low":lo,"ci_high":hi,"fraction_rho_positive":float(np.mean(valid>0)) if len(valid) else np.nan})
    return pd.DataFrame(rows)


def _pairwise(per, context, resamples, seed):
    rows=[]
    for ds in [dataset for dataset in DATASETS if dataset in context.dataset_ids]:
        wide=per[per.dataset_id.eq(ds)].pivot(index="target_id",columns="method_id",values="spearman_rho")
        for left,right in combinations(METHODS,2):
            common=wide[[left,right]].dropna(); values=(common[left]-common[right]).to_numpy()
            est,lo,hi=bootstrap_median_ci(values,resamples,stable_seed(seed,"direct_expression_difference",ds,left,right))
            rows.append({"dataset_id":ds,"method_left":left,"method_right":right,"common_perturbations":len(common),"median_rho_difference":est,"ci_low":lo,"ci_high":hi})
    return pd.DataFrame(rows)


def _trends(bulks, context, resamples, seed):
    rows=[]
    for ds in [dataset for dataset in DATASETS if dataset in context.dataset_ids]:
        for method in METHODS:
            subset = bulks[(bulks.dataset_id == ds) & (bulks.method_id == method)]
            for quantile, group in subset.groupby("score_quantile", sort=True, observed=True):
                values=group.direct_expression_pseudobulk.to_numpy()
                est,lo,hi=_bootstrap_median_exact(values,resamples,seed + DATASETS.index(ds)*1000 + METHODS.index(method)*100 + int(quantile))
                rows.append({"dataset_id":ds,"method_id":method,"score_quantile":int(quantile),"median_direct_expression_pseudobulk":est,"ci_low":lo,"ci_high":hi,"perturbations":len(values),"cells":int(group.cells.sum())})
    return pd.DataFrame(rows)


def _bootstrap_median_exact(values, resamples, seed):
    """Match the frozen direct-expression validation bootstrap exactly."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True)
    boot = np.median(draws, axis=1)
    return float(np.median(values)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
