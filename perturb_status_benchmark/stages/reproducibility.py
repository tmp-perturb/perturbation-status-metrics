"""Construct-aware perturbation-profile reproducibility."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..context import StageContext
from ..io import atomic_savefig, atomic_write_tsv
from ..statistics import bootstrap_median_ci, stable_seed
from .harmonize import METHOD_ORDER


def run(context: StageContext) -> tuple[list[Path], dict[str, Any]]:
    records, pair_records = [], []
    for dataset_id in context.dataset_ids:
        dataset = context.config["datasets"][dataset_id]
        if not dataset.get("reproducibility", {}).get("enabled", True):
            continue
        root = context.upstream("pseudobulk", "pseudobulk", f"dataset_id={dataset_id}")
        groups = pd.read_parquet(root / "group_manifest.parquet")
        features = pd.read_parquet(root / "profile_features.parquet")
        weighted = np.load(root / "weighted_profiles.npy", mmap_mode="r")
        unweighted = np.load(root / "unweighted_profiles.npy", mmap_mode="r")
        symbols = features["feature_symbol"].astype(str).to_numpy()
        delimiter = dataset.get("perturbation_delimiter")
        for (method_id, target_id), target_groups in groups.groupby(["method_id", "target_id"], sort=True, observed=True):
            excluded = set(str(target_id).split(delimiter)) if delimiter else {str(target_id)}
            keep = ~np.isin(symbols, list(excluded))
            valid_pairs = []
            for (_, left), (_, right) in combinations(target_groups.iterrows(), 2):
                left_guides = set(str(left.guide_set).split("|"))
                right_guides = set(str(right.guide_set).split("|"))
                shared = sorted(left_guides & right_guides)
                weighted_rho = _spearman_pair(weighted[int(left.profile_row), keep], weighted[int(right.profile_row), keep])
                unweighted_rho = _spearman_pair(unweighted[int(left.profile_row), keep], unweighted[int(right.profile_row), keep])
                disjoint = not shared
                pair_records.append({
                    "dataset_id": dataset_id, "method_id": method_id, "target_id": target_id,
                    "construct_left": left.construct_id, "construct_right": right.construct_id,
                    "guide_set_left": left.guide_set, "guide_set_right": right.guide_set,
                    "shared_guides": "|".join(shared), "guide_sets_disjoint": disjoint,
                    "rho_weighted": weighted_rho, "rho_unweighted": unweighted_rho,
                    "delta_rho": weighted_rho - unweighted_rho,
                })
                if disjoint and np.isfinite(weighted_rho) and np.isfinite(unweighted_rho):
                    valid_pairs.append((weighted_rho, unweighted_rho))
            if not valid_pairs:
                continue
            weighted_rho = float(np.median([value[0] for value in valid_pairs]))
            unweighted_rho = float(np.median([value[1] for value in valid_pairs]))
            records.append({
                "dataset_id": dataset_id, "method_id": method_id, "target_id": target_id,
                "perturbation_order": int(target_groups["perturbation_order"].iloc[0]),
                "constructs": len(target_groups),
                "all_construct_pairs": len(target_groups) * (len(target_groups) - 1) // 2,
                "independent_guide_set_pairs": len(valid_pairs), "profile_features": int(keep.sum()),
                "median_construct_cells": float(target_groups["cells"].median()),
                "median_effective_weighted_cells": float(target_groups["effective_weighted_cells"].median()),
                "rho_weighted": weighted_rho, "rho_unweighted": unweighted_rho,
                "delta_rho": weighted_rho - unweighted_rho,
            })
    per_target, pairs = pd.DataFrame(records), pd.DataFrame(pair_records)
    summary = _summaries(per_target, context)
    pairwise = _paired_differences(per_target, context)
    sensitivity_target, sensitivity_summary = _two_guide_sensitivity(pairs, context)
    outputs = {
        "per": context.stage_dir / "reproducibility_per_perturbation.tsv",
        "summary": context.stage_dir / "reproducibility_summary.tsv",
        "pairwise": context.stage_dir / "reproducibility_pairwise.tsv",
        "pairs": context.stage_dir / "construct_pair_audit.tsv",
        "sensitivity_per": context.stage_dir / "two_observed_guides_sensitivity_per_perturbation.tsv",
        "sensitivity_summary": context.stage_dir / "two_observed_guides_sensitivity_summary.tsv",
    }
    for frame, key in [(per_target,"per"),(summary,"summary"),(pairwise,"pairwise"),(pairs,"pairs"),
                       (sensitivity_target,"sensitivity_per"),(sensitivity_summary,"sensitivity_summary")]:
        atomic_write_tsv(frame, outputs[key])
    figures = _plot(per_target, summary, context.stage_dir)
    return [*outputs.values(), *figures], {
        "perturbation_method_rows": len(per_target), "construct_pair_rows": len(pairs),
        "profile_unit": "construct_id", "pair_rule": "different constructs with disjoint observed guide sets",
    }


def _spearman_pair(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = rankdata(left), rankdata(right)
    left_rank -= left_rank.mean(); right_rank -= right_rank.mean()
    denominator = np.linalg.norm(left_rank) * np.linalg.norm(right_rank)
    return np.nan if denominator == 0 else float(np.dot(left_rank, right_rank) / denominator)


def _strata(dataset: dict[str, Any]) -> list[tuple[str, int | None]]:
    result = [("all", None)]
    if dataset.get("perturbation_delimiter"):
        result.extend([("single_gene", 1), ("multi_gene", 2)])
    return result


def _select(frame: pd.DataFrame, order: int | None) -> pd.DataFrame:
    if order is None: return frame
    return frame[frame["perturbation_order"].eq(1) if order == 1 else frame["perturbation_order"].gt(1)]


def _summaries(frame: pd.DataFrame, context: StageContext) -> pd.DataFrame:
    rows=[]; resamples=int(context.config["statistics"]["bootstrap_resamples"]); seed=int(context.config["statistics"]["seed"])
    for dataset_id in context.dataset_ids:
        dataset=context.config["datasets"][dataset_id]; enabled=dataset.get("reproducibility",{}).get("enabled",True)
        dataset_rows=frame[frame.dataset_id.eq(dataset_id)] if len(frame) else frame
        for stratum,order in _strata(dataset):
            selected=_select(dataset_rows,order)
            for method_id in METHOD_ORDER:
                method=selected[selected.method_id.eq(method_id)] if len(selected) else selected
                if enabled:
                    estimate,low,high=bootstrap_median_ci(method.delta_rho.to_numpy(),resamples,stable_seed(seed,"reproducibility",dataset_id,stratum,method_id))
                    status="estimable" if len(method) else "not_estimable"
                else:
                    estimate=low=high=np.nan; status=dataset.get("reproducibility",{}).get("status","not_applicable")
                rows.append({"dataset_id":dataset_id,"stratum":stratum,"method_id":method_id,"status":status,
                             "perturbations":len(method),"median_delta_rho":estimate,"ci_low":low,"ci_high":high,
                             "fraction_delta_gt_0_01":float((method.delta_rho>.01).mean()) if len(method) else np.nan,
                             "median_effective_weighted_cells":float(method.median_effective_weighted_cells.median()) if len(method) else np.nan})
    return pd.DataFrame(rows)


def _paired_differences(frame: pd.DataFrame, context: StageContext) -> pd.DataFrame:
    rows=[]; resamples=int(context.config["statistics"]["bootstrap_resamples"]); seed=int(context.config["statistics"]["seed"])
    for dataset_id in context.dataset_ids:
        dataset=context.config["datasets"][dataset_id]; dataset_rows=frame[frame.dataset_id.eq(dataset_id)] if len(frame) else frame
        for stratum,order in _strata(dataset):
            selected=_select(dataset_rows,order)
            wide=selected.pivot(index="target_id",columns="method_id",values="delta_rho").reindex(columns=METHOD_ORDER) if len(selected) else pd.DataFrame(columns=METHOD_ORDER)
            for left,right in combinations(METHOD_ORDER,2):
                common=wide[[left,right]].dropna()
                estimate,low,high=bootstrap_median_ci((common[left]-common[right]).to_numpy(),resamples,stable_seed(seed,"reproducibility_difference",dataset_id,stratum,left,right))
                rows.append({"dataset_id":dataset_id,"stratum":stratum,"method_left":left,"method_right":right,
                             "common_perturbations":len(common),"median_delta_rho_difference":estimate,"ci_low":low,"ci_high":high})
    return pd.DataFrame(rows)


def _two_guide_sensitivity(pairs: pd.DataFrame, context: StageContext) -> tuple[pd.DataFrame,pd.DataFrame]:
    if pairs.empty: return pd.DataFrame(),pd.DataFrame()
    selected=pairs[pairs.dataset_id.eq("replogle2022") & pairs.guide_sets_disjoint].copy()
    selected=selected[(selected.guide_set_left.str.count(r"\|")+1).eq(2) & (selected.guide_set_right.str.count(r"\|")+1).eq(2)]
    per=(selected.groupby(["dataset_id","method_id","target_id"],observed=True)
         .agg(rho_weighted=("rho_weighted","median"),rho_unweighted=("rho_unweighted","median"),independent_pairs=("target_id","size")).reset_index())
    per["delta_rho"]=per.rho_weighted-per.rho_unweighted
    rows=[]; seed=int(context.config["statistics"]["seed"]); resamples=int(context.config["statistics"]["bootstrap_resamples"])
    for method in METHOD_ORDER:
        x=per[per.method_id.eq(method)]
        estimate,low,high=bootstrap_median_ci(x.delta_rho.to_numpy(),resamples,stable_seed(seed,"two_observed_guides",method))
        rows.append({"dataset_id":"replogle2022","method_id":method,"perturbations":len(x),"median_delta_rho":estimate,"ci_low":low,"ci_high":high})
    return per,pd.DataFrame(rows)


def _plot(frame: pd.DataFrame, summary: pd.DataFrame, output: Path) -> list[Path]:
    datasets=list(frame.dataset_id.unique()); figure,axes=plt.subplots(1,len(datasets),figsize=(8.4,4),sharey=True,squeeze=False)
    colors={"mixscape":"#0072B2","mixscale":"#CC79A7","ps":"#009E73"}; rng=np.random.default_rng(20260913)
    for axis,dataset in zip(axes.ravel(),datasets):
        for index,method in enumerate(METHOD_ORDER):
            values=frame[(frame.dataset_id==dataset)&(frame.method_id==method)].delta_rho.to_numpy()
            axis.scatter(index+rng.uniform(-.12,.12,len(values)),values,s=7,color=colors[method],alpha=.2,linewidths=0)
            row=summary[(summary.dataset_id==dataset)&(summary.stratum=="all")&(summary.method_id==method)].iloc[0]
            if pd.notna(row.median_delta_rho):
                axis.errorbar(index,row.median_delta_rho,yerr=[[row.median_delta_rho-row.ci_low],[row.ci_high-row.median_delta_rho]],fmt="o",color=colors[method],capsize=3,zorder=4)
        axis.axhline(0,color="#777",lw=.7); axis.set_xticks(range(len(METHOD_ORDER)),["Mixscape","Mixscale","PS"],rotation=20,ha="right"); axis.set_title(dataset)
    axes.ravel()[0].set_ylabel("Median pairwise Δρ (weighted − unweighted)"); figure.suptitle("Independent construct-profile gain"); figure.tight_layout()
    png=output/"reproducibility.png"; pdf=output/"reproducibility.pdf"
    atomic_savefig(figure,png,dpi=200); atomic_savefig(figure,pdf); plt.close(figure)
    return [png,pdf]
