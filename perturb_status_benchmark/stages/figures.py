"""Dataset-separated scientific figures assembled from validated metric tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..context import StageContext
from ..io import atomic_savefig
from .harmonize import METHOD_ORDER


COLORS = {"mixscape": "#0072B2", "mixscale": "#CC79A7", "ps": "#009E73"}
LABELS = {"mixscape": "Mixscape", "mixscale": "Mixscale", "ps": "PS"}
MODALITY = {"papalexi2021": "CRISPRko", "norman2019": "CRISPRa", "replogle2022": "CRISPRi"}


def run(context: StageContext) -> tuple[list[Path], dict[str, Any]]:
    tables = {
        "coverage": _read(context, "coverage", "coverage_summary.tsv"),
        "engagement_per": _read(context, "engagement", "engagement_per_perturbation.tsv"),
        "engagement_summary": _read(context, "engagement", "engagement_summary.tsv"),
        "effect": _read(context, "effect", "effect_association.tsv"),
        "effect_medians": _read(context, "effect", "perturbation_score_medians.tsv"),
        "effect_targets": _read(context, "effect", "transcriptomic_effect.tsv"),
        "repro_per": _read(context, "reproducibility", "reproducibility_per_perturbation.tsv"),
        "repro_summary": _read(context, "reproducibility", "reproducibility_summary.tsv"),
        "agreement_cell": _read(context, "agreement", "agreement_cell_summary.tsv"),
        "agreement_target": _read(context, "agreement", "agreement_target_medians.tsv"),
        "resources": _read(context, "resources", "resource_summary.tsv"),
    }
    artifacts: list[Path] = []
    for dataset_id in context.dataset_ids:
        output = context.stage_dir / f"dataset_id={dataset_id}"
        output.mkdir(parents=True, exist_ok=True)
        artifacts.extend(_main_figure(dataset_id, tables, output))
        artifacts.extend(_agreement_figure(dataset_id, tables, output))
        artifacts.extend(_resource_figure(dataset_id, tables, output))
        artifacts.extend(_effect_diagnostic(dataset_id, tables, output))
        if dataset_id != "norman2019":
            artifacts.extend(_reproducibility_diagnostic(dataset_id, tables, output))
        if dataset_id == "norman2019":
            artifacts.extend(_norman_strata_figure(tables, output))
    return artifacts, {"datasets": len(context.dataset_ids), "figure_files": len(artifacts)}


def _read(context: StageContext, stage: str, name: str) -> pd.DataFrame:
    path = context.upstream(stage, name)
    if not path.exists():
        raise RuntimeError(f"figures requires {stage}/{name}")
    return pd.read_csv(path, sep="\t")


def _main_figure(dataset: str, tables: dict[str, pd.DataFrame], output: Path) -> list[Path]:
    if dataset == "norman2019":
        figure, axes = plt.subplots(
            1, 3, figsize=(15.0, 6.1),
            gridspec_kw={"width_ratios": [0.92, 1.06, 1.12], "wspace": 0.42},
        )
        _coverage_panel(axes[0], dataset, tables)
        _distribution_panel(
            axes[1], dataset, tables["engagement_per"], tables["engagement_summary"],
            methods=METHOD_ORDER, value="rho_engagement", estimate="median_rho",
            panel="B", title="Within-perturbation target-expression correlation", ylabel="Within-perturbation Spearman ρ",
        )
        _forest_panel(
            axes[2], dataset, tables["effect"], METHOD_ORDER, "spearman_rho",
            panel="C", title="Across-perturbation transcriptome-effect correlation", xlabel="Score–E-distance Spearman ρ",
        )
        figure.suptitle(f"{dataset} · {MODALITY.get(dataset, '')} perturbation-status benchmark", fontsize=16, fontweight="bold", y=0.995)
        figure.subplots_adjust(left=0.07, right=0.985, bottom=0.14, top=0.84)
        return _save(figure, output / f"main_benchmark_{dataset}")
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.2))
    _coverage_panel(axes[0, 0], dataset, tables)
    _distribution_panel(
        axes[0, 1], dataset, tables["engagement_per"], tables["engagement_summary"],
        methods=METHOD_ORDER, value="rho_engagement", estimate="median_rho",
        panel="B", title="Within-perturbation target-expression correlation", ylabel="Within-perturbation Spearman ρ",
    )
    _forest_panel(
        axes[1, 0], dataset, tables["effect"], METHOD_ORDER, "spearman_rho",
        panel="C", title="Across-perturbation transcriptome-effect correlation", xlabel="Score–E-distance Spearman ρ",
    )
    repro_title = "Independent-construct profile gain"
    _distribution_panel(
        axes[1, 1], dataset, tables["repro_per"], tables["repro_summary"],
        methods=METHOD_ORDER, value="delta_rho", estimate="median_delta_rho",
        panel="D", title=repro_title, ylabel="Weighted − unweighted construct ρ", zero=True,
    )
    figure.suptitle(f"{dataset} · {MODALITY.get(dataset, '')} perturbation-status benchmark", fontsize=16, fontweight="bold", y=0.995)
    figure.text(0.5, 0.012, "Points and distributions are perturbation-level; intervals are 95% perturbation-bootstrap CIs.", ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0, 0.035, 1, 0.965), h_pad=3.0, w_pad=2.4)
    return _save(figure, output / f"main_benchmark_{dataset}")


def _coverage_panel(axis: plt.Axes, dataset: str, tables: dict[str, pd.DataFrame]) -> None:
    data = tables["coverage"].query("dataset_id == @dataset and stratum == 'all'").set_index("method_id").reindex(METHOD_ORDER)
    y = np.arange(len(METHOD_ORDER))[::-1]
    target = data["perturbation_coverage"].to_numpy()
    cells = data["targeted_cell_coverage"].to_numpy()
    for yi, method, left, right in zip(y, METHOD_ORDER, target, cells):
        axis.plot([left, right], [yi, yi], color="#A0A0A0", linewidth=2, zorder=1)
        axis.scatter(left, yi, s=70, color=COLORS[method], marker="o", zorder=2, label="Perturbations" if yi == y[0] else None)
        axis.scatter(right, yi, s=75, facecolor="white", edgecolor=COLORS[method], linewidth=2, marker="o", zorder=2, label="Targeted cells" if yi == y[0] else None)
        n, total = int(data.loc[method, "scorable_perturbations"]), int(data.loc[method, "represented_perturbations"])
        axis.text(max(left, right) + 0.025, yi, f"{n}/{total}", va="center", fontsize=8)
    axis.set_yticks(y, [LABELS[m] for m in METHOD_ORDER])
    axis.set_xlim(-0.03, 1.16)
    axis.set_xlabel("Coverage fraction")
    axis.set_title("A  Coverage and assessability", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    axis.legend(frameon=False, fontsize=8, loc="lower left")


def _distribution_panel(axis: plt.Axes, dataset: str, per: pd.DataFrame, summary: pd.DataFrame, methods: list[str], value: str, estimate: str, panel: str, title: str, ylabel: str, zero: bool = False) -> None:
    per = per[per["dataset_id"].eq(dataset)]
    summary = summary[summary["dataset_id"].eq(dataset) & summary["stratum"].eq("all")].set_index("method_id")
    positions = np.arange(len(methods))
    rng = np.random.default_rng(20260914)
    tick_labels = []
    for position, method in zip(positions, methods):
        values = per.loc[per["method_id"].eq(method), value].dropna().to_numpy()
        if len(values):
            axis.scatter(position + rng.uniform(-0.18, 0.18, len(values)), values,
                         s=8 if len(values) <= 180 else 5, color=COLORS[method],
                         alpha=0.24 if len(values) <= 180 else 0.14,
                         linewidth=0, rasterized=True)
        row = summary.loc[method]
        axis.errorbar(position, row[estimate], yerr=[[row[estimate] - row["ci_low"]], [row["ci_high"] - row[estimate]]], fmt="o", color=COLORS[method], markeredgecolor="black", markeredgewidth=0.5, markersize=7, capsize=3, linewidth=1.7, zorder=4)
        denominator = int(row["perturbations"] if "perturbations" in row else row["scorable_correlations"])
        tick_labels.append(f"{LABELS[method]}\n(n={denominator})")
    if zero:
        axis.axhline(0, color="#555555", linewidth=0.9, linestyle="--")
    axis.set_xticks(positions, tick_labels, rotation=18, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{panel}  {title}", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E8E8E8", linewidth=0.7)


def _forest_panel(axis: plt.Axes, dataset: str, table: pd.DataFrame, methods: list[str], estimate: str, panel: str, title: str, xlabel: str) -> None:
    data = table.query("dataset_id == @dataset and stratum == 'all'").set_index("method_id").reindex(methods)
    y = np.arange(len(methods))[::-1]
    for yi, method in zip(y, methods):
        row = data.loc[method]
        axis.errorbar(row[estimate], yi, xerr=[[row[estimate] - row["ci_low"]], [row["ci_high"] - row[estimate]]], fmt="o", color=COLORS[method], markeredgecolor="black", markeredgewidth=0.5, markersize=8, capsize=3, linewidth=1.8)
        axis.text(1.02, yi, f"n={int(row['perturbations'])}", transform=axis.get_yaxis_transform(), va="center", fontsize=8)
    axis.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    axis.set_xlim(-0.2, 1.0)
    axis.set_yticks(y, [LABELS[m] for m in methods])
    axis.set_xlabel(xlabel)
    axis.set_title(f"{panel}  {title}", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E8E8E8", linewidth=0.7)
    axis.text(0.0, -0.24, "Cross-target calibration; interpret target-specific score scales cautiously.", transform=axis.transAxes, fontsize=8, color="#666666")


def _agreement_figure(dataset: str, tables: dict[str, pd.DataFrame], output: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    image = None
    for axis, key, value, title in [
        (axes[0], "agreement_cell", "median_rho_cell", "Within-target cell ranking"),
        (axes[1], "agreement_target", "rho_target_median", "Target-median score ranking"),
    ]:
        data = tables[key].query("dataset_id == @dataset and stratum == 'all'")
        matrix = np.eye(len(METHOD_ORDER))
        for row in data.itertuples():
            i, j = METHOD_ORDER.index(row.method_left), METHOD_ORDER.index(row.method_right)
            matrix[i, j] = matrix[j, i] = getattr(row, value)
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        axis.set_xticks(range(len(METHOD_ORDER)), [LABELS[m] for m in METHOD_ORDER], rotation=35, ha="right")
        axis.set_yticks(range(len(METHOD_ORDER)), [LABELS[m] for m in METHOD_ORDER])
        axis.set_title(title, fontweight="bold")
        for i in range(len(METHOD_ORDER)):
            for j in range(len(METHOD_ORDER)):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.035, label="Spearman ρ")
    figure.suptitle(f"{dataset} · descriptive cross-method agreement", fontweight="bold")
    figure.text(0.5, 0.01, "Agreement describes concordance, not correctness.", ha="center", fontsize=9, color="#555555")
    figure.subplots_adjust(left=0.11, right=0.87, bottom=0.22, top=0.82, wspace=0.48)
    return _save(figure, output / f"supplement_agreement_{dataset}")


def _resource_figure(dataset: str, tables: dict[str, pd.DataFrame], output: Path) -> list[Path]:
    data = tables["resources"][tables["resources"]["dataset_id"].eq(dataset)].set_index("method_id").reindex(METHOD_ORDER)
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for axis, column, title in [
        (axes[0], "incremental_wall_seconds", "Incremental scoring time"),
        (axes[1], "standalone_wall_seconds", "Standalone end-to-end time"),
    ]:
        values = data[column].to_numpy() / 60
        bars = axis.bar(range(len(METHOD_ORDER)), np.nan_to_num(values), color=[COLORS[m] for m in METHOD_ORDER], alpha=0.82)
        for i, (bar, value) in enumerate(zip(bars, values)):
            if np.isnan(value):
                bar.set_facecolor("white"); bar.set_edgecolor(COLORS[METHOD_ORDER[i]]); bar.set_hatch("//")
                axis.text(i, 0, "not recorded", rotation=90, ha="center", va="bottom", fontsize=8)
            else:
                axis.text(i, value, f"{value:.1f}m", ha="center", va="bottom", fontsize=8)
        axis.set_xticks(range(len(METHOD_ORDER)), [LABELS[m] for m in METHOD_ORDER], rotation=25, ha="right")
        axis.set_ylabel("Wall time (minutes)")
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.7)
    figure.suptitle(f"{dataset} · recorded computational cost", fontweight="bold")
    figure.text(0.5, 0.01, "Missing timing is retained as missing; memory upper bounds are reported in the resource table.", ha="center", fontsize=8.5, color="#555555")
    figure.tight_layout(rect=(0, 0.05, 1, 0.92), w_pad=2.5)
    return _save(figure, output / f"supplement_resources_{dataset}")


def _effect_diagnostic(dataset: str, tables: dict[str, pd.DataFrame], output: Path) -> list[Path]:
    medians = tables["effect_medians"][tables["effect_medians"]["dataset_id"].eq(dataset)]
    effects = tables["effect_targets"][tables["effect_targets"]["dataset_id"].eq(dataset)]
    data = medians.merge(effects[["target_id", "energy_distance", "targeted_cells"]], on="target_id", how="inner")
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.8), sharex=True, sharey=True)
    summary = tables["effect"].query("dataset_id == @dataset and stratum == 'all'").set_index("method_id")
    for axis, method in zip(axes.flat, METHOD_ORDER):
        selected = data[data["method_id"].eq(method)].dropna(subset=["median_score", "energy_distance"])
        x = selected["median_score"].rank(pct=True).to_numpy()
        y = selected["energy_distance"].rank(pct=True).to_numpy()
        axis.hexbin(x, y, gridsize=28, mincnt=1, cmap="Greys", linewidths=0, alpha=0.85)
        row = summary.loc[method]
        axis.text(0.04, 0.94, f"ρ={row.spearman_rho:.2f}\nn={int(row.perturbations)}", transform=axis.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        axis.set_title(LABELS[method], color=COLORS[method], fontweight="bold")
        axis.grid(color="#EEEEEE", linewidth=0.6)
    figure.supxlabel("Within-method percentile rank of target median score")
    figure.supylabel("Percentile rank of transcriptomic E-distance")
    figure.suptitle(f"{dataset} · target-level effect calibration diagnostics", fontweight="bold")
    figure.tight_layout(rect=(0.04, 0.04, 1, 0.94))
    return _save(figure, output / f"supplement_effect_diagnostics_{dataset}")


def _reproducibility_diagnostic(dataset: str, tables: dict[str, pd.DataFrame], output: Path) -> list[Path]:
    data = tables["repro_per"][tables["repro_per"]["dataset_id"].eq(dataset)]
    summary = tables["repro_summary"].query("dataset_id == @dataset and stratum == 'all'").set_index("method_id")
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.8), sharex=True, sharey=True)
    bounds = data[["rho_unweighted", "rho_weighted"]].to_numpy()
    low, high = np.nanquantile(bounds, [0.005, 0.995]) if len(bounds) else (-1, 1)
    margin = max((high - low) * 0.05, 0.03)
    for axis, method in zip(axes.flat, METHOD_ORDER):
        selected = data[data["method_id"].eq(method)].dropna(subset=["rho_unweighted", "rho_weighted"])
        axis.scatter(selected["rho_unweighted"], selected["rho_weighted"], s=8, color=COLORS[method], alpha=0.25, linewidth=0, rasterized=True)
        axis.plot([low - margin, high + margin], [low - margin, high + margin], color="#555555", linestyle="--", linewidth=0.9)
        row = summary.loc[method]
        axis.text(0.04, 0.94, f"median Δρ={row.median_delta_rho:.3f}\nn={int(row.perturbations)}", transform=axis.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        axis.set_title(LABELS[method], color=COLORS[method], fontweight="bold")
        axis.set_xlim(low - margin, high + margin); axis.set_ylim(low - margin, high + margin)
        axis.grid(color="#EEEEEE", linewidth=0.6)
    figure.supxlabel("Unweighted independent-construct Spearman ρ")
    figure.supylabel("Weighted independent-construct Spearman ρ")
    figure.suptitle(f"{dataset} · per-perturbation construct-profile reproducibility", fontweight="bold")
    figure.tight_layout(rect=(0.04, 0.04, 1, 0.94))
    return _save(figure, output / f"supplement_reproducibility_diagnostics_{dataset}")


def _norman_strata_figure(tables: dict[str, pd.DataFrame], output: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    specifications = [
        (axes[0], tables["engagement_summary"], "median_rho", "Within-perturbation target-expression correlation"),
        (axes[1], tables["effect"], "spearman_rho", "Across-perturbation transcriptome-effect correlation"),
    ]
    methods = METHOD_ORDER
    offsets = {"single_gene": -0.13, "multi_gene": 0.13}
    markers = {"single_gene": "o", "multi_gene": "s"}
    for axis, table, estimate, title in specifications:
        data = table[table["dataset_id"].eq("norman2019")]
        for i, method in enumerate(methods):
            for stratum in ["single_gene", "multi_gene"]:
                row = data[data["method_id"].eq(method) & data["stratum"].eq(stratum)].iloc[0]
                axis.errorbar(i + offsets[stratum], row[estimate], yerr=[[row[estimate] - row.ci_low], [row.ci_high - row[estimate]]], fmt=markers[stratum], color=COLORS[method], markeredgecolor="black", markeredgewidth=0.4, capsize=3, label=stratum.replace("_", " ") if i == 0 else None)
        axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        axis.set_xticks(range(len(methods)), [LABELS[m] for m in methods])
        axis.set_ylabel("Spearman ρ")
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.7)
    axes[0].legend(frameon=False)
    figure.suptitle("norman2019 · single- versus multi-gene perturbations", fontweight="bold")
    figure.text(0.5, 0.01, "Construct-profile reproducibility is not applicable because positional constructs share the targeting guide.", ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0, 0.05, 1, 0.92), w_pad=2.5)
    return _save(figure, output / "supplement_strata_norman2019")


def _save(figure: plt.Figure, stem: Path) -> list[Path]:
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    atomic_savefig(figure, png, dpi=240, bbox_inches="tight", facecolor="white")
    atomic_savefig(figure, pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png, pdf]
