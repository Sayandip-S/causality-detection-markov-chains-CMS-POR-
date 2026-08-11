from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.storm.model_utils import PROJECT_ROOT


DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "results/systematic/brp_stress_error/sample_size"
)
MODEL_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest"]
MODEL_COLORS = {
    "Logistic Regression": "#1f77b4",
    "Decision Tree": "#d62728",
    "Random Forest": "#2ca02c",
}


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Save one non-empty figure without replacing an existing plot."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing plot: {path}")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    if path.stat().st_size == 0:
        raise RuntimeError(f"Generated plot is empty: {path}")


def set_sample_ticks(axis: plt.Axes, values: Iterable[int]) -> None:
    sizes = sorted({int(value) for value in values})
    axis.set_xticks(sizes)
    axis.set_xticklabels([f"{value:,}" for value in sizes])


def model_metric_plot(
    aggregated: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    *,
    reference: float | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for model in MODEL_ORDER:
        rows = aggregated[aggregated["model"] == model].sort_values(
            "training_sample_size"
        )
        axis.errorbar(
            rows["training_sample_size"],
            rows[f"{metric}_mean"],
            yerr=rows[f"{metric}_std"],
            marker="o",
            capsize=4,
            linewidth=1.8,
            label=model,
            color=MODEL_COLORS[model],
        )
    if reference is not None:
        axis.axhline(reference, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Training sample size")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    set_sample_ticks(axis, aggregated["training_sample_size"])
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save_figure(figure, path)


def candidate_metric_plot(
    aggregated: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    *,
    reference_at_full: bool = False,
) -> None:
    rows = aggregated.sort_values("training_sample_size")
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.errorbar(
        rows["training_sample_size"],
        rows[f"{metric}_mean"],
        yerr=rows[f"{metric}_std"],
        marker="o",
        capsize=4,
        linewidth=1.8,
        color="#6f42c1",
    )
    if reference_at_full:
        full = rows.loc[rows["training_sample_size"].idxmax(), f"{metric}_mean"]
        axis.axhline(full, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Training sample size")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    set_sample_ticks(axis, rows["training_sample_size"])
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, path)


def ranking_method_plot(comparison: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    labels = {
        "combined": "Combined",
        "empirical_difference_only": "Empirical difference",
        "frequency_support_only": "Frequency/support",
        "logistic_coefficient_only": "Logistic coefficient",
        "random_forest_importance_only": "RF importance",
    }
    for method, rows in comparison.groupby("ranking_method"):
        summary = rows.groupby("training_sample_size")[
            "exact_probability_raising_count"
        ].agg(["mean", "std"]).fillna(0).reset_index()
        axis.errorbar(
            summary["training_sample_size"],
            summary["mean"],
            yerr=summary["std"],
            marker="o",
            capsize=3,
            label=labels.get(method, method),
        )
    axis.set_xlabel("Training sample size")
    axis.set_ylabel("Exact probability-raising states (top 20)")
    axis.set_title("Exact quality of candidate-ranking methods")
    set_sample_ticks(axis, comparison["training_sample_size"])
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    save_figure(figure, path)


def reliability_plot(reliability: pd.DataFrame, path: Path) -> None:
    criteria = [
        "f1_within_0_05_of_full",
        "roc_auc_within_0_02_of_full",
        "f1_standard_deviation_below_0_05",
        "top10_overlap_at_least_0_70",
        "top20_overlap_at_least_0_60",
        "at_least_15_exact_probability_raising",
        "exact_pass_count_standard_deviation_below_2",
    ]
    labels = [
        "F1 gap",
        "ROC-AUC gap",
        "F1 SD",
        "Top-10 overlap",
        "Top-20 overlap",
        "Exact pass count",
        "Exact pass SD",
    ]
    ordered = reliability.sort_values(["training_sample_size", "model"])
    row_labels = [
        f"{int(row.training_sample_size):,} — {row.model}"
        for row in ordered.itertuples()
    ]
    matrix = ordered[criteria].astype(int).to_numpy()
    figure_height = max(5.5, 0.28 * len(ordered))
    figure, axis = plt.subplots(figsize=(9.2, figure_height))
    image = axis.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title("Analyst-selected operational reliability criteria")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                "pass" if matrix[row, column] else "fail",
                ha="center",
                va="center",
                fontsize=7,
            )
    figure.colorbar(image, ax=axis, ticks=[0, 1], label="criterion result")
    save_figure(figure, path)


def plot_experiment(input_root: Path, output_directory: Path) -> list[Path]:
    """Generate every required figure from experiment CSV artifacts."""

    input_root = input_root.resolve()
    output_directory = output_directory.resolve()
    prediction = pd.read_csv(input_root / "prediction_aggregated.csv")
    stability = pd.read_csv(input_root / "candidate_stability_aggregated.csv")
    quality = pd.read_csv(input_root / "exact_candidate_quality_aggregated.csv")
    comparison = pd.read_csv(input_root / "ranking_method_comparison.csv")
    reliability = pd.read_csv(input_root / "reliability_assessment.csv")
    if quality.empty or comparison.empty or reliability.empty:
        raise ValueError("Required exact, baseline, or reliability results are empty.")

    paths = {
        name: output_directory / name
        for name in (
            "f1_vs_training_sample_size.png",
            "roc_auc_vs_training_sample_size.png",
            "precision_vs_training_sample_size.png",
            "recall_vs_training_sample_size.png",
            "model_training_time_vs_sample_size.png",
            "top10_candidate_overlap_vs_sample_size.png",
            "top20_candidate_overlap_vs_sample_size.png",
            "candidate_rank_correlation_vs_sample_size.png",
            "exact_probability_raising_count_vs_sample_size.png",
            "mean_exact_probability_increase_vs_sample_size.png",
            "mean_exact_candidate_reachability_vs_sample_size.png",
            "ranking_method_exact_pass_comparison.png",
            "reliability_criteria_heatmap.png",
        )
    }
    model_metric_plot(
        prediction, "f1", "F1", "F1 by training sample size", paths["f1_vs_training_sample_size.png"]
    )
    model_metric_plot(
        prediction,
        "roc_auc",
        "ROC-AUC",
        "ROC-AUC by training sample size",
        paths["roc_auc_vs_training_sample_size.png"],
        reference=0.5,
    )
    model_metric_plot(
        prediction,
        "precision",
        "Precision",
        "Precision by training sample size",
        paths["precision_vs_training_sample_size.png"],
    )
    model_metric_plot(
        prediction,
        "recall",
        "Recall",
        "Recall by training sample size",
        paths["recall_vs_training_sample_size.png"],
    )
    model_metric_plot(
        prediction,
        "model_training_seconds",
        "Training time (seconds)",
        "Model training time by sample size",
        paths["model_training_time_vs_sample_size.png"],
    )
    candidate_metric_plot(
        stability,
        "top10_overlap_fraction",
        "Top-10 overlap fraction",
        "Top-10 overlap with full-training ranking",
        paths["top10_candidate_overlap_vs_sample_size.png"],
    )
    candidate_metric_plot(
        stability,
        "top20_overlap_fraction",
        "Top-20 overlap fraction",
        "Top-20 overlap with full-training ranking",
        paths["top20_candidate_overlap_vs_sample_size.png"],
    )
    candidate_metric_plot(
        stability,
        "spearman_rank_correlation",
        "Spearman correlation on shared states",
        "Candidate rank correlation with full training",
        paths["candidate_rank_correlation_vs_sample_size.png"],
    )
    candidate_metric_plot(
        quality,
        "exact_probability_raising_count",
        "Exact probability-raising states (out of 20)",
        "Exact probability-raising quality",
        paths["exact_probability_raising_count_vs_sample_size.png"],
        reference_at_full=True,
    )
    candidate_metric_plot(
        quality,
        "mean_exact_probability_difference",
        "Mean exact target-probability difference",
        "Mean exact probability increase of top candidates",
        paths["mean_exact_probability_increase_vs_sample_size.png"],
    )
    candidate_metric_plot(
        quality,
        "mean_exact_candidate_reachability",
        "Mean exact candidate reachability",
        "Mean exact reachability of top candidates",
        paths["mean_exact_candidate_reachability_vs_sample_size.png"],
    )
    ranking_method_plot(comparison, paths["ranking_method_exact_pass_comparison.png"])
    reliability_plot(reliability, paths["reliability_criteria_heatmap.png"])
    return list(paths.values())


def presentation_sample_labels(values: Iterable[int]) -> list[str]:
    """Return compact, slide-readable sample-size labels."""

    labels = []
    for value in values:
        size = int(value)
        if size < 1000:
            labels.append(str(size))
        elif size % 1000 == 0:
            labels.append(f"{size // 1000}k")
        else:
            labels.append(f"{size / 1000:.1f}k")
    return labels


def save_presentation_figure(figure: plt.Figure, path: Path) -> None:
    """Save a high-resolution presentation figure without overwriting it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing presentation plot: {path}"
        )
    figure.tight_layout()
    figure.savefig(path, dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    if path.stat().st_size == 0:
        raise RuntimeError(f"Generated presentation plot is empty: {path}")


def configure_presentation_sample_axis(
    axis: plt.Axes,
    sample_sizes: Iterable[int],
    *,
    logarithmic: bool = True,
) -> None:
    """Configure numeric sample positions with compact labels."""

    sizes = sorted({int(value) for value in sample_sizes})
    if logarithmic:
        axis.set_xscale("log")
        axis.set_xlim(min(sizes) * 0.88, max(sizes) * 1.14)
    axis.set_xticks(sizes)
    axis.set_xticklabels(presentation_sample_labels(sizes))
    axis.minorticks_off()
    axis.set_xlabel("Training sample size")


def presentation_prediction_plot(
    prediction: pd.DataFrame,
    metric: str,
    path: Path,
) -> None:
    """Plot presentation-ready F1 or ROC-AUC stability."""

    figure, axis = plt.subplots(figsize=(11.5, 6.4))
    full_size = int(prediction["training_sample_size"].max())
    for model in MODEL_ORDER:
        rows = prediction[prediction["model"] == model].sort_values(
            "training_sample_size"
        )
        axis.errorbar(
            rows["training_sample_size"],
            rows[f"{metric}_mean"],
            yerr=rows[f"{metric}_std"],
            marker="o",
            markersize=8,
            capsize=5,
            linewidth=2.3,
            label=model,
            color=MODEL_COLORS[model],
        )
        if metric == "f1":
            full_value = float(
                rows.loc[
                    rows["training_sample_size"] == full_size,
                    f"{metric}_mean",
                ].iloc[0]
            )
            axis.axhline(
                full_value,
                color=MODEL_COLORS[model],
                linestyle="--",
                linewidth=1.2,
                alpha=0.45,
            )
            axis.text(
                0.99,
                full_value,
                f" {model.replace('Logistic Regression', 'LR').replace('Decision Tree', 'DT').replace('Random Forest', 'RF')} full",
                transform=axis.get_yaxis_transform(),
                color=MODEL_COLORS[model],
                fontsize=10,
                alpha=0.8,
                va="center",
                ha="right",
            )

    configure_presentation_sample_axis(
        axis, prediction["training_sample_size"], logarithmic=True
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(loc="best", frameon=False, ncol=3)
    axis.text(
        0.01,
        0.02,
        "Error bars: ±1 standard deviation",
        transform=axis.transAxes,
        fontsize=11,
        color="#444444",
    )
    if metric == "f1":
        axis.set_ylabel("Mean F1")
        axis.set_title("Prediction F1 stability across training sample sizes")
        axis.text(
            0.01,
            0.96,
            "Higher small-sample F1 can reflect threshold and recall changes; it does not imply better discrimination.",
            transform=axis.transAxes,
            va="top",
            fontsize=11,
            color="#444444",
        )
    else:
        axis.axhline(
            0.5,
            color="#333333",
            linestyle="--",
            linewidth=1.5,
            label="Random ranking (0.5)",
        )
        values = pd.concat(
            [
                prediction[f"{metric}_mean"] - prediction[f"{metric}_std"],
                prediction[f"{metric}_mean"] + prediction[f"{metric}_std"],
            ]
        )
        lower = math.floor((float(values.min()) - 0.005) * 100) / 100
        upper = math.ceil((float(values.max()) + 0.005) * 100) / 100
        axis.set_ylim(lower, upper)
        axis.set_ylabel("Mean ROC-AUC")
        axis.set_title("Prediction ROC-AUC remains near chance")
        axis.text(
            0.01,
            0.96,
            "All values remain near random ranking; the narrowed axis is shown explicitly.",
            transform=axis.transAxes,
            va="top",
            fontsize=11,
            color="#444444",
        )
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles, labels, loc="best", frameon=False, ncol=2)
    save_presentation_figure(figure, path)


def presentation_candidate_stability_plot(
    stability: pd.DataFrame,
    thresholds: dict[str, float],
    path: Path,
) -> None:
    """Plot top-10 and top-20 overlap without mixing rank correlation."""

    rows = stability.sort_values("training_sample_size")
    positions = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(11.5, 6.4))
    series = [
        ("top10_overlap_fraction", "Top-10 overlap", "#7b2cbf", -0.09),
        ("top20_overlap_fraction", "Top-20 overlap", "#168aad", 0.09),
    ]
    for metric, label, color, offset in series:
        axis.errorbar(
            positions + offset,
            rows[f"{metric}_mean"],
            yerr=rows[f"{metric}_std"],
            fmt="o",
            markersize=9,
            capsize=5,
            elinewidth=2,
            label=label,
            color=color,
        )
    axis.axhline(
        thresholds["top10_overlap_fraction_min"],
        color="#7b2cbf",
        linestyle="--",
        linewidth=1.5,
        alpha=0.65,
        label="Top-10 criterion",
    )
    axis.axhline(
        thresholds["top20_overlap_fraction_min"],
        color="#168aad",
        linestyle=":",
        linewidth=1.8,
        alpha=0.75,
        label="Top-20 criterion",
    )
    axis.set_xticks(
        positions,
        presentation_sample_labels(rows["training_sample_size"]),
    )
    axis.set_xlabel("Training sample size")
    axis.set_ylabel("Overlap with full-training ranking")
    axis.set_ylim(0, 1.08)
    axis.set_title("Candidate identity stabilizes more slowly than prediction metrics")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2)
    axis.text(
        0.01,
        0.02,
        "Fractions show |sample top-k ∩ full top-k| / k; error bars are ±1 SD.",
        transform=axis.transAxes,
        fontsize=11,
        color="#444444",
    )
    save_presentation_figure(figure, path)


def presentation_rank_correlation_plot(
    stability: pd.DataFrame,
    path: Path,
) -> None:
    """Plot Spearman rank correlation separately from set overlap."""

    rows = stability.sort_values("training_sample_size")
    figure, axis = plt.subplots(figsize=(11.5, 6.4))
    axis.errorbar(
        rows["training_sample_size"],
        rows["spearman_rank_correlation_mean"],
        yerr=rows["spearman_rank_correlation_std"],
        marker="o",
        markersize=9,
        capsize=5,
        linewidth=2.3,
        color="#6a4c93",
    )
    full_reference = float(
        rows.loc[
            rows["training_sample_size"] == rows["training_sample_size"].max(),
            "spearman_rank_correlation_mean",
        ].iloc[0]
    )
    axis.axhline(0, color="#555555", linewidth=1.2)
    axis.axhline(
        full_reference,
        color="#333333",
        linestyle="--",
        linewidth=1.4,
        label=f"Full reference ({full_reference:.0f})",
    )
    configure_presentation_sample_axis(
        axis, rows["training_sample_size"], logarithmic=True
    )
    axis.set_ylabel("Spearman correlation on shared top-20 states")
    axis.set_ylim(-0.65, 1.12)
    axis.set_title("Candidate ordering depends strongly on sampled traces")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    axis.text(
        0.01,
        0.02,
        "Large error bars indicate that ranking order changes substantially across sampled training sets.",
        transform=axis.transAxes,
        fontsize=11,
        color="#444444",
    )
    save_presentation_figure(figure, path)


def presentation_exact_quality_plot(
    quality: pd.DataFrame,
    path: Path,
) -> None:
    """Plot the exact probability-raising pass count out of top 20."""

    rows = quality.sort_values("training_sample_size")
    metric = "exact_probability_raising_count"
    figure, axis = plt.subplots(figsize=(11.5, 6.4))
    axis.errorbar(
        rows["training_sample_size"],
        rows[f"{metric}_mean"],
        yerr=rows[f"{metric}_std"],
        marker="o",
        markersize=9,
        capsize=5,
        linewidth=2.5,
        color="#2a9d8f",
    )
    full_value = float(
        rows.loc[
            rows["training_sample_size"] == rows["training_sample_size"].max(),
            f"{metric}_mean",
        ].iloc[0]
    )
    axis.axhline(
        full_value,
        color="#333333",
        linestyle="--",
        linewidth=1.5,
        label=f"Full-training value ({full_value:.0f})",
    )
    for row in rows.itertuples():
        axis.annotate(
            f"{getattr(row, f'{metric}_mean'):.1f}",
            (row.training_sample_size, getattr(row, f"{metric}_mean")),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=11,
            weight="bold",
        )
    configure_presentation_sample_axis(
        axis, rows["training_sample_size"], logarithmic=True
    )
    axis.set_ylabel("Verified candidates out of 20")
    axis.set_ylim(0, 20)
    axis.set_title("Exact Storm probability-raising quality improves with sample size")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    axis.text(
        0.01,
        0.02,
        "Error bars: ±1 standard deviation across sampled training sets.",
        transform=axis.transAxes,
        fontsize=11,
        color="#444444",
    )
    save_presentation_figure(figure, path)


def presentation_ranking_method_plot(
    comparison: pd.DataFrame,
    random_distribution: pd.DataFrame,
    path: Path,
) -> None:
    """Compare deterministic and random ranking methods with grouped bars."""

    deterministic = (
        comparison.groupby(["training_sample_size", "ranking_method"])[
            "exact_probability_raising_count"
        ]
        .agg(["mean", "std"])
        .fillna(0)
    )
    random = (
        random_distribution.groupby("training_sample_size")[
            "exact_probability_raising_count"
        ]
        .agg(["mean", "std"])
        .fillna(0)
    )
    sizes = sorted(comparison["training_sample_size"].astype(int).unique())
    methods = [
        ("combined", "Combined", "#6a4c93", ""),
        ("empirical_difference_only", "Empirical only", "#0077b6", "//"),
        ("logistic_coefficient_only", "LR only", "#f4a261", ""),
        ("random_forest_importance_only", "RF only", "#2a9d8f", ""),
        ("frequency_support_only", "Frequency only", "#9c6644", ""),
        ("random", "Random", "#adb5bd", ".."),
    ]
    positions = np.arange(len(sizes))
    width = 0.13
    figure, axis = plt.subplots(figsize=(13.5, 6.8))
    for method_index, (method, label, color, hatch) in enumerate(methods):
        offset = (method_index - (len(methods) - 1) / 2) * width
        if method == "random":
            means = [float(random.loc[size, "mean"]) for size in sizes]
            errors = [float(random.loc[size, "std"]) for size in sizes]
        else:
            means = [float(deterministic.loc[(size, method), "mean"]) for size in sizes]
            errors = [float(deterministic.loc[(size, method), "std"]) for size in sizes]
        axis.bar(
            positions + offset,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="#333333",
            linewidth=0.5,
        )
    axis.set_xticks(positions, presentation_sample_labels(sizes))
    axis.set_xlabel("Training sample size")
    axis.set_ylabel("Mean verified candidates out of 20")
    axis.set_ylim(0, 20)
    axis.set_title("Exact quality differs across candidate-ranking methods")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    axis.text(
        0.99,
        0.02,
        "Bars show means; error bars are ±1 SD.",
        transform=axis.transAxes,
        ha="right",
        fontsize=11,
        color="#444444",
    )
    save_presentation_figure(figure, path)


def prediction_criteria_pass(row: pd.Series) -> bool:
    """Return whether all prediction-relative criteria pass for one model."""

    return bool(
        row["f1_within_0_05_of_full"]
        and row["roc_auc_within_0_02_of_full"]
        and row["f1_standard_deviation_below_0_05"]
    )


def presentation_reliability_plot(
    reliability: pd.DataFrame,
    path: Path,
) -> None:
    """Summarize model and candidate reliability by sample size."""

    sizes = sorted(reliability["training_sample_size"].astype(int).unique())
    columns = [
        "LR prediction\ncriteria",
        "DT prediction\ncriteria",
        "RF prediction\ncriteria",
        "Top-10\noverlap",
        "Top-20\noverlap",
        "Exact pass\ncount",
        "Candidate\nvariance",
        "All\ncriteria",
    ]
    matrix: list[list[int]] = []
    model_columns = ["Logistic Regression", "Decision Tree", "Random Forest"]
    for size in sizes:
        rows = reliability[reliability["training_sample_size"] == size]
        model_passes = [
            prediction_criteria_pass(rows[rows["model"] == model].iloc[0])
            for model in model_columns
        ]
        candidate = rows.iloc[0]
        candidate_passes = [
            bool(candidate["top10_overlap_at_least_0_70"]),
            bool(candidate["top20_overlap_at_least_0_60"]),
            bool(candidate["at_least_15_exact_probability_raising"]),
            bool(candidate["exact_pass_count_standard_deviation_below_2"]),
        ]
        all_pass = all([*model_passes, *candidate_passes])
        matrix.append([int(value) for value in [*model_passes, *candidate_passes, all_pass]])
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(13, 6.5))
    color_map = matplotlib.colors.ListedColormap(["#e76f51", "#52b788"])
    axis.imshow(values, aspect="auto", cmap=color_map, vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(columns)), columns)
    axis.set_yticks(np.arange(len(sizes)), [f"{size:,}" for size in sizes])
    axis.set_ylabel("Training sample size")
    axis.set_title("Operational reliability criteria: prediction stabilizes before candidates")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                "PASS" if values[row, column] else "FAIL",
                ha="center",
                va="center",
                weight="bold",
                color="white",
                fontsize=10,
            )
    axis.legend(
        handles=[
            Patch(facecolor="#52b788", label="Pass"),
            Patch(facecolor="#e76f51", label="Fail"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        ncol=2,
    )
    axis.text(
        0.0,
        -0.25,
        "Thresholds are analyst-selected operational criteria, not universal laws.",
        transform=axis.transAxes,
        fontsize=11,
        color="#444444",
    )
    save_presentation_figure(figure, path)


def presentation_runtime_plot(
    prediction: pd.DataFrame,
    metadata: dict[str, object],
    path: Path,
) -> None:
    """Plot model-fitting time and annotate the recorded runner runtime."""

    figure, axis = plt.subplots(figsize=(11.5, 6.4))
    for model in MODEL_ORDER:
        rows = prediction[prediction["model"] == model].sort_values(
            "training_sample_size"
        )
        axis.errorbar(
            rows["training_sample_size"],
            rows["model_training_seconds_mean"],
            yerr=rows["model_training_seconds_std"],
            marker="o",
            markersize=8,
            capsize=5,
            linewidth=2.3,
            label=model,
            color=MODEL_COLORS[model],
        )
    configure_presentation_sample_axis(
        axis, prediction["training_sample_size"], logarithmic=True
    )
    axis.set_yscale("log")
    axis.set_ylabel("Mean model-training time (seconds, log scale)")
    axis.set_title("Model-fitting cost by training sample size")
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    runtime = float(metadata["total_runtime_seconds"])
    axis.text(
        0.01,
        0.03,
        f"Recorded full experiment runner: {runtime:.2f} s. Uses an existing dataset; excludes trace generation and presentation plotting.",
        transform=axis.transAxes,
        fontsize=11,
        color="#444444",
    )
    save_presentation_figure(figure, path)


def plot_presentation_experiment(
    input_root: Path,
    output_directory: Path,
) -> list[Path]:
    """Generate the presentation-focused plot set from persisted results."""

    input_root = input_root.resolve()
    output_directory = output_directory.resolve()
    prediction = pd.read_csv(input_root / "prediction_aggregated.csv")
    stability = pd.read_csv(input_root / "candidate_stability_aggregated.csv")
    quality = pd.read_csv(input_root / "exact_candidate_quality_aggregated.csv")
    comparison = pd.read_csv(input_root / "ranking_method_comparison.csv")
    random_distribution = pd.read_csv(input_root / "random_baseline_distribution.csv")
    reliability = pd.read_csv(input_root / "reliability_assessment.csv")
    metadata = json.loads(
        (input_root / "prediction_metadata.json").read_text(encoding="utf-8")
    )
    required_frames = [
        prediction,
        stability,
        quality,
        comparison,
        random_distribution,
        reliability,
    ]
    if any(frame.empty for frame in required_frames):
        raise ValueError("Presentation plotting requires non-empty full-run artifacts.")
    paths = {
        name: output_directory / name
        for name in (
            "prediction_f1_stability.png",
            "prediction_roc_auc_stability.png",
            "candidate_stability_summary.png",
            "candidate_rank_correlation.png",
            "exact_candidate_quality.png",
            "ranking_method_comparison.png",
            "reliability_summary.png",
            "runtime_summary.png",
        )
    }
    style = {
        "font.size": 13,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    }
    with plt.rc_context(style):
        presentation_prediction_plot(
            prediction, "f1", paths["prediction_f1_stability.png"]
        )
        presentation_prediction_plot(
            prediction, "roc_auc", paths["prediction_roc_auc_stability.png"]
        )
        presentation_candidate_stability_plot(
            stability,
            metadata["reliability_thresholds"],
            paths["candidate_stability_summary.png"],
        )
        presentation_rank_correlation_plot(
            stability, paths["candidate_rank_correlation.png"]
        )
        presentation_exact_quality_plot(
            quality, paths["exact_candidate_quality.png"]
        )
        presentation_ranking_method_plot(
            comparison,
            random_distribution,
            paths["ranking_method_comparison.png"],
        )
        presentation_reliability_plot(
            reliability, paths["reliability_summary.png"]
        )
        presentation_runtime_plot(
            prediction, metadata, paths["runtime_summary.png"]
        )
    return list(paths.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BRP sample-size experiment results.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="Generate the eight presentation-focused plots.",
    )
    args = parser.parse_args()
    default_directory = "presentation_plots" if args.presentation else "plots"
    output = args.output_directory or args.input_root / default_directory
    if args.presentation:
        paths = plot_presentation_experiment(args.input_root, output)
    else:
        paths = plot_experiment(args.input_root, output)
    print(f"Generated {len(paths)} plots under {output.resolve()}.")


if __name__ == "__main__":
    main()
