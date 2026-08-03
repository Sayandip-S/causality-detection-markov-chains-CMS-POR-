#!/usr/bin/env python3
"""Generate presentation-ready plots for the current BRP experiments."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/brp_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = PROJECT_ROOT / "results/systematic/brp_stress_error/plots"
REPORT_PATH = (
    PROJECT_ROOT
    / "results/systematic/brp_stress_error/reports/current_plot_summary.md"
)

COMMON_METRICS_PATH = (
    PROJECT_ROOT
    / "results/systematic/brp_stress_error/metrics/common_cohort_per_model.csv"
)
COMPARISON_PATH = (
    PROJECT_ROOT
    / "results/systematic/brp_stress_error/metrics/operational_vs_common_cohort.csv"
)
COMMON_MANIFEST_PATH = (
    PROJECT_ROOT
    / "results/systematic/brp_stress_error/metrics/common_cohort_manifest.json"
)
CANDIDATE_PATH = (
    PROJECT_ROOT
    / "results/systematic/brp_stress_error/reachability/candidate_exact_reachability.csv"
)
CANDIDATE_METADATA_PATH = CANDIDATE_PATH.with_suffix(".metadata.json")
OPERATIONAL_PATHS = {
    k: PROJECT_ROOT / f"results/metrics/brp_fixed_windows/k{k}.json"
    for k in (5, 10, 20, 50)
}

MODEL_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest"]
MODEL_STYLE = {
    "Logistic Regression": ("#0072B2", "o"),
    "Decision Tree": ("#D55E00", "s"),
    "Random Forest": ("#009E73", "^"),
}


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{relative(source)} is missing columns: {sorted(missing)}")


def load_inputs() -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    dict | None,
    dict | None,
    pd.DataFrame | None,
]:
    common = pd.read_csv(COMMON_METRICS_PATH) if COMMON_METRICS_PATH.is_file() else None
    comparison = pd.read_csv(COMPARISON_PATH) if COMPARISON_PATH.is_file() else None
    candidates = pd.read_csv(CANDIDATE_PATH) if CANDIDATE_PATH.is_file() else None
    manifest = (
        json.loads(COMMON_MANIFEST_PATH.read_text(encoding="utf-8"))
        if COMMON_MANIFEST_PATH.is_file()
        else None
    )
    candidate_metadata = (
        json.loads(CANDIDATE_METADATA_PATH.read_text(encoding="utf-8"))
        if CANDIDATE_METADATA_PATH.is_file()
        else None
    )

    if common is not None:
        require_columns(
            common,
            {"observation_window", "model", "feature_count", "f1", "roc_auc"},
            COMMON_METRICS_PATH,
        )
    if comparison is not None:
        require_columns(
            comparison,
            {
                "observation_window",
                "model",
                "f1_common_cohort",
                "f1_operational",
                "roc_auc_common_cohort",
                "roc_auc_operational",
            },
            COMPARISON_PATH,
        )
    if candidates is not None:
        require_columns(
            candidates,
            {
                "original_rank",
                "state_id",
                "empirical_support_fraction",
                "exact_candidate_reachability",
                "baseline_target_probability",
                "target_probability_from_candidate",
                "probability_difference_from_baseline",
                "raises_probability_from_state",
                "risk_weighted_coverage",
            },
            CANDIDATE_PATH,
        )

    operational = None
    if all(path.is_file() for path in OPERATIONAL_PATHS.values()):
        operational_rows: list[dict[str, Any]] = []
        for window, path in OPERATIONAL_PATHS.items():
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            reference = metrics[MODEL_ORDER[0]]
            invariant_fields = (
                "training_row_count",
                "test_row_count",
                "positive_rate",
            )
            for model in MODEL_ORDER[1:]:
                for field in invariant_fields:
                    if metrics[model][field] != reference[field]:
                        raise ValueError(
                            f"{relative(path)} has model-dependent {field}"
                        )
            total = int(
                reference["training_row_count"] + reference["test_row_count"]
            )
            target = round(total * float(reference["positive_rate"]))
            if abs(target / total - float(reference["positive_rate"])) > 1e-12:
                raise ValueError(f"{relative(path)} has an inconsistent positive rate")
            operational_rows.append(
                {
                    "observation_window": window,
                    "total_traces": total,
                    "target_traces": target,
                    "success_traces": total - target,
                    "target_rate": float(reference["positive_rate"]),
                }
            )
        operational = pd.DataFrame(operational_rows).sort_values("observation_window")
    return common, comparison, candidates, manifest, candidate_metadata, operational


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10.5, 6.4),
            "font.size": 13,
            "axes.titlesize": 19,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_common_metric(
    common: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    cohort_rows: int,
) -> None:
    fig, ax = plt.subplots()
    for model in MODEL_ORDER:
        rows = common[common["model"] == model].sort_values("observation_window")
        color, marker = MODEL_STYLE[model]
        ax.plot(
            rows["observation_window"],
            rows[metric],
            color=color,
            marker=marker,
            linewidth=2.8,
            markersize=8,
            label=model,
        )
    if metric == "roc_auc":
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1.8, label="Chance")
        ax.set_ylim(0.48, 0.55)
    else:
        ax.set_ylim(0, 0.6)
    ax.set_xticks([5, 10, 20, 50])
    ax.set_xlabel("Observation window (transitions)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=28)
    ax.text(
        0.5,
        1.015,
        f"Same {cohort_rows:,} traces and identical train/test IDs at every window",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        color="#444444",
    )
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, filename)


def plot_operational_common(
    comparison: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots()
    for model in MODEL_ORDER:
        rows = comparison[comparison["model"] == model].sort_values(
            "observation_window"
        )
        color, marker = MODEL_STYLE[model]
        ax.plot(
            rows["observation_window"],
            rows[f"{metric}_operational"],
            color=color,
            marker=marker,
            linestyle="--",
            linewidth=2.3,
        )
        ax.plot(
            rows["observation_window"],
            rows[f"{metric}_common_cohort"],
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=2.8,
        )
    if metric == "roc_auc":
        ax.axhline(0.5, color="#555555", linestyle=":", linewidth=1.8)
        ax.set_ylim(0.48, 0.55)
    else:
        ax.set_ylim(0, 0.6)
    ax.set_xticks([5, 10, 20, 50])
    ax.set_xlabel("Observation window (transitions)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    model_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_STYLE[model][0],
            marker=MODEL_STYLE[model][1],
            linewidth=2.5,
            label=model,
        )
        for model in MODEL_ORDER
    ]
    cohort_handles = [
        Line2D([0], [0], color="#333333", linewidth=2.8, label="Common cohort"),
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=2.3,
            linestyle="--",
            label="Operational cohort",
        ),
    ]
    first = ax.legend(handles=model_handles, frameon=False, loc="upper left")
    ax.add_artist(first)
    ax.legend(handles=cohort_handles, frameon=False, loc="upper right")
    save_figure(fig, filename)


def plot_operational_counts(operational: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    for column, label, color, marker in (
        ("total_traces", "Retained traces", "#0072B2", "o"),
        ("target_traces", "Target traces", "#D55E00", "s"),
        ("success_traces", "Success traces", "#009E73", "^"),
    ):
        ax.plot(
            operational["observation_window"],
            operational[column],
            label=label,
            color=color,
            marker=marker,
            linewidth=2.8,
            markersize=8,
        )
    ax.set_xticks([5, 10, 20, 50])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel("Operational observation window (transitions)")
    ax.set_ylabel("Trace count")
    ax.set_title("Operational cohorts shrink at longer windows")
    ax.legend(frameon=False)
    save_figure(fig, "operational_retained_traces_by_window.png")


def plot_operational_rate(operational: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.plot(
        operational["observation_window"],
        operational["target_rate"],
        color="#CC79A7",
        marker="o",
        linewidth=3,
        markersize=9,
    )
    ax.set_xticks([5, 10, 20, 50])
    ax.set_ylim(0, 0.4)
    ax.set_xlabel("Operational observation window (transitions)")
    ax.set_ylabel("Target rate")
    ax.set_title("Operational target rate changes with retention window")
    save_figure(fig, "operational_target_rate_by_window.png")


def candidate_label(row: pd.Series) -> str:
    return f"#{int(row['original_rank'])} · state {int(row['state_id'])}"


def plot_candidate_risk(candidates: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    raising = candidates["raises_probability_from_state"].astype(bool)
    ax.scatter(
        candidates.loc[raising, "exact_candidate_reachability"],
        candidates.loc[raising, "target_probability_from_candidate"],
        s=72,
        color="#009E73",
        alpha=0.88,
        label="Raises target probability",
    )
    ax.scatter(
        candidates.loc[~raising, "exact_candidate_reachability"],
        candidates.loc[~raising, "target_probability_from_candidate"],
        s=78,
        marker="X",
        color="#D55E00",
        label="Does not raise target probability",
    )
    baseline = float(candidates["baseline_target_probability"].iloc[0])
    ax.axhline(
        baseline,
        color="#333333",
        linestyle="--",
        linewidth=2,
        label=f"Initial baseline ({baseline:.3f})",
    )
    annotate = candidates[
        candidates["original_rank"].le(5) | ~raising
    ].sort_values("original_rank")
    label_positions = {
        1: (0.065, 0.503),
        2: (0.065, 0.495),
        3: (0.065, 0.487),
        4: (0.065, 0.469),
        5: (0.055, 0.398),
        15: (0.205, 0.306),
        16: (0.020, 0.348),
        20: (0.665, 0.321),
    }
    for _, row in annotate.iterrows():
        rank = int(row["original_rank"])
        ax.annotate(
            candidate_label(row),
            (
                row["exact_candidate_reachability"],
                row["target_probability_from_candidate"],
            ),
            xytext=label_positions[rank],
            textcoords="data",
            fontsize=9.5,
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.7},
        )
    ax.set_xlim(0, 0.82)
    ax.set_ylim(0.28, 0.515)
    ax.set_xlabel(r"Exact candidate reachability $P_{\mathrm{initial}}(F\ candidate)$")
    ax.set_ylabel(r"Target risk from candidate $P_{\mathrm{candidate}}(F\ target)$")
    ax.set_title("Candidate reachability versus exact future target risk")
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, "exact_candidate_reachability_vs_target_risk.png")


def plot_candidate_bars(
    candidates: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    filename: str,
    colors: list[str] | str,
    subtitle: str | None = None,
) -> None:
    ordered = candidates.sort_values(column, ascending=True)
    if isinstance(colors, list):
        color_map = dict(zip(candidates.index, colors))
        bar_colors = [color_map[index] for index in ordered.index]
    else:
        bar_colors = colors
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    labels = [candidate_label(row) for _, row in ordered.iterrows()]
    ax.barh(labels, ordered[column], color=bar_colors)
    ax.axvline(0, color="#333333", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Candidate")
    ax.set_title(title, pad=28 if subtitle else 12)
    if subtitle:
        ax.text(
            0.5,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            fontsize=11.5,
            color="#444444",
        )
    save_figure(fig, filename)


def plot_support_comparison(candidates: pd.DataFrame, metadata: dict) -> None:
    fig, ax = plt.subplots()
    ax.scatter(
        candidates["exact_candidate_reachability"],
        candidates["empirical_support_fraction"],
        color="#0072B2",
        s=72,
        alpha=0.88,
    )
    upper = max(
        candidates["exact_candidate_reachability"].max(),
        candidates["empirical_support_fraction"].max(),
    )
    upper = min(1.0, float(upper) * 1.08)
    ax.plot([0, upper], [0, upper], color="#555555", linestyle="--", linewidth=1.8)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Exact unbounded, unconditional reachability")
    ax.set_ylabel("Empirical bounded, survival-conditioned support")
    ax.set_title("Empirical k20 support and exact reachability")
    note = (
        "Not the same event: empirical support observes the first "
        f"{metadata['observation_window_transitions']} transitions among "
        f"{metadata['empirical_population_rows']:,} traces with "
        f"{metadata['empirical_population_condition']}; exact reachability is "
        "unbounded and unconditional."
    )
    ax.text(
        0.03,
        0.97,
        textwrap.fill(note, 72),
        transform=ax.transAxes,
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.9},
    )
    save_figure(fig, "empirical_support_vs_exact_reachability.png")


def best_window(common: pd.DataFrame, model: str, metric: str) -> tuple[int, float]:
    rows = common[common["model"] == model]
    row = rows.loc[rows[metric].idxmax()]
    return int(row["observation_window"]), float(row[metric])


def write_report(
    common: pd.DataFrame,
    comparison: pd.DataFrame,
    candidates: pd.DataFrame,
    manifest: dict,
    metadata: dict,
    operational: pd.DataFrame,
) -> None:
    cohort_rows = int(manifest["retained_row_count"])
    raising_count = int(candidates["raises_probability_from_state"].astype(bool).sum())
    nonraising_count = len(candidates) - raising_count
    max_gap_row = candidates.loc[
        (
            candidates["empirical_support_fraction"]
            - candidates["exact_candidate_reachability"]
        )
        .abs()
        .idxmax()
    ]
    max_reach = candidates.loc[candidates["exact_candidate_reachability"].idxmax()]
    max_risk = candidates.loc[candidates["target_probability_from_candidate"].idxmax()]
    max_increase = candidates.loc[
        candidates["probability_difference_from_baseline"].idxmax()
    ]
    max_coverage = candidates.loc[candidates["risk_weighted_coverage"].idxmax()]
    lr_f1 = best_window(common, "Logistic Regression", "f1")
    lr_auc = best_window(common, "Logistic Regression", "roc_auc")
    dt_f1 = best_window(common, "Decision Tree", "f1")
    rf_f1 = best_window(common, "Random Forest", "f1")
    rf_auc = best_window(common, "Random Forest", "roc_auc")
    k50 = comparison[comparison["observation_window"] == 50]
    k50_max_difference = max(
        (k50["f1_common_cohort"] - k50["f1_operational"]).abs().max(),
        (
            k50["roc_auc_common_cohort"] - k50["roc_auc_operational"]
        ).abs().max(),
    )
    min_operational = operational.iloc[-1]
    max_operational = operational.iloc[0]

    sections = [
        (
            "common_cohort_f1_by_window.png",
            [COMMON_METRICS_PATH, COMMON_MANIFEST_PATH],
            "How does F1 change as more prefix transitions are observed on one fixed cohort?",
            (
                f"Logistic Regression peaks at k={lr_f1[0]} (F1={lr_f1[1]:.3f}); "
                f"Decision Tree peaks at k={dt_f1[0]} (F1={dt_f1[1]:.3f}); "
                f"Random Forest peaks at k={rf_f1[0]} (F1={rf_f1[1]:.3f})."
            ),
            (
                "The common cohort fixes composition and membership, but finite-sample "
                "variation and model-fitting choices still affect the curves."
            ),
            (
                f"Across the same {cohort_rows:,} traces and identical split IDs, longer "
                "prefixes do not improve F1 monotonically."
            ),
        ),
        (
            "common_cohort_roc_auc_by_window.png",
            [COMMON_METRICS_PATH, COMMON_MANIFEST_PATH],
            "Does ranking discrimination improve with a longer fixed-cohort prefix?",
            (
                f"Logistic Regression peaks at k={lr_auc[0]} "
                f"(ROC-AUC={lr_auc[1]:.3f}); Random Forest peaks at k={rf_auc[0]} "
                f"(ROC-AUC={rf_auc[1]:.3f}), while every score remains near 0.5."
            ),
            "Near-chance ROC-AUC indicates weak discrimination, not proof of no signal.",
            (
                "ROC-AUC is close to chance at every window; Logistic Regression and "
                "Random Forest are best at k=50."
            ),
        ),
        (
            "operational_vs_common_f1.png",
            [COMPARISON_PATH],
            "How much of the operational F1 pattern changes after fixing cohort membership?",
            "Operational and common-cohort F1 differ at k=5, 10, and 20, but coincide at k=50.",
            (
                "Cohort and split changes are not the only possible source of differences; "
                "training stochasticity and feature availability also matter."
            ),
            "The cohort-controlled comparison exposes a clearly non-monotonic F1 response.",
        ),
        (
            "operational_vs_common_roc_auc.png",
            [COMPARISON_PATH],
            "How does cohort control change the observed ROC-AUC pattern?",
            "All operational and common-cohort ROC-AUC values remain close to the 0.5 chance line.",
            "Small ROC-AUC differences should not be over-interpreted without uncertainty estimates.",
            "Cohort control changes details, but does not turn the visited-state models into strong discriminators.",
        ),
        (
            "operational_retained_traces_by_window.png",
            list(OPERATIONAL_PATHS.values()),
            "How does the operational survival filter change the analysed population?",
            (
                f"Retained traces fall from {int(max_operational['total_traces']):,} "
                f"at k=5 to {int(min_operational['total_traces']):,} at k=50; "
                f"target traces fall from {int(max_operational['target_traces']):,} "
                f"to {int(min_operational['target_traces']):,}, while success traces "
                f"remain {int(min_operational['success_traces']):,}."
            ),
            "Counts describe retained datasets and do not measure predictive performance.",
            "Longer operational windows selectively remove early target-ending traces.",
        ),
        (
            "operational_target_rate_by_window.png",
            list(OPERATIONAL_PATHS.values()),
            "How does operational retention alter class balance?",
            (
                f"The target rate declines from {max_operational['target_rate']:.3f} "
                f"at k=5 to {min_operational['target_rate']:.3f} at k=50."
            ),
            "The rate change is induced by cohort retention and is not a model effect.",
            "Operational comparisons mix added prefix information with a changing class balance.",
        ),
        (
            "exact_candidate_reachability_vs_target_risk.png",
            [CANDIDATE_PATH, CANDIDATE_METADATA_PATH],
            "Which candidates combine frequent exact reachability with high future target risk?",
            (
                f"State {int(max_reach['state_id'])} is most reachable "
                f"({max_reach['exact_candidate_reachability']:.3f}); state "
                f"{int(max_risk['state_id'])} has the highest target risk "
                f"({max_risk['target_probability_from_candidate']:.3f}). "
                f"{raising_count} candidates lie above and {nonraising_count} below the baseline."
            ),
            (
                "Quadrants are descriptive: upper-right means comparatively reachable and "
                "high-risk, upper-left rare and high-risk, lower-right reachable and lower-risk, "
                "and lower-left rare and lower-risk. No quadrant establishes causality."
            ),
            "Reachability and future risk are distinct axes; a common state need not raise target risk.",
        ),
        (
            "exact_probability_increase_by_candidate.png",
            [CANDIDATE_PATH],
            "Which candidate states raise exact future target probability above the initial baseline?",
            (
                f"{raising_count} of {len(candidates)} candidates raise probability; "
                f"the largest increase is {max_increase['probability_difference_from_baseline']:.3f} "
                f"at state {int(max_increase['state_id'])}."
            ),
            "State-based probability raising relative to baseline is not by itself causal.",
            f"The exact comparison identifies {raising_count} raising and {nonraising_count} non-raising candidates.",
        ),
        (
            "empirical_support_vs_exact_reachability.png",
            [CANDIDATE_PATH, CANDIDATE_METADATA_PATH],
            "How do bounded k20 empirical support and unbounded exact reachability differ?",
            (
                f"State {int(max_gap_row['state_id'])} has the largest absolute descriptive "
                f"gap ({abs(max_gap_row['empirical_support_fraction'] - max_gap_row['exact_candidate_reachability']):.3f})."
            ),
            (
                f"Empirical k20 support uses {metadata['empirical_population_rows']:,} retained "
                f"traces with {metadata['empirical_population_condition']} and observes only "
                f"the {metadata['empirical_observation_horizon']}; exact reachability is "
                "unbounded and unconditional. They are different events, so the gap is not "
                "pure estimation error."
            ),
            "Distance from the identity line mixes horizon, conditioning, sampling, and model-data effects.",
        ),
        (
            "exact_candidate_reachability_by_candidate.png",
            [CANDIDATE_PATH],
            "How reachable is each selected candidate from the model's initial state?",
            (
                f"Exact reachability ranges from {candidates['exact_candidate_reachability'].min():.3f} "
                f"to {candidates['exact_candidate_reachability'].max():.3f}; state "
                f"{int(max_reach['state_id'])} is highest."
            ),
            "Values apply to this model/build and are not empirical visitation frequencies.",
            "The selected candidates span more than an order of magnitude in exact reachability.",
        ),
        (
            "risk_weighted_coverage_by_candidate.png",
            [CANDIDATE_PATH, CANDIDATE_METADATA_PATH],
            "Which candidates score highest on the reachability-times-risk heuristic?",
            (
                f"State {int(max_coverage['state_id'])} has the largest "
                f"risk-weighted coverage ({max_coverage['risk_weighted_coverage']:.3f})."
            ),
            (
                "risk_weighted_coverage is only the descriptive product of exact reachability "
                "and target probability from the state; it is not causal, path-conditioned, "
                "or a formal probability-raising measure."
            ),
            "Use the heuristic for descriptive prioritisation only, not causal attribution.",
        ),
    ]

    lines = [
        "# Current BRP experiment plot summary",
        "",
        (
            f"All common-cohort windows use the same **{cohort_rows:,} traces**, constant "
            "class balance, and identical train/test membership. Logistic Regression has "
            f"its best common-cohort F1 at k={lr_f1[0]} and ROC-AUC at k={lr_auc[0]}; "
            f"Decision Tree has its best F1 at k={dt_f1[0]}; Random Forest has its best "
            f"F1 at k={rf_f1[0]} and best ROC-AUC at k={rf_auc[0]}. The patterns are "
            "non-monotonic and ROC-AUC remains near 0.5."
        ),
        "",
        (
            f"At k=50 the common and operational populations are identical; the maximum "
            f"reported F1/ROC-AUC difference is {k50_max_difference:.3g}. Models use "
            "visited-state presence and do not preserve full sequence order."
        ),
        "",
    ]
    for filename, sources, question, observation, limitation, sentence in sections:
        lines.extend(
            [
                f"## `{filename}`",
                "",
                "**Sources:** "
                + ", ".join(f"`{relative(source)}`" for source in sources),
                "",
                f"**Question:** {question}",
                "",
                f"**Main numerical observation:** {observation}",
                "",
                f"**Limitation:** {limitation}",
                "",
                f"**Presentation sentence:** {sentence}",
                "",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    common, comparison, candidates, manifest, metadata, operational = load_inputs()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    configure_style()
    generated: list[str] = []
    skipped: list[str] = []

    def skip(filename: str, paths: list[Path]) -> None:
        missing = [relative(path) for path in paths if not path.is_file()]
        message = f"Skipped {filename}; missing: {', '.join(missing)}"
        skipped.append(message)
        print(message)

    if common is not None and manifest is not None:
        cohort_rows = int(manifest["retained_row_count"])
        plot_common_metric(
            common,
            "f1",
            "Common-cohort F1 by observation window",
            "F1 score",
            "common_cohort_f1_by_window.png",
            cohort_rows,
        )
        plot_common_metric(
            common,
            "roc_auc",
            "Common-cohort ROC-AUC by observation window",
            "ROC-AUC",
            "common_cohort_roc_auc_by_window.png",
            cohort_rows,
        )
        generated.extend(
            [
                "common_cohort_f1_by_window.png",
                "common_cohort_roc_auc_by_window.png",
            ]
        )
    else:
        for filename in (
            "common_cohort_f1_by_window.png",
            "common_cohort_roc_auc_by_window.png",
        ):
            skip(filename, [COMMON_METRICS_PATH, COMMON_MANIFEST_PATH])

    if comparison is not None:
        plot_operational_common(
            comparison,
            "f1",
            "Operational versus common-cohort F1",
            "F1 score",
            "operational_vs_common_f1.png",
        )
        plot_operational_common(
            comparison,
            "roc_auc",
            "Operational versus common-cohort ROC-AUC",
            "ROC-AUC",
            "operational_vs_common_roc_auc.png",
        )
        generated.extend(
            [
                "operational_vs_common_f1.png",
                "operational_vs_common_roc_auc.png",
            ]
        )
    else:
        for filename in (
            "operational_vs_common_f1.png",
            "operational_vs_common_roc_auc.png",
        ):
            skip(filename, [COMPARISON_PATH])

    if operational is not None:
        plot_operational_counts(operational)
        plot_operational_rate(operational)
        generated.extend(
            [
                "operational_retained_traces_by_window.png",
                "operational_target_rate_by_window.png",
            ]
        )
    else:
        for filename in (
            "operational_retained_traces_by_window.png",
            "operational_target_rate_by_window.png",
        ):
            skip(filename, list(OPERATIONAL_PATHS.values()))

    if candidates is not None:
        plot_candidate_risk(candidates)
        raising = candidates["raises_probability_from_state"].astype(bool)
        increase_colors = ["#009E73" if value else "#D55E00" for value in raising]
        plot_candidate_bars(
            candidates,
            "probability_difference_from_baseline",
            "Exact target-probability increase by candidate",
            "Probability difference from initial-state baseline",
            "exact_probability_increase_by_candidate.png",
            increase_colors,
            (
                f"{int(raising.sum())} candidates raise exact future target probability; "
                f"{int((~raising).sum())} do not"
            ),
        )
        plot_candidate_bars(
            candidates,
            "exact_candidate_reachability",
            "Exact candidate reachability from the initial state",
            r"$P_{\mathrm{initial}}(F\ candidate)$",
            "exact_candidate_reachability_by_candidate.png",
            "#0072B2",
        )
        plot_candidate_bars(
            candidates,
            "risk_weighted_coverage",
            "Risk-weighted coverage by candidate",
            "Exact reachability × target probability from candidate",
            "risk_weighted_coverage_by_candidate.png",
            "#CC79A7",
            "Descriptive heuristic only — not a causal measure",
        )
        generated.extend(
            [
                "exact_candidate_reachability_vs_target_risk.png",
                "exact_probability_increase_by_candidate.png",
                "exact_candidate_reachability_by_candidate.png",
                "risk_weighted_coverage_by_candidate.png",
            ]
        )
        if metadata is not None:
            plot_support_comparison(candidates, metadata)
            generated.append("empirical_support_vs_exact_reachability.png")
        else:
            skip(
                "empirical_support_vs_exact_reachability.png",
                [CANDIDATE_PATH, CANDIDATE_METADATA_PATH],
            )
    else:
        for filename in (
            "exact_candidate_reachability_vs_target_risk.png",
            "exact_probability_increase_by_candidate.png",
            "empirical_support_vs_exact_reachability.png",
            "exact_candidate_reachability_by_candidate.png",
            "risk_weighted_coverage_by_candidate.png",
        ):
            paths = [CANDIDATE_PATH]
            if filename == "empirical_support_vs_exact_reachability.png":
                paths.append(CANDIDATE_METADATA_PATH)
            skip(filename, paths)

    if all(
        item is not None
        for item in (
            common,
            comparison,
            candidates,
            manifest,
            metadata,
            operational,
        )
    ):
        write_report(
            common,
            comparison,
            candidates,
            manifest,
            metadata,
            operational,
        )
    else:
        REPORT_PATH.write_text(
            "\n".join(
                [
                    "# Current BRP experiment plot summary",
                    "",
                    "The run was incomplete because one or more source artifacts were missing.",
                    "",
                    "## Generated plots",
                    "",
                    *[f"- `{filename}`" for filename in generated],
                    "",
                    "## Skipped plots",
                    "",
                    *[f"- {message}" for message in skipped],
                    "",
                ]
            ),
            encoding="utf-8",
        )

    print(f"Generated {len(generated)} plots in {relative(PLOT_DIR)}")
    print(f"Wrote {relative(REPORT_PATH)}")


if __name__ == "__main__":
    main()
