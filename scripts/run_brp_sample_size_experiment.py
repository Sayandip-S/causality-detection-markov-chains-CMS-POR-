from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import train_test_split

from scripts.extract_brp_candidate_states import rank_candidate_states
from scripts.generate_brp_common_cohort_datasets import ordered_values_sha256
from scripts.map_brp_candidate_states import get_prism_variables, python_scalar
from scripts.run_brp_common_cohort_baselines import create_shared_trace_split
from scripts.verify_brp_candidate_states import (
    candidate_reachabilities,
    unique_initial_state,
)
from src.ml.train_brp_baselines import (
    calculate_binary_metrics,
    capture_source_provenance,
    create_models,
    dependency_versions,
    get_positive_probabilities,
    repository_relative_path,
    select_feature_columns,
    sha256_file,
)
from src.storm.model_utils import PROJECT_ROOT, load_prism_model


DEFAULT_CONFIG = PROJECT_ROOT / "experiments/brp_k20_sample_size_stability.json"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results/systematic/brp_stress_error/sample_size"
)
EXPECTED_ROWS = 9177
MODEL_SLUG_TO_NAME = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}
OUTPUT_FILENAMES = (
    "prediction_per_run.csv",
    "prediction_aggregated.csv",
    "prediction_metadata.json",
    "candidate_rankings_per_run.csv",
    "candidate_stability_per_run.csv",
    "candidate_stability_aggregated.csv",
    "exact_candidate_cache.csv",
    "exact_candidate_quality_per_run.csv",
    "exact_candidate_quality_aggregated.csv",
    "reliability_assessment.csv",
    "ranking_method_comparison.csv",
    "random_baseline_distribution.csv",
)


def load_config(path: Path) -> dict[str, Any]:
    """Load a repository-portable experiment configuration."""

    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "experiment_name",
        "dataset",
        "training_sample_sizes",
        "sampling_seeds",
        "full_training_size",
        "split_seed",
        "test_size",
        "top_k",
        "candidate_support_rule",
        "models",
        "reliability_thresholds",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Experiment config is missing fields: {missing}")
    for field in (
        "dataset",
        "common_cohort_manifest",
        "common_cohort_summary",
        "existing_exact_reachability",
        "prism_model",
        "property_file",
    ):
        value = config.get(field)
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError(f"Config path {field!r} must be repository-relative.")
    return config


def adaptive_minimum_support(training_sample_size: int, config: dict[str, Any]) -> int:
    """Return the documented sample-size-scaled candidate support count."""

    rule = config["candidate_support_rule"]
    return max(
        int(rule["minimum_count_floor"]),
        math.ceil(float(rule["training_fraction"]) * training_sample_size),
    )


def validate_dataset(
    dataset: pd.DataFrame,
    expected_rows: int = EXPECTED_ROWS,
) -> list[str]:
    """Validate the common-cohort dataset and return leakage-safe features."""

    if len(dataset) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} common-cohort rows, found {len(dataset):,}."
        )
    if "trace_id" not in dataset or dataset["trace_id"].isna().any():
        raise ValueError("Dataset must contain non-missing trace_id values.")
    if dataset["trace_id"].duplicated().any():
        raise ValueError("Dataset trace_id values must be unique.")
    if "target" not in dataset or dataset["target"].isna().any():
        raise ValueError("Dataset must contain a non-missing target column.")
    if set(dataset["target"].unique()) != {0, 1}:
        raise ValueError("Dataset target must contain both binary classes 0 and 1.")
    features = select_feature_columns(dataset, "visited_states_only")
    invalid = [
        column
        for column in features
        if not set(dataset[column].dropna().unique()).issubset({0, 1})
        or dataset[column].isna().any()
    ]
    if invalid:
        raise ValueError(f"Visited-state features must be binary: {invalid}")
    return features


def reconstruct_fixed_split(
    dataset: pd.DataFrame,
    summary: dict[str, Any],
    *,
    test_size: float,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Recreate and validate the prior fixed common-cohort split."""

    train_ids, test_ids = create_shared_trace_split(
        dataset,
        test_size=test_size,
        random_seed=split_seed,
    )
    train_hash = ordered_values_sha256(train_ids)
    test_hash = ordered_values_sha256(test_ids)
    expected_hashes = summary["split_trace_id_hashes"]
    if train_hash != expected_hashes["train"]:
        raise ValueError("Reconstructed training trace hash differs from provenance.")
    if test_hash != expected_hashes["test"]:
        raise ValueError("Reconstructed test trace hash differs from provenance.")
    indexed = dataset.set_index("trace_id", drop=False)
    train_pool = indexed.loc[train_ids].reset_index(drop=True)
    test_set = indexed.loc[test_ids].reset_index(drop=True)
    if set(train_pool["trace_id"]).intersection(test_set["trace_id"]):
        raise RuntimeError("Train/test trace-ID overlap detected.")
    split_config = summary["split_configuration"]
    if len(train_pool) != split_config["train_rows"] or len(test_set) != split_config[
        "test_rows"
    ]:
        raise ValueError("Reconstructed split row counts differ from provenance.")
    return train_pool, test_set, train_hash, test_hash


def stratified_training_sample(
    train_pool: pd.DataFrame,
    sample_size: int,
    sampling_seed: int | None,
    full_training_size: int,
) -> pd.DataFrame:
    """Select a deterministic stratified subset of the fixed training pool."""

    if sample_size <= 0 or sample_size > len(train_pool):
        raise ValueError("Training sample size must be within the training pool.")
    if sample_size == full_training_size:
        if len(train_pool) != full_training_size:
            raise ValueError("Configured full training size differs from split size.")
        return train_pool.copy()
    if sampling_seed is None:
        raise ValueError("Reduced training samples require a sampling seed.")
    sampled, _ = train_test_split(
        train_pool,
        train_size=sample_size,
        random_state=sampling_seed,
        stratify=train_pool["target"],
    )
    return sampled.reset_index(drop=True)


def experiment_conditions(
    sample_sizes: Iterable[int],
    sampling_seeds: Iterable[int],
    full_training_size: int,
) -> list[tuple[int, int | None]]:
    """Return reduced repeats plus one deterministic full-pool condition."""

    conditions: list[tuple[int, int | None]] = []
    for size in sample_sizes:
        if size == full_training_size:
            conditions.append((size, None))
        else:
            conditions.extend((size, seed) for seed in sampling_seeds)
    return conditions


def ranking_stability(
    ranking: pd.DataFrame,
    reference: pd.DataFrame,
    top_k: int = 20,
) -> dict[str, Any]:
    """Calculate set and shared-state rank agreement with a reference."""

    result: dict[str, Any] = {}
    for k in (5, 10, 20):
        effective_k = min(k, top_k)
        current = set(ranking.head(effective_k)["state_id"].astype(int))
        full = set(reference.head(effective_k)["state_id"].astype(int))
        overlap = len(current & full)
        union = len(current | full)
        result[f"top{k}_overlap_count"] = overlap
        result[f"top{k}_overlap_fraction"] = overlap / effective_k
        result[f"top{k}_jaccard_similarity"] = overlap / union if union else float("nan")

    ranks = ranking.head(top_k).set_index("state_id")["rank"]
    reference_ranks = reference.head(top_k).set_index("state_id")["rank"]
    shared = sorted(set(ranks.index).intersection(reference_ranks.index))
    result["shared_candidate_count"] = len(shared)
    if len(shared) < 2:
        result.update(
            {
                "spearman_rank_correlation": float("nan"),
                "kendall_rank_correlation": float("nan"),
                "mean_absolute_rank_displacement": (
                    float(abs(ranks[shared[0]] - reference_ranks[shared[0]]))
                    if shared
                    else float("nan")
                ),
                "rank_correlation_explanation": (
                    "undefined: fewer than two shared top-ranked states"
                ),
            }
        )
    else:
        current_values = [float(ranks[state]) for state in shared]
        reference_values = [float(reference_ranks[state]) for state in shared]
        result.update(
            {
                "spearman_rank_correlation": float(
                    spearmanr(current_values, reference_values).statistic
                ),
                "kendall_rank_correlation": float(
                    kendalltau(current_values, reference_values).statistic
                ),
                "mean_absolute_rank_displacement": float(
                    np.mean(np.abs(np.asarray(current_values) - reference_values))
                ),
                "rank_correlation_explanation": "defined on shared top-20 states",
            }
        )
    for state_id in reference.head(5)["state_id"].astype(int):
        result[f"full_top5_state_{state_id}_present"] = state_id in set(ranks.index)
    return result


def aggregate_numeric(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    """Aggregate metrics with repeat count, spread, range, and 95% CI."""

    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["number_of_repeats"] = len(group)
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            count = len(values)
            mean = float(values.mean()) if count else float("nan")
            standard_deviation = float(values.std(ddof=1)) if count > 1 else 0.0
            margin = 1.96 * standard_deviation / math.sqrt(count) if count else float("nan")
            row.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_std": standard_deviation,
                    f"{metric}_min": float(values.min()) if count else float("nan"),
                    f"{metric}_max": float(values.max()) if count else float("nan"),
                    f"{metric}_ci95_low": mean - margin,
                    f"{metric}_ci95_high": mean + margin,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def exact_cache_from_existing(path: Path) -> pd.DataFrame:
    """Load exact rows already verified by the prior full-data experiment."""

    existing = pd.read_csv(path)
    required = {
        "state_id",
        "exact_candidate_reachability",
        "baseline_target_probability",
        "target_probability_from_candidate",
        "probability_difference_from_baseline",
        "probability_ratio_to_baseline",
        "raises_probability_from_state",
    }
    missing = sorted(required - set(existing))
    if missing:
        raise ValueError(f"Existing exact reachability is missing columns: {missing}")
    cache = existing.copy()
    cache["exact_value_source"] = "existing_exact_reachability"
    return cache


def verify_missing_exact_states(
    cache: pd.DataFrame,
    state_ids: Iterable[int],
    model_path: Path,
    property_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify uncached states with one model build and one target result vector."""

    requested = sorted({int(value) for value in state_ids})
    cached = set(cache["state_id"].astype(int))
    missing = [state_id for state_id in requested if state_id not in cached]
    if not missing:
        return cache[cache["state_id"].astype(int).isin(requested)].copy(), {
            "missing_state_count_verified": 0,
            "model_built": False,
        }

    import stormpy

    program, properties, model = load_prism_model(model_path, property_path)
    if len(properties) != 1:
        raise ValueError("Expected exactly one target property.")
    initial_state = unique_initial_state(model)
    target_result = stormpy.model_checking(model, properties[0])
    if not target_result.result_for_all_states:
        raise RuntimeError("Storm target result is not available for all states.")
    invalid = [state_id for state_id in missing if not 0 <= state_id < model.nr_states]
    if invalid:
        raise ValueError(f"Candidate state IDs are outside the Storm model: {invalid}")
    reachabilities = candidate_reachabilities(
        model, initial_state, pd.Series(missing, dtype=int)
    )
    baseline = float(target_result.at(initial_state))
    variables = get_prism_variables(program)
    rows: list[dict[str, Any]] = []
    for state_id, reachability in zip(missing, reachabilities):
        target_probability = float(target_result.at(state_id))
        row: dict[str, Any] = {
            "state_id": state_id,
            "exact_candidate_reachability": reachability,
            "baseline_target_probability": baseline,
            "target_probability_from_candidate": target_probability,
            "probability_difference_from_baseline": target_probability - baseline,
            "probability_ratio_to_baseline": (
                target_probability / baseline if baseline else float("nan")
            ),
            "raises_probability_from_state": target_probability > baseline,
            "exact_value_source": "sample_size_storm_verification",
        }
        for variable in variables:
            row[variable.name] = python_scalar(
                model.state_valuations.get_value(
                    state_id, variable.expression_variable
                )
            )
        row["labels"] = "|".join(sorted(model.states[state_id].labels))
        rows.append(row)
    combined = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True, sort=False)
    combined = combined.drop_duplicates("state_id", keep="first")
    selected = combined[combined["state_id"].astype(int).isin(requested)].copy()
    return selected.sort_values("state_id").reset_index(drop=True), {
        "missing_state_count_verified": len(missing),
        "model_built": True,
        "storm_model_state_count": int(model.nr_states),
        "storm_model_transition_count": int(model.nr_transitions),
        "initial_state_id": initial_state,
        "baseline_target_probability": baseline,
        "stormpy_version": getattr(stormpy, "__version__", None),
    }


def exact_quality(ranking: pd.DataFrame, exact_cache: pd.DataFrame) -> dict[str, Any]:
    """Summarize exact probability-raising quality for one ranking."""

    joined = ranking.merge(exact_cache, on="state_id", how="left", validate="one_to_one")
    if joined["target_probability_from_candidate"].isna().any():
        raise RuntimeError("Exact cache does not cover every ranked candidate.")
    empirical = joined["empirical_probability_difference"]
    exact = joined["probability_difference_from_baseline"]
    raises = joined["raises_probability_from_state"].astype(bool)
    return {
        "candidate_count": len(joined),
        "exact_probability_raising_count": int(raises.sum()),
        "exact_probability_raising_fraction": float(raises.mean()),
        "mean_exact_probability_difference": float(exact.mean()),
        "median_exact_probability_difference": float(exact.median()),
        "maximum_exact_probability_difference": float(exact.max()),
        "mean_exact_candidate_reachability": float(
            joined["exact_candidate_reachability"].mean()
        ),
        "positive_empirical_negative_exact_count": int(
            ((empirical > 0) & (exact < 0)).sum()
        ),
        "negative_empirical_positive_exact_count": int(
            ((empirical < 0) & (exact > 0)).sum()
        ),
    }


def ranking_method_rows(
    all_candidates: pd.DataFrame,
    exact_cache: pd.DataFrame,
    *,
    sample_size: int,
    sampling_seed: int | None,
    top_k: int,
) -> list[dict[str, Any]]:
    """Evaluate combined and four deterministic single-component rankings."""

    methods = {
        "combined": ("combined_ranking_score", False),
        "empirical_difference_only": ("probability_difference", False),
        "frequency_support_only": ("visited_trace_count", False),
        "logistic_coefficient_only": ("logistic_regression_coefficient", False),
        "random_forest_importance_only": ("random_forest_feature_importance", False),
    }
    rows = []
    for method, (column, ascending) in methods.items():
        selected = all_candidates.sort_values(
            [column, "visited_trace_count", "state_id"],
            ascending=[ascending, False, True],
        ).head(top_k)
        joined = selected.merge(exact_cache, on="state_id", validate="one_to_one")
        rows.append(
            {
                "training_sample_size": sample_size,
                "sampling_seed": sampling_seed,
                "ranking_method": method,
                "candidate_count": len(joined),
                "exact_probability_raising_count": int(
                    joined["raises_probability_from_state"].astype(bool).sum()
                ),
                "mean_exact_probability_difference": float(
                    joined["probability_difference_from_baseline"].mean()
                ),
                "mean_exact_candidate_reachability": float(
                    joined["exact_candidate_reachability"].mean()
                ),
            }
        )
    return rows


def random_baseline_rows(
    all_candidates: pd.DataFrame,
    exact_cache: pd.DataFrame,
    *,
    sample_size: int,
    sampling_seed: int | None,
    top_k: int,
    repeats: int,
) -> list[dict[str, Any]]:
    """Evaluate deterministic random eligible-state sets."""

    if repeats <= 0:
        return []
    eligible = sorted(set(all_candidates["state_id"].astype(int)))
    if len(eligible) < top_k:
        raise ValueError("Too few eligible states for a random top-k baseline.")
    base_seed = (sampling_seed if sampling_seed is not None else 0) + sample_size * 1009
    rows = []
    for repeat in range(repeats):
        random_seed = base_seed + repeat
        rng = np.random.default_rng(random_seed)
        selected = rng.choice(eligible, size=top_k, replace=False)
        joined = exact_cache[exact_cache["state_id"].astype(int).isin(selected)]
        rows.append(
            {
                "training_sample_size": sample_size,
                "sampling_seed": sampling_seed,
                "random_repeat": repeat + 1,
                "random_seed": random_seed,
                "exact_probability_raising_count": int(
                    joined["raises_probability_from_state"].astype(bool).sum()
                ),
                "mean_exact_probability_difference": float(
                    joined["probability_difference_from_baseline"].mean()
                ),
                "mean_exact_candidate_reachability": float(
                    joined["exact_candidate_reachability"].mean()
                ),
            }
        )
    return rows


def ensure_outputs_absent(output_root: Path, *, allow_cache: bool = False) -> None:
    """Refuse to overwrite prior experiment artifacts."""

    existing = []
    for filename in OUTPUT_FILENAMES:
        if allow_cache and filename == "exact_candidate_cache.csv":
            continue
        path = output_root / filename
        if path.exists():
            existing.append(path)
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {formatted}")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV only when the destination does not already exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    frame.to_csv(path, index=False)


def write_json(document: dict[str, Any], path: Path) -> None:
    """Write strict JSON only when the destination does not already exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(document, output, indent=2, allow_nan=False)
        output.write("\n")


def reliability_assessment(
    prediction_aggregated: pd.DataFrame,
    stability_aggregated: pd.DataFrame,
    quality_aggregated: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply analyst-selected operational reliability criteria."""

    thresholds = config["reliability_thresholds"]
    full_size = config["full_training_size"]
    full_prediction = prediction_aggregated[
        prediction_aggregated["training_sample_size"] == full_size
    ].set_index("model")
    stability = stability_aggregated.set_index("training_sample_size")
    quality = quality_aggregated.set_index("training_sample_size")
    rows = []
    for _, prediction in prediction_aggregated.iterrows():
        size = int(prediction["training_sample_size"])
        model = prediction["model"]
        f1_gap = abs(prediction["f1_mean"] - full_prediction.loc[model, "f1_mean"])
        auc_gap = abs(
            prediction["roc_auc_mean"] - full_prediction.loc[model, "roc_auc_mean"]
        )
        row = {
            "training_sample_size": size,
            "model": model,
            "thresholds_are_analyst_selected": True,
            "f1_gap_from_full": f1_gap,
            "f1_within_0_05_of_full": (
                f1_gap <= thresholds["f1_absolute_gap_from_full_max"]
            ),
            "roc_auc_gap_from_full": auc_gap,
            "roc_auc_within_0_02_of_full": (
                auc_gap <= thresholds["roc_auc_absolute_gap_from_full_max"]
            ),
            "f1_standard_deviation": prediction["f1_std"],
            "f1_standard_deviation_below_0_05": (
                prediction["f1_std"] < thresholds["f1_standard_deviation_max"]
            ),
            "top10_overlap_fraction_mean": stability.loc[
                size, "top10_overlap_fraction_mean"
            ],
            "top10_overlap_at_least_0_70": (
                stability.loc[size, "top10_overlap_fraction_mean"]
                >= thresholds["top10_overlap_fraction_min"]
            ),
            "top20_overlap_fraction_mean": stability.loc[
                size, "top20_overlap_fraction_mean"
            ],
            "top20_overlap_at_least_0_60": (
                stability.loc[size, "top20_overlap_fraction_mean"]
                >= thresholds["top20_overlap_fraction_min"]
            ),
            "exact_probability_raising_count_mean": quality.loc[
                size, "exact_probability_raising_count_mean"
            ],
            "at_least_15_exact_probability_raising": (
                quality.loc[size, "exact_probability_raising_count_mean"]
                >= thresholds["exact_probability_raising_count_min"]
            ),
            "exact_pass_count_standard_deviation": quality.loc[
                size, "exact_probability_raising_count_std"
            ],
            "exact_pass_count_standard_deviation_below_2": (
                quality.loc[size, "exact_probability_raising_count_std"]
                < thresholds["exact_pass_count_standard_deviation_max"]
            ),
        }
        pass_columns = [
            key
            for key in row
            if key.endswith(("_full", "_0_05", "_0_70", "_0_60", "_raising", "_2"))
            and isinstance(row[key], (bool, np.bool_))
        ]
        row["all_operational_criteria_pass"] = all(row[key] for key in pass_columns)
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(
    config_path: Path,
    output_root: Path,
    *,
    exact_cache_path: Path | None = None,
    sample_sizes: list[int] | None = None,
    sampling_seeds: list[int] | None = None,
    quick: bool = False,
    skip_exact_verification: bool = False,
    skip_baselines: bool = False,
) -> dict[str, Any]:
    """Run the BRP k=20 sample-size and candidate-stability experiment."""

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    config = load_config(config_path)
    sizes = sample_sizes or [int(value) for value in config["training_sample_sizes"]]
    seeds = sampling_seeds or [int(value) for value in config["sampling_seeds"]]
    if quick and sampling_seeds is None:
        seeds = seeds[:3]
    full_size = int(config["full_training_size"])
    if full_size not in sizes:
        raise ValueError("Selected sample sizes must include the full training size.")
    if len(sizes) != len(set(sizes)) or len(seeds) != len(set(seeds)):
        raise ValueError("Sample sizes and sampling seeds must be unique.")
    ensure_outputs_absent(output_root)
    source_provenance = capture_source_provenance(started_at.isoformat())

    resolved = {
        field: (PROJECT_ROOT / config[field]).resolve()
        for field in (
            "dataset",
            "common_cohort_manifest",
            "common_cohort_summary",
            "existing_exact_reachability",
            "prism_model",
            "property_file",
        )
    }
    if exact_cache_path is not None:
        exact_cache_path = exact_cache_path.resolve()
        repository_relative_path(exact_cache_path, "Exact cache path")
        resolved["existing_exact_reachability"] = exact_cache_path
    for field, path in resolved.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required {field} file not found: {path}")
    dataset = pd.read_csv(resolved["dataset"])
    features = validate_dataset(dataset)
    manifest = json.loads(resolved["common_cohort_manifest"].read_text(encoding="utf-8"))
    summary = json.loads(resolved["common_cohort_summary"].read_text(encoding="utf-8"))
    dataset_digest = sha256_file(resolved["dataset"])
    k20_entry = next(
        entry for entry in manifest["windows"] if entry["k"] == config["observation_window"]
    )
    if dataset_digest != k20_entry["output_sha256"]:
        raise ValueError("Dataset SHA-256 differs from common-cohort provenance.")
    train_pool, test_set, train_hash, test_hash = reconstruct_fixed_split(
        dataset,
        summary,
        test_size=float(config["test_size"]),
        split_seed=int(config["split_seed"]),
    )
    x_test = test_set[features]
    y_test = test_set["target"].astype(int)

    prediction_rows: list[dict[str, Any]] = []
    ranking_frames: list[pd.DataFrame] = []
    all_candidate_frames: dict[tuple[int, int | None], pd.DataFrame] = {}
    sampling_records: list[dict[str, Any]] = []
    conditions = experiment_conditions(sizes, seeds, full_size)
    for sample_size, sampling_seed in conditions:
        condition_started = time.perf_counter()
        sampling_started = time.perf_counter()
        sampled = stratified_training_sample(
            train_pool, sample_size, sampling_seed, full_size
        )
        sampling_seconds = time.perf_counter() - sampling_started
        if set(sampled["trace_id"]).intersection(test_set["trace_id"]):
            raise RuntimeError("Sampled training rows overlap the fixed test set.")
        sampled_hash = ordered_values_sha256(sampled["trace_id"])
        preparation_started = time.perf_counter()
        x_train = sampled[features]
        y_train = sampled["target"].astype(int)
        feature_preparation_seconds = time.perf_counter() - preparation_started
        models = create_models(int(config["split_seed"]))
        fitted: dict[str, Any] = {}
        for model_name, model in models.items():
            model_slug = next(
                slug for slug, name in MODEL_SLUG_TO_NAME.items() if name == model_name
            )
            if model_slug not in config["models"]:
                continue
            training_started = time.perf_counter()
            model.fit(x_train, y_train)
            training_seconds = time.perf_counter() - training_started
            fitted[model_slug] = model
            evaluation_started = time.perf_counter()
            predictions = model.predict(x_test)
            probabilities = get_positive_probabilities(model, x_test)
            evaluation_seconds = time.perf_counter() - evaluation_started
            metrics = calculate_binary_metrics(
                y_test,
                predictions,
                probabilities,
                training_row_count=len(sampled),
                positive_rate=float(sampled["target"].mean()),
                number_of_features=len(features),
            )
            prediction_rows.append(
                {
                    "training_sample_size": sample_size,
                    "sampling_seed": sampling_seed,
                    "sampling_seed_applicable": sampling_seed is not None,
                    "model": model_name,
                    "model_random_state": int(config["split_seed"]),
                    "train_rows": len(sampled),
                    "test_rows": len(test_set),
                    "train_target_count": int(y_train.sum()),
                    "train_success_count": int((y_train == 0).sum()),
                    "train_target_rate": float(y_train.mean()),
                    "test_target_count": int(y_test.sum()),
                    "test_success_count": int((y_test == 0).sum()),
                    "test_target_rate": float(y_test.mean()),
                    "feature_count": len(features),
                    "sampled_training_trace_id_sha256": sampled_hash,
                    "fixed_test_trace_id_sha256": test_hash,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tp": metrics["tp"],
                    "sampling_seconds": sampling_seconds,
                    "feature_preparation_seconds": feature_preparation_seconds,
                    "model_training_seconds": training_seconds,
                    "prediction_evaluation_seconds": evaluation_seconds,
                    "total_seconds": (
                        sampling_seconds
                        + feature_preparation_seconds
                        + training_seconds
                        + evaluation_seconds
                    ),
                }
            )
        minimum_support = adaptive_minimum_support(sample_size, config)
        all_candidates = rank_candidate_states(
            sampled,
            features,
            fitted["logistic_regression"],
            fitted["random_forest"],
            minimum_support,
        )
        if len(all_candidates) < int(config["top_k"]):
            raise RuntimeError(
                f"Only {len(all_candidates)} states meet support for sample size "
                f"{sample_size}, seed {sampling_seed}."
            )
        all_candidate_frames[(sample_size, sampling_seed)] = all_candidates
        top = all_candidates.head(int(config["top_k"])).copy()
        top.insert(0, "sampling_seed", sampling_seed)
        top.insert(0, "training_sample_size", sample_size)
        top = top.rename(
            columns={
                "rank": "candidate_rank",
                "visited_trace_count": "visited_count",
                "random_forest_feature_importance": "random_forest_importance",
                "target_probability_when_visited": (
                    "empirical_target_probability_if_visited"
                ),
                "target_probability_when_not_visited": (
                    "empirical_target_probability_if_not_visited"
                ),
                "probability_difference": "empirical_probability_difference",
                "support_reliability_weight": "support_weight",
                "normalized_positive_logistic_coefficient": (
                    "positive_logistic_score"
                ),
            }
        )
        top["minimum_support_used"] = minimum_support
        top["minimum_support_fraction_used"] = minimum_support / sample_size
        required_columns = [
            "training_sample_size",
            "sampling_seed",
            "candidate_rank",
            "state_id",
            "visited_count",
            "support_fraction",
            "minimum_support_used",
            "minimum_support_fraction_used",
            "logistic_regression_coefficient",
            "positive_logistic_score",
            "random_forest_importance",
            "empirical_target_probability_if_visited",
            "empirical_target_probability_if_not_visited",
            "empirical_probability_difference",
            "support_weight",
            "combined_ranking_score",
        ]
        ranking_frames.append(top[required_columns])
        sampling_records.append(
            {
                "training_sample_size": sample_size,
                "sampling_seed": sampling_seed,
                "sampled_training_trace_id_sha256": sampled_hash,
                "target_count": int(y_train.sum()),
                "target_rate": float(y_train.mean()),
                "minimum_support_used": minimum_support,
                "ranking_eligible_state_count": len(all_candidates),
                "condition_seconds": time.perf_counter() - condition_started,
            }
        )

    prediction = pd.DataFrame(prediction_rows)
    expected_prediction_rows = (len(conditions) * len(config["models"]))
    if len(prediction) != expected_prediction_rows:
        raise RuntimeError("Unexpected prediction evaluation count.")
    if prediction.duplicated(
        ["training_sample_size", "sampling_seed", "model"]
    ).any():
        raise RuntimeError("Duplicate prediction experiment keys detected.")
    if prediction["fixed_test_trace_id_sha256"].nunique() != 1:
        raise RuntimeError("Fixed test trace hash changed across evaluations.")
    ranking = pd.concat(ranking_frames, ignore_index=True)
    if not (ranking.groupby(["training_sample_size", "sampling_seed"], dropna=False).size() == int(config["top_k"])).all():
        raise RuntimeError("Every ranking condition must contain exactly top-k rows.")

    reference = ranking[ranking["training_sample_size"] == full_size].copy()
    reference = reference.rename(columns={"candidate_rank": "rank"})
    stability_rows = []
    for (size, seed), group in ranking.groupby(
        ["training_sample_size", "sampling_seed"], dropna=False, sort=True
    ):
        current = group.rename(columns={"candidate_rank": "rank"})
        stability_rows.append(
            {
                "training_sample_size": int(size),
                "sampling_seed": None if pd.isna(seed) else int(seed),
                **ranking_stability(current, reference, int(config["top_k"])),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability_metrics = [
        column
        for column in stability.columns
        if column not in {"training_sample_size", "sampling_seed", "rank_correlation_explanation"}
        and not column.startswith("full_top5_state_")
    ]
    stability_aggregated = aggregate_numeric(
        stability, ["training_sample_size"], stability_metrics
    )

    exact_metadata: dict[str, Any] = {"skipped": skip_exact_verification}
    quality = pd.DataFrame()
    quality_aggregated = pd.DataFrame()
    method_comparison = pd.DataFrame()
    random_distribution = pd.DataFrame(
        columns=[
            "training_sample_size",
            "sampling_seed",
            "random_repeat",
            "random_seed",
            "exact_probability_raising_count",
            "mean_exact_probability_difference",
            "mean_exact_candidate_reachability",
        ]
    )
    exact_cache = pd.DataFrame()
    if not skip_exact_verification:
        exact_cache = exact_cache_from_existing(resolved["existing_exact_reachability"])
        all_eligible_state_ids = {
            int(state_id)
            for candidates in all_candidate_frames.values()
            for state_id in candidates["state_id"]
        }
        exact_cache, exact_metadata = verify_missing_exact_states(
            exact_cache,
            all_eligible_state_ids,
            resolved["prism_model"],
            resolved["property_file"],
        )
        cache_columns = [
            "state_id",
            "exact_candidate_reachability",
            "baseline_target_probability",
            "target_probability_from_candidate",
            "probability_difference_from_baseline",
            "probability_ratio_to_baseline",
            "raises_probability_from_state",
            "exact_value_source",
            *[
                column
                for column in exact_cache.columns
                if column
                not in {
                    "state_id",
                    "exact_candidate_reachability",
                    "baseline_target_probability",
                    "target_probability_from_candidate",
                    "probability_difference_from_baseline",
                    "probability_ratio_to_baseline",
                    "raises_probability_from_state",
                    "exact_value_source",
                }
                and column in {"T", "br", "bs", "fr", "fs", "i", "k", "l", "lr", "ls", "nrtr", "r", "r_ab", "recv", "rrep", "s", "s_ab", "srep", "labels"}
            ],
        ]
        exact_cache = exact_cache[cache_columns].sort_values("state_id")
        quality_rows = []
        method_rows: list[dict[str, Any]] = []
        random_rows: list[dict[str, Any]] = []
        random_repeats = 0 if quick else 100
        for (size, seed), group in ranking.groupby(
            ["training_sample_size", "sampling_seed"], dropna=False, sort=True
        ):
            parsed_seed = None if pd.isna(seed) else int(seed)
            metrics = exact_quality(group, exact_cache)
            quality_rows.append(
                {
                    "training_sample_size": int(size),
                    "sampling_seed": parsed_seed,
                    **metrics,
                }
            )
            if not skip_baselines:
                all_candidates = all_candidate_frames[(int(size), parsed_seed)]
                method_rows.extend(
                    ranking_method_rows(
                        all_candidates,
                        exact_cache,
                        sample_size=int(size),
                        sampling_seed=parsed_seed,
                        top_k=int(config["top_k"]),
                    )
                )
                random_rows.extend(
                    random_baseline_rows(
                        all_candidates,
                        exact_cache,
                        sample_size=int(size),
                        sampling_seed=parsed_seed,
                        top_k=int(config["top_k"]),
                        repeats=random_repeats,
                    )
                )
        quality = pd.DataFrame(quality_rows)
        quality_aggregated = aggregate_numeric(
            quality,
            ["training_sample_size"],
            [
                "exact_probability_raising_count",
                "exact_probability_raising_fraction",
                "mean_exact_probability_difference",
                "median_exact_probability_difference",
                "maximum_exact_probability_difference",
                "mean_exact_candidate_reachability",
                "positive_empirical_negative_exact_count",
                "negative_empirical_positive_exact_count",
            ],
        )
        method_comparison = pd.DataFrame(method_rows)
        random_distribution = pd.DataFrame(random_rows)

    prediction_aggregated = aggregate_numeric(
        prediction,
        ["training_sample_size", "model"],
        [
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "accuracy",
            "model_training_seconds",
            "total_seconds",
        ],
    )
    reliability = (
        reliability_assessment(
            prediction_aggregated, stability_aggregated, quality_aggregated, config
        )
        if not skip_exact_verification
        else pd.DataFrame()
    )

    output_root.mkdir(parents=True, exist_ok=True)
    frames = {
        "prediction_per_run.csv": prediction,
        "prediction_aggregated.csv": prediction_aggregated,
        "candidate_rankings_per_run.csv": ranking,
        "candidate_stability_per_run.csv": stability,
        "candidate_stability_aggregated.csv": stability_aggregated,
        "exact_candidate_cache.csv": exact_cache,
        "exact_candidate_quality_per_run.csv": quality,
        "exact_candidate_quality_aggregated.csv": quality_aggregated,
        "reliability_assessment.csv": reliability,
        "ranking_method_comparison.csv": method_comparison,
        "random_baseline_distribution.csv": random_distribution,
    }
    for filename, frame in frames.items():
        write_csv(frame, output_root / filename)

    completed_at = datetime.now(timezone.utc)
    input_hashes = {
        field: {
            "path": repository_relative_path(path, f"{field} path"),
            "sha256": sha256_file(path),
        }
        for field, path in resolved.items()
    }
    try:
        portable_output_root = repository_relative_path(output_root, "Output root")
        output_root_is_repository_relative = True
    except ValueError:
        # For an explicitly external --output-root, "." denotes the directory
        # containing this metadata file and avoids persisting a host path.
        portable_output_root = "."
        output_root_is_repository_relative = False
    metadata = {
        **source_provenance,
        **dependency_versions(),
        "experiment_name": config["experiment_name"],
        "execution_mode": "quick" if quick else "full_or_custom",
        "config_path": repository_relative_path(config_path, "Config path"),
        "config_sha256": sha256_file(config_path),
        "output_root": portable_output_root,
        "output_root_is_repository_relative": output_root_is_repository_relative,
        "input_artifacts": input_hashes,
        "selected_training_sample_sizes": sizes,
        "selected_sampling_seeds": seeds,
        "full_training_sampling_seed": None,
        "prediction_evaluation_count": len(prediction),
        "candidate_ranking_count": len(stability),
        "fixed_split": {
            "split_seed": int(config["split_seed"]),
            "test_size": float(config["test_size"]),
            "train_rows": len(train_pool),
            "test_rows": len(test_set),
            "train_trace_id_sha256": train_hash,
            "test_trace_id_sha256": test_hash,
            "no_overlap": True,
        },
        "candidate_support_rule": config["candidate_support_rule"],
        "candidate_statistics_population": "sampled training rows only",
        "sampling_records": sampling_records,
        "exact_verification": exact_metadata,
        "ranking_baselines_skipped": skip_baselines,
        "random_baseline_repeats_per_condition": (
            0 if quick or skip_baselines else 100
        ),
        "reliability_thresholds": config["reliability_thresholds"],
        "reliability_threshold_interpretation": (
            "Analyst-selected exploratory operational criteria, not universal "
            "statistical laws."
        ),
        "score_interpretation": (
            "combined_ranking_score is a ranking heuristic; it is neither a "
            "probability nor a causal score."
        ),
        "causal_interpretation": (
            "State-based exact probability raising does not establish historical "
            "path-conditioned causality."
        ),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "total_runtime_seconds": time.perf_counter() - started,
    }
    write_json(metadata, output_root / "prediction_metadata.json")
    return metadata


def comma_separated_ints(value: str) -> list[int]:
    """Parse a comma-separated CLI list of integers."""

    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("Values must be positive integers.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BRP k=20 training-size and candidate-stability analysis."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample-sizes", type=comma_separated_ints)
    parser.add_argument("--sampling-seeds", type=comma_separated_ints)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--exact-cache",
        type=Path,
        help="Optional existing exact-state cache to reuse as a read-only input.",
    )
    parser.add_argument("--skip-exact-verification", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    metadata = run_experiment(
        args.config,
        args.output_root,
        exact_cache_path=args.exact_cache,
        sample_sizes=args.sample_sizes,
        sampling_seeds=args.sampling_seeds,
        quick=args.quick,
        skip_exact_verification=args.skip_exact_verification,
        skip_baselines=args.skip_baselines,
    )
    print(
        f"Completed {metadata['prediction_evaluation_count']} model evaluations "
        f"and {metadata['candidate_ranking_count']} candidate rankings in "
        f"{metadata['total_runtime_seconds']:.2f} seconds."
    )


if __name__ == "__main__":
    main()
