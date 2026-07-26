from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.ml.train_brp_baselines import (
    repository_relative_path,
    reproducibility_metadata,
)


STATE_FEATURE_PATTERN = re.compile(r"^visited_state_(\d+)$")


def load_json(path: Path) -> Any:
    """Load one JSON document."""

    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def safe_probability(numerator: int, denominator: int) -> float:
    """Return a probability, or NaN when its population is empty."""

    if denominator == 0:
        return float("nan")
    return numerator / denominator


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio when defined, otherwise NaN."""

    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def normalize_positive(values: pd.Series) -> pd.Series:
    """Min-max normalize non-negative evidence to the interval [0, 1]."""

    positive_values = values.clip(lower=0.0)
    minimum = positive_values.min()
    maximum = positive_values.max()
    if maximum == minimum:
        return pd.Series(
            1.0 if maximum > 0.0 else 0.0,
            index=values.index,
            dtype=float,
        )
    return (positive_values - minimum) / (maximum - minimum)


def validate_feature_schema(feature_columns: Any) -> list[str]:
    """Validate and return the persisted ordered visited-state schema."""

    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("feature_columns.json must contain a non-empty list.")
    if not all(isinstance(column, str) for column in feature_columns):
        raise ValueError("Every persisted feature name must be a string.")
    invalid = [
        column
        for column in feature_columns
        if STATE_FEATURE_PATTERN.fullmatch(column) is None
    ]
    if invalid:
        raise ValueError(
            "Candidate extraction requires only visited_state_<id> features; "
            f"invalid persisted features: {invalid}"
        )
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Persisted feature names must be unique.")
    return feature_columns


def validate_model(model: Any, feature_columns: list[str], name: str) -> None:
    """Check that a fitted model matches the persisted schema."""

    if getattr(model, "n_features_in_", None) != len(feature_columns):
        raise ValueError(f"{name} does not match the persisted feature schema.")
    model_feature_names = list(getattr(model, "feature_names_in_", []))
    if model_feature_names != feature_columns:
        raise ValueError(
            f"{name} feature names or ordering do not match the persisted schema."
        )
    classes = list(getattr(model, "classes_", []))
    if classes != [0, 1]:
        raise ValueError(f"{name} must be fitted for binary classes [0, 1].")


def extract_candidates(
    dataset_path: Path,
    model_dir: Path,
    minimum_support: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculate empirical and fitted-model evidence for candidate states."""

    if minimum_support <= 0:
        raise ValueError("minimum-support must be a positive integer.")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    feature_columns = validate_feature_schema(
        load_json(model_dir / "feature_columns.json")
    )
    metadata = load_json(model_dir / "metadata.json")
    dataset_columns = pd.read_csv(dataset_path, nrows=0).columns
    missing_features = [
        column for column in feature_columns if column not in dataset_columns
    ]
    if missing_features:
        raise ValueError(f"Dataset is missing expected features: {missing_features}")

    dataset = pd.read_csv(dataset_path)
    if dataset.empty:
        raise ValueError("Dataset contains no retained traces.")
    if "target" not in dataset.columns:
        raise ValueError("Dataset must contain a target column.")
    if not set(dataset["target"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target column must be binary with no missing values.")
    if dataset["target"].isna().any():
        raise ValueError("Target column must be binary with no missing values.")

    features = dataset[feature_columns]
    invalid_binary_columns = [
        column
        for column in feature_columns
        if features[column].isna().any()
        or not set(features[column].unique()).issubset({0, 1})
    ]
    if invalid_binary_columns:
        raise ValueError(
            "Visited-state features must contain only 0 and 1: "
            f"{invalid_binary_columns}"
        )

    logistic_model = joblib.load(model_dir / "logistic_regression.joblib")
    random_forest_model = joblib.load(model_dir / "random_forest.joblib")
    validate_model(logistic_model, feature_columns, "Logistic Regression")
    validate_model(random_forest_model, feature_columns, "Random Forest")

    logistic_coefficients = np.asarray(logistic_model.coef_)
    forest_importances = np.asarray(random_forest_model.feature_importances_)
    if logistic_coefficients.shape != (1, len(feature_columns)):
        raise ValueError("Unexpected Logistic Regression coefficient shape.")
    if forest_importances.shape != (len(feature_columns),):
        raise ValueError("Unexpected Random Forest importance shape.")

    targets = dataset["target"].astype(int)
    total_count = len(dataset)
    total_targets = int(targets.sum())
    rows: list[dict[str, int | float]] = []

    for position, feature_name in enumerate(feature_columns):
        state_match = STATE_FEATURE_PATTERN.fullmatch(feature_name)
        if state_match is None:
            continue
        visited = features[feature_name].astype(bool)
        visited_count = int(visited.sum())
        not_visited_count = total_count - visited_count
        targets_when_visited = int(targets[visited].sum())
        targets_when_not_visited = total_targets - targets_when_visited
        success_when_visited = visited_count - targets_when_visited
        target_probability_visited = safe_probability(
            targets_when_visited, visited_count
        )
        target_probability_not_visited = safe_probability(
            targets_when_not_visited, not_visited_count
        )
        probability_difference = (
            target_probability_visited - target_probability_not_visited
            if np.isfinite(target_probability_visited)
            and np.isfinite(target_probability_not_visited)
            else float("nan")
        )
        coefficient = float(logistic_coefficients[0, position])
        rows.append(
            {
                "state_id": int(state_match.group(1)),
                "logistic_regression_coefficient": coefficient,
                "absolute_logistic_regression_coefficient": abs(coefficient),
                "random_forest_feature_importance": float(
                    forest_importances[position]
                ),
                "total_retained_trace_count": total_count,
                "visited_trace_count": visited_count,
                "target_traces_when_visited": targets_when_visited,
                "success_traces_when_visited": success_when_visited,
                "target_probability_when_visited": target_probability_visited,
                "target_probability_when_not_visited": (
                    target_probability_not_visited
                ),
                "probability_difference": probability_difference,
                "probability_ratio": safe_ratio(
                    target_probability_visited,
                    target_probability_not_visited,
                ),
                "support_fraction": visited_count / total_count,
            }
        )

    candidates = pd.DataFrame(rows)
    candidates = candidates[
        candidates["visited_trace_count"] >= minimum_support
    ].copy()
    if candidates.empty:
        raise ValueError("No candidate states meet the minimum support threshold.")

    candidates["normalized_positive_logistic_coefficient"] = (
        normalize_positive(candidates["logistic_regression_coefficient"])
    )
    candidates["normalized_random_forest_importance"] = normalize_positive(
        candidates["random_forest_feature_importance"]
    )
    candidates["normalized_positive_probability_difference"] = (
        normalize_positive(candidates["probability_difference"].fillna(0.0))
    )
    candidates["support_reliability_weight"] = (
        candidates["visited_trace_count"]
        / (candidates["visited_trace_count"] + minimum_support)
    )
    candidates["combined_ranking_score"] = candidates[
        [
            "normalized_positive_logistic_coefficient",
            "normalized_random_forest_importance",
            "normalized_positive_probability_difference",
        ]
    ].mean(axis=1) * candidates["support_reliability_weight"]
    candidates = candidates.sort_values(
        ["combined_ranking_score", "visited_trace_count", "state_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))

    extraction_metadata = {
        "observation_window": metadata.get("observation_window"),
        "candidate_definition": (
            "A visited_state_<id> feature meeting minimum support, ranked by "
            "the mean of min-max normalized positive Logistic Regression "
            "coefficient, Random Forest importance, and positive full-dataset "
            "empirical probability difference, multiplied by the support "
            "reliability weight visited_count/(visited_count+minimum_support)."
        ),
        "minimum_support": minimum_support,
        "baseline_target_probability": total_targets / total_count,
        "retained_row_count": total_count,
        "dataset_path": repository_relative_path(
            dataset_path,
            "Dataset path",
        ),
        "model_directory": repository_relative_path(
            model_dir,
            "Model directory",
        ),
        "methodological_limitations": [
            "Empirical candidate statistics use the complete processed dataset, "
            "including rows used to train and test the saved models.",
            "This is exploratory candidate discovery, not unbiased performance "
            "evaluation or confirmatory causal evidence.",
            "Logistic coefficients and Random Forest importances are associative "
            "and can be affected by correlated visited-state features.",
            "Candidate ranking is data- and model-dependent and requires separate "
            "validation and model-checking analysis.",
        ],
        **reproducibility_metadata(
            datetime.now(timezone.utc).isoformat()
        ),
    }
    return candidates, extraction_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ranked BRP candidate states from persisted models."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--minimum-support", type=int, default=50)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("top-k must be a positive integer.")

    candidates, metadata = extract_candidates(
        dataset_path=args.dataset.resolve(),
        model_dir=args.model_dir.resolve(),
        minimum_support=args.minimum_support,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output, index=False)
    metadata_output = output.with_suffix(".metadata.json")
    metadata_output.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    display_columns = [
        "rank",
        "state_id",
        "combined_ranking_score",
        "logistic_regression_coefficient",
        "random_forest_feature_importance",
        "visited_trace_count",
        "target_probability_when_visited",
        "probability_difference",
    ]
    print(f"Top {min(args.top_k, len(candidates))} candidate states:")
    print(candidates[display_columns].head(args.top_k).to_string(index=False))
    print(f"\nCandidate CSV written to: {output}")
    print(f"Candidate metadata written to: {metadata_output}")


if __name__ == "__main__":
    main()
