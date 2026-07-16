from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.storm.model_utils import PROJECT_ROOT


DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "brp_prefix_dataset.csv"
)

FEATURE_SET_CHOICES = (
    "visited_states_only",
    "no_last_state",
    "all_features",
)

ALWAYS_EXCLUDED_FEATURES = {
    "target",
    "trace_id",
    "terminal_label",
    "reached_target",
}

MODEL_SLUGS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def select_feature_columns(
    dataset: pd.DataFrame,
    feature_set: str,
) -> list[str]:
    """Select an ordered, leakage-safe feature-column list."""

    if feature_set == "visited_states_only":
        feature_columns = [
            column
            for column in dataset.columns
            if column.startswith("visited_state_")
        ]
    elif feature_set == "no_last_state":
        feature_columns = [
            column
            for column in dataset.columns
            if column not in ALWAYS_EXCLUDED_FEATURES
            and column != "last_state"
        ]
    elif feature_set == "all_features":
        feature_columns = [
            column
            for column in dataset.columns
            if column not in ALWAYS_EXCLUDED_FEATURES
        ]
    else:
        raise ValueError(
            f"Unknown feature set {feature_set!r}. "
            f"Expected one of {FEATURE_SET_CHOICES}."
        )

    if not feature_columns:
        raise ValueError(
            f"Feature set {feature_set!r} selected no columns."
        )

    non_numeric_columns = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(dataset[column])
    ]

    if non_numeric_columns:
        raise ValueError(
            "Selected feature columns must be numeric: "
            f"{non_numeric_columns}"
        )

    return feature_columns


def create_models(random_seed: int) -> dict[str, Any]:
    """Create the three baseline classifiers."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_seed,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=random_seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ),
    }


def get_positive_probabilities(
    model: Any,
    x_test: pd.DataFrame,
) -> Any | None:
    """Return probabilities for target class 1 when supported."""

    if not hasattr(model, "predict_proba"):
        return None

    class_labels = list(model.classes_)

    if 1 not in class_labels:
        return None

    positive_class_index = class_labels.index(1)
    return model.predict_proba(x_test)[:, positive_class_index]


def evaluate_model(
    name: str,
    model: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    positive_rate: float,
    number_of_features: int,
) -> tuple[dict[str, float | int | None], Any, Any | None]:
    """Train one classifier and return metrics and predictions."""

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = get_positive_probabilities(model, x_test)
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    roc_auc = None

    if probabilities is not None and y_test.nunique() == 2:
        roc_auc = float(roc_auc_score(y_test, probabilities))

    metrics: dict[str, float | int | None] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "training_row_count": len(x_train),
        "test_row_count": len(x_test),
        "positive_rate": positive_rate,
        "number_of_features": number_of_features,
    }

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print()
    print("Confusion matrix:")
    print([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return metrics, predictions, probabilities


def run_experiment(
    dataset_path: Path,
    feature_set: str = "visited_states_only",
    test_size: float = 0.2,
    random_seed: int = 42,
    metrics_output: Path | None = None,
    predictions_output: Path | None = None,
) -> dict[str, Any]:
    """Train and evaluate all BRP baseline models on one dataset."""

    dataset_path = dataset_path.resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if not 0.0 < test_size < 1.0:
        raise ValueError("test-size must be in (0, 1).")

    dataset = pd.read_csv(dataset_path)

    if "target" not in dataset.columns:
        raise ValueError("Dataset must contain a target column.")

    if dataset.empty:
        raise ValueError("Dataset contains no rows.")

    y = dataset["target"]

    if not set(y.unique()).issubset({0, 1}) or y.nunique() != 2:
        raise ValueError(
            "Target column must contain both binary classes 0 and 1."
        )

    feature_columns = select_feature_columns(dataset, feature_set)
    x = dataset[feature_columns]
    train_indices, test_indices = train_test_split(
        dataset.index,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )
    x_train = x.loc[train_indices]
    x_test = x.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]
    positive_rate = float(y.mean())

    print("Loaded dataset:")
    print(f"Path: {dataset_path}")
    print(f"Rows: {len(dataset)}")
    print(f"Feature set: {feature_set}")
    print(f"Number of features: {len(feature_columns)}")
    print()
    print("Class distribution:")
    print(y.value_counts())
    print()
    print("Train class distribution:")
    print(y_train.value_counts())
    print()
    print("Test class distribution:")
    print(y_test.value_counts())

    models = create_models(random_seed)
    model_parameters = {
        name: model.get_params(deep=False)
        for name, model in models.items()
    }
    all_metrics: dict[str, dict[str, float | int | None]] = {}
    predictions_dataset = pd.DataFrame(
        {
            "original_row_index": test_indices,
            "true_target": y_test.to_numpy(),
        }
    )

    for name, model in models.items():
        metrics, predictions, probabilities = evaluate_model(
            name=name,
            model=model,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            positive_rate=positive_rate,
            number_of_features=len(feature_columns),
        )
        all_metrics[name] = metrics
        model_slug = MODEL_SLUGS[name]
        predictions_dataset[f"{model_slug}_prediction"] = predictions

        if probabilities is not None:
            predictions_dataset[
                f"{model_slug}_target_probability"
            ] = probabilities

    results = {
        "input_dataset_path": str(dataset_path),
        "input_dataset_sha256": sha256_file(dataset_path),
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "test_size": test_size,
        "random_seed": random_seed,
        "model_parameters": model_parameters,
        "metrics": all_metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if metrics_output is not None:
        metrics_output = metrics_output.resolve()
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        metrics_output.write_text(
            json.dumps(results, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Metrics JSON written to: {metrics_output}")

    if predictions_output is not None:
        predictions_output = predictions_output.resolve()
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        predictions_dataset.to_csv(predictions_output, index=False)
        print(f"Predictions CSV written to: {predictions_output}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train baseline ML models on the BRP "
            "prefix-feature dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to prefix dataset CSV.",
    )
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SET_CHOICES,
        default="visited_states_only",
        help="Columns used as model features (default: visited_states_only).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows used for testing (default: 0.2).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for splitting and models (default: 42).",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional output path for metrics JSON.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        help="Optional output path for test-set predictions CSV.",
    )
    args = parser.parse_args()

    run_experiment(
        dataset_path=args.dataset,
        feature_set=args.feature_set,
        test_size=args.test_size,
        random_seed=args.random_seed,
        metrics_output=args.metrics_output,
        predictions_output=args.predictions_output,
    )


if __name__ == "__main__":
    main()
