from __future__ import annotations

import argparse
from pathlib import Path

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


def evaluate_model(
    name: str,
    model,
    x_train,
    x_test,
    y_train,
    y_test,
) -> None:
    """
    Train and evaluate one classifier.
    """

    model.fit(
        x_train,
        y_train,
    )

    predictions = model.predict(
        x_test
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Accuracy:  "
        f"{accuracy_score(y_test, predictions):.4f}"
    )
    print(
        f"Precision: "
        f"{precision_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:    "
        f"{recall_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"F1-score:  "
        f"{f1_score(y_test, predictions, zero_division=0):.4f}"
    )

    print()
    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )


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

    args = parser.parse_args()

    dataset = pd.read_csv(
        args.dataset
    )

    x = dataset.drop(
        columns=["target"]
    )

    y = dataset["target"]

    print("Loaded dataset:")
    print(f"Rows: {len(dataset)}")
    print(f"Features: {list(x.columns)}")
    print()
    print("Class distribution:")
    print(y.value_counts())
    print()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Train class distribution:")
    print(y_train.value_counts())
    print()
    print("Test class distribution:")
    print(y_test.value_counts())

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        evaluate_model(
            name=name,
            model=model,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )


if __name__ == "__main__":
    main()