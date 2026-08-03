from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.generate_brp_common_cohort_datasets import (
    ordered_values_sha256,
)
from src.ml.train_brp_baselines import (
    calculate_binary_metrics,
    create_models,
    dependency_versions,
    git_output,
    get_positive_probabilities,
    repository_relative_path,
    select_feature_columns,
    sha256_file,
)
from src.storm.model_utils import PROJECT_ROOT


WINDOWS = (5, 10, 20, 50)
TEST_SIZE = 0.2
RANDOM_SEED = 42
METRICS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "systematic"
    / "brp_stress_error"
    / "metrics"
)
DEFAULT_MANIFEST_PATH = METRICS_DIRECTORY / "common_cohort_manifest.json"
DEFAULT_PER_MODEL_PATH = (
    METRICS_DIRECTORY / "common_cohort_per_model.csv"
)
DEFAULT_SUMMARY_PATH = (
    METRICS_DIRECTORY / "common_cohort_summary.json"
)
DEFAULT_COMPARISON_PATH = (
    METRICS_DIRECTORY / "operational_vs_common_cohort.csv"
)
DEFAULT_OPERATIONAL_METRICS_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "brp_fixed_windows"
    / "combined_metrics.csv"
)
RESULT_COLUMNS = [
    "observation_window",
    "model",
    "total_rows",
    "train_rows",
    "test_rows",
    "feature_count",
    "positive_rate",
    "train_positive_rate",
    "test_positive_rate",
    "train_trace_id_sha256",
    "test_trace_id_sha256",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "tn",
    "fp",
    "fn",
    "tp",
    "data_preparation_seconds",
    "training_seconds",
    "prediction_seconds",
    "total_seconds",
]
FORBIDDEN_FEATURES = {
    "trace_id",
    "target",
    "prefix_length",
    "last_state",
}
SOURCE_PROVENANCE_FIELDS = (
    "source_git_commit_sha",
    "source_git_branch",
    "source_working_tree_dirty",
    "provenance_capture_timestamp",
)


def validate_manifest_source_provenance(
    manifest: dict[str, Any],
    *,
    current_commit_sha: str | None = None,
) -> dict[str, str | bool]:
    """Validate and return the source snapshot stored in the manifest."""

    missing = [
        field for field in SOURCE_PROVENANCE_FIELDS if field not in manifest
    ]
    if missing:
        raise ValueError(
            "Common-cohort manifest is missing source provenance fields: "
            f"{missing}. Regenerate the manifest from the current source."
        )

    for field in (
        "source_git_commit_sha",
        "source_git_branch",
        "provenance_capture_timestamp",
    ):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise ValueError(
                f"Common-cohort manifest {field} must be a non-empty string."
            )
    if not isinstance(manifest["source_working_tree_dirty"], bool):
        raise ValueError(
            "Common-cohort manifest source_working_tree_dirty must be boolean."
        )

    current_sha = current_commit_sha or git_output("rev-parse", "HEAD")
    manifest_sha = manifest["source_git_commit_sha"]
    if manifest_sha != current_sha:
        raise ValueError(
            "Common-cohort manifest source SHA does not match the current "
            f"checkout ({manifest_sha} versus {current_sha}). Regenerate "
            "the common-cohort datasets before baseline evaluation."
        )

    return {field: manifest[field] for field in SOURCE_PROVENANCE_FIELDS}


def summary_provenance_fields(
    source_provenance: dict[str, str | bool],
    output_generation_timestamp: str,
) -> dict[str, Any]:
    """Build summary provenance without querying Git after output writes."""

    return {
        **source_provenance,
        "output_generation_timestamp": output_generation_timestamp,
        # Legacy aliases remain for consumers of the earlier metadata schema.
        "git_commit_sha": source_provenance["source_git_commit_sha"],
        "git_branch": source_provenance["source_git_branch"],
        "working_tree_dirty": source_provenance[
            "source_working_tree_dirty"
        ],
        **dependency_versions(),
        "timestamp": output_generation_timestamp,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate the common-cohort manifest."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Common-cohort manifest not found: {path}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("windows")
    if not isinstance(entries, list):
        raise ValueError("Manifest windows must be a list.")
    if [entry.get("k") for entry in entries] != list(WINDOWS):
        raise ValueError(
            f"Manifest must contain windows in order {WINDOWS}."
        )
    validate_manifest_source_provenance(manifest)
    return manifest


def load_common_datasets(
    manifest: dict[str, Any],
) -> tuple[dict[int, pd.DataFrame], list[dict[str, Any]]]:
    """Load datasets and verify paths, hashes, cohort order, and targets."""

    datasets: dict[int, pd.DataFrame] = {}
    inputs = []
    reference_trace_ids: pd.Series | None = None
    reference_targets: pd.Series | None = None
    for entry in manifest["windows"]:
        window = int(entry["k"])
        path = (PROJECT_ROOT / entry["output_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Common-cohort k{window} dataset not found: {path}"
            )
        digest = sha256_file(path)
        if digest != entry["output_sha256"]:
            raise ValueError(
                f"k{window} dataset SHA-256 differs from the manifest."
            )

        dataset = pd.read_csv(path)
        required = {"trace_id", "target"}
        missing = sorted(required - set(dataset.columns))
        if missing:
            raise ValueError(
                f"k{window} dataset is missing columns: {missing}"
            )
        if dataset["trace_id"].duplicated().any():
            raise ValueError(f"k{window} trace IDs must be unique.")
        if len(dataset) != manifest["retained_row_count"]:
            raise ValueError(
                f"k{window} row count differs from the manifest cohort."
            )

        trace_ids = dataset["trace_id"].reset_index(drop=True)
        targets = dataset["target"].astype(int).reset_index(drop=True)
        if reference_trace_ids is None:
            reference_trace_ids = trace_ids
            reference_targets = targets
        elif not trace_ids.equals(reference_trace_ids):
            raise ValueError(
                f"k{window} ordered trace IDs differ from k{WINDOWS[0]}."
            )
        elif not targets.equals(reference_targets):
            raise ValueError(
                f"k{window} ordered targets differ from k{WINDOWS[0]}."
            )

        datasets[window] = dataset
        inputs.append(
            {
                "observation_window": window,
                "path": repository_relative_path(
                    path,
                    f"k{window} dataset path",
                ),
                "sha256": digest,
            }
        )

    return datasets, inputs


def create_shared_trace_split(
    reference: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.Series, pd.Series]:
    """Create one stratified split of trace IDs for every window."""

    train_trace_ids, test_trace_ids = train_test_split(
        reference["trace_id"],
        test_size=test_size,
        random_state=random_seed,
        stratify=reference["target"],
    )
    train_trace_ids = train_trace_ids.reset_index(drop=True)
    test_trace_ids = test_trace_ids.reset_index(drop=True)
    overlap = set(train_trace_ids).intersection(test_trace_ids)
    if overlap:
        raise RuntimeError(
            f"Train/test trace-ID overlap detected: {sorted(overlap)}"
        )
    if len(train_trace_ids) + len(test_trace_ids) != len(reference):
        raise RuntimeError("Shared split does not cover the full cohort.")
    return train_trace_ids, test_trace_ids


def prepare_window_data(
    dataset: pd.DataFrame,
    train_trace_ids: pd.Series,
    test_trace_ids: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    list[str],
]:
    """Index one window with the precomputed shared trace-ID split."""

    feature_columns = select_feature_columns(
        dataset,
        "visited_states_only",
    )
    if FORBIDDEN_FEATURES.intersection(feature_columns):
        raise RuntimeError("Metadata or target columns entered ML features.")
    if any(
        not column.startswith("visited_state_")
        for column in feature_columns
    ):
        raise RuntimeError("Unexpected non-visited-state ML feature.")

    indexed = dataset.set_index("trace_id", verify_integrity=True)
    missing_train = set(train_trace_ids) - set(indexed.index)
    missing_test = set(test_trace_ids) - set(indexed.index)
    if missing_train or missing_test:
        raise ValueError(
            "Dataset does not contain every shared-split trace ID."
        )
    train_rows = indexed.loc[train_trace_ids]
    test_rows = indexed.loc[test_trace_ids]
    return (
        train_rows[feature_columns],
        test_rows[feature_columns],
        train_rows["target"].astype(int),
        test_rows["target"].astype(int),
        feature_columns,
    )


def validate_results(results: pd.DataFrame) -> None:
    """Validate the complete 4-window by 3-model result table."""

    if list(results.columns) != RESULT_COLUMNS:
        raise RuntimeError("Common-cohort result schema is incorrect.")
    if len(results) != 12:
        raise RuntimeError(
            f"Expected 12 result rows, found {len(results)}."
        )
    if results[
        ["observation_window", "model"]
    ].duplicated().any():
        raise RuntimeError("Duplicate window/model result pairs found.")
    if results["train_trace_id_sha256"].nunique() != 1:
        raise RuntimeError("Train trace-ID hashes differ across windows.")
    if results["test_trace_id_sha256"].nunique() != 1:
        raise RuntimeError("Test trace-ID hashes differ across windows.")
    confusion_total = results[["tn", "fp", "fn", "tp"]].sum(axis=1)
    if not confusion_total.equals(results["test_rows"]):
        raise RuntimeError("Confusion-matrix totals differ from test_rows.")
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "positive_rate",
        "train_positive_rate",
        "test_positive_rate",
    ]
    valid_metrics = results[metric_columns].apply(
        lambda column: column.between(0.0, 1.0)
    )
    if not valid_metrics.all(axis=None):
        raise RuntimeError("One or more metric values are outside [0, 1].")
    timing_columns = [
        "data_preparation_seconds",
        "training_seconds",
        "prediction_seconds",
        "total_seconds",
    ]
    if (results[timing_columns] < 0.0).any(axis=None):
        raise RuntimeError("Timing values must be non-negative.")


def compare_with_operational(
    common_results: pd.DataFrame,
    operational_path: Path,
) -> pd.DataFrame:
    """Join common-cohort metrics with existing operational results."""

    if not operational_path.is_file():
        raise FileNotFoundError(
            f"Operational metrics not found: {operational_path}"
        )
    operational = pd.read_csv(operational_path).rename(
        columns={"window": "observation_window"}
    )
    required = {"observation_window", "model", "f1", "roc_auc"}
    missing = sorted(required - set(operational.columns))
    if missing:
        raise ValueError(
            f"Operational metrics are missing columns: {missing}"
        )
    common = common_results[
        ["observation_window", "model", "f1", "roc_auc"]
    ]
    comparison = common.merge(
        operational[list(required)],
        on=["observation_window", "model"],
        suffixes=("_common_cohort", "_operational"),
        validate="one_to_one",
    )
    if len(comparison) != 12:
        raise RuntimeError("Operational comparison must contain 12 rows.")
    comparison["common_cohort_f1_minus_operational_f1"] = (
        comparison["f1_common_cohort"]
        - comparison["f1_operational"]
    )
    comparison[
        "common_cohort_roc_auc_minus_operational_roc_auc"
    ] = (
        comparison["roc_auc_common_cohort"]
        - comparison["roc_auc_operational"]
    )
    return comparison[
        [
            "observation_window",
            "model",
            "f1_common_cohort",
            "f1_operational",
            "common_cohort_f1_minus_operational_f1",
            "roc_auc_common_cohort",
            "roc_auc_operational",
            "common_cohort_roc_auc_minus_operational_roc_auc",
        ]
    ]


def run_common_cohort_baselines(
    manifest_path: Path,
    per_model_path: Path,
    summary_path: Path,
    comparison_path: Path,
    operational_metrics_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run all common-cohort baselines with one shared trace-ID split."""

    run_started = time.perf_counter()
    manifest_path = manifest_path.resolve()
    operational_metrics_path = operational_metrics_path.resolve()
    manifest = load_manifest(manifest_path)
    source_provenance = {
        field: manifest[field] for field in SOURCE_PROVENANCE_FIELDS
    }
    datasets, input_datasets = load_common_datasets(manifest)
    reference = datasets[WINDOWS[0]]
    train_trace_ids, test_trace_ids = create_shared_trace_split(
        reference
    )
    train_hash = ordered_values_sha256(train_trace_ids)
    test_hash = ordered_values_sha256(test_trace_ids)
    if set(train_trace_ids).intersection(test_trace_ids):
        raise RuntimeError("A trace appears in both train and test.")

    reference_train_targets = (
        reference.set_index("trace_id")
        .loc[train_trace_ids, "target"]
        .astype(int)
        .reset_index(drop=True)
    )
    reference_test_targets = (
        reference.set_index("trace_id")
        .loc[test_trace_ids, "target"]
        .astype(int)
        .reset_index(drop=True)
    )
    rows = []
    feature_counts: dict[str, int] = {}

    for window in WINDOWS:
        dataset = datasets[window]
        preparation_started = time.perf_counter()
        x_train, x_test, y_train, y_test, feature_columns = (
            prepare_window_data(
                dataset,
                train_trace_ids,
                test_trace_ids,
            )
        )
        data_preparation_seconds = (
            time.perf_counter() - preparation_started
        )
        if not y_train.reset_index(drop=True).equals(
            reference_train_targets
        ):
            raise RuntimeError(
                f"k{window} training targets differ from the shared split."
            )
        if not y_test.reset_index(drop=True).equals(
            reference_test_targets
        ):
            raise RuntimeError(
                f"k{window} test targets differ from the shared split."
            )
        feature_counts[f"k{window}"] = len(feature_columns)

        for model_name, model in create_models(RANDOM_SEED).items():
            training_started = time.perf_counter()
            model.fit(x_train, y_train)
            training_seconds = time.perf_counter() - training_started

            prediction_started = time.perf_counter()
            predictions = model.predict(x_test)
            probabilities = get_positive_probabilities(
                model,
                x_test,
            )
            prediction_seconds = (
                time.perf_counter() - prediction_started
            )
            metrics = calculate_binary_metrics(
                y_test,
                predictions,
                probabilities,
                training_row_count=len(x_train),
                positive_rate=float(dataset["target"].mean()),
                number_of_features=len(feature_columns),
            )
            rows.append(
                {
                    "observation_window": window,
                    "model": model_name,
                    "total_rows": len(dataset),
                    "train_rows": len(x_train),
                    "test_rows": len(x_test),
                    "feature_count": len(feature_columns),
                    "positive_rate": float(
                        dataset["target"].mean()
                    ),
                    "train_positive_rate": float(y_train.mean()),
                    "test_positive_rate": float(y_test.mean()),
                    "train_trace_id_sha256": train_hash,
                    "test_trace_id_sha256": test_hash,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tp": metrics["tp"],
                    "data_preparation_seconds": (
                        data_preparation_seconds
                    ),
                    "training_seconds": training_seconds,
                    "prediction_seconds": prediction_seconds,
                    "total_seconds": (
                        data_preparation_seconds
                        + training_seconds
                        + prediction_seconds
                    ),
                }
            )

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    validate_results(results)
    comparison = compare_with_operational(
        results,
        operational_metrics_path,
    )

    per_model_path = per_model_path.resolve()
    summary_path = summary_path.resolve()
    comparison_path = comparison_path.resolve()
    per_model_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(per_model_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    completed_at = datetime.now(timezone.utc)
    total_runtime_seconds = time.perf_counter() - run_started
    output_generation_timestamp = completed_at.isoformat()
    summary = {
        **summary_provenance_fields(
            source_provenance,
            output_generation_timestamp,
        ),
        "input_datasets": input_datasets,
        "common_cohort_manifest": {
            "path": repository_relative_path(
                manifest_path,
                "Common-cohort manifest path",
            ),
            "sha256": sha256_file(manifest_path),
        },
        "cohort_definition": {
            "rule": manifest["cohort_rule"],
            "row_count": manifest["retained_row_count"],
            "target_count": manifest["retained_target_count"],
            "success_count": manifest["retained_success_count"],
            "positive_rate": manifest["retained_positive_rate"],
        },
        "split_configuration": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_SEED,
            "stratify": "target",
            "generated_once_from": (
                "common-cohort k5 trace_id and target columns"
            ),
            "train_rows": len(train_trace_ids),
            "test_rows": len(test_trace_ids),
            "train_target_count": int(
                reference_train_targets.sum()
            ),
            "test_target_count": int(
                reference_test_targets.sum()
            ),
            "no_trace_id_overlap": True,
        },
        "split_trace_id_hashes": {
            "train": train_hash,
            "test": test_hash,
        },
        "per_window_feature_counts": feature_counts,
        "operational_metrics": {
            "path": repository_relative_path(
                operational_metrics_path,
                "Operational metrics path",
            ),
            "sha256": sha256_file(operational_metrics_path),
        },
        "total_runtime_seconds": total_runtime_seconds,
        "timing_definition": (
            "Per-model total_seconds is the sum of window data preparation, "
            "model fitting, and prediction plus probability prediction."
        ),
        "methodological_interpretation": [
            "All windows use the same 9,177 traces, class labels, and exact "
            "train/test trace-ID membership.",
            "Differences across windows are therefore attributable mainly to "
            "additional observed-prefix information rather than cohort "
            "composition or split membership.",
            "Features encode only whether states were present in the prefix; "
            "they do not preserve full state sequence order or multiplicity.",
            "This is a predictive comparison and does not establish causal "
            "effects of states or longer observation windows.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    display = results[
        [
            "observation_window",
            "model",
            "feature_count",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "training_seconds",
        ]
    ]
    print("Common-cohort baseline results:")
    print(display.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nPer-model results: {per_model_path}")
    print(f"Summary: {summary_path}")
    print(f"Operational comparison: {comparison_path}")
    return results, comparison, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run BRP fixed-window baselines on one shared common cohort."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--per-model-output",
        type=Path,
        default=DEFAULT_PER_MODEL_PATH,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=DEFAULT_COMPARISON_PATH,
    )
    parser.add_argument(
        "--operational-metrics",
        type=Path,
        default=DEFAULT_OPERATIONAL_METRICS_PATH,
    )
    args = parser.parse_args()
    run_common_cohort_baselines(
        manifest_path=args.manifest,
        per_model_path=args.per_model_output,
        summary_path=args.summary_output,
        comparison_path=args.comparison_output,
        operational_metrics_path=args.operational_metrics,
    )


if __name__ == "__main__":
    main()
