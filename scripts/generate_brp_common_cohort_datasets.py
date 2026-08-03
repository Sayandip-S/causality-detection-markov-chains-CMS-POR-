from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml.create_brp_visited_state_dataset import (
    build_visited_state_dataset,
    select_trace_cohort,
)
from src.ml.train_brp_baselines import (
    capture_source_provenance,
    repository_relative_path,
    select_feature_columns,
    sha256_file,
)
from src.storm.model_utils import PROJECT_ROOT


WINDOWS = (5, 10, 20, 50)
DEFAULT_MINIMUM_TRANSITIONS = 50
DEFAULT_RAW_DATASET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "brp_stress_tuned_traces_10000.csv"
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed"
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "systematic"
    / "brp_stress_error"
    / "metrics"
    / "common_cohort_manifest.json"
)
NON_FEATURE_COLUMNS = {
    "trace_id",
    "target",
    "prefix_length",
    "last_state",
}


def ordered_values_sha256(values: pd.Series) -> str:
    """Hash an ordered column using newline-delimited scalar values."""

    payload = "".join(f"{value}\n" for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_membership_sha256(dataset: pd.DataFrame) -> str:
    """Hash the deterministic train/test trace-ID membership."""

    train_indices, test_indices = train_test_split(
        dataset.index,
        test_size=0.2,
        random_state=42,
        stratify=dataset["target"],
    )
    parts = ["train\n"]
    parts.extend(
        f"{trace_id}\n"
        for trace_id in dataset.loc[train_indices, "trace_id"]
    )
    parts.append("test\n")
    parts.extend(
        f"{trace_id}\n"
        for trace_id in dataset.loc[test_indices, "trace_id"]
    )
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def output_path_for_window(
    output_directory: Path,
    minimum_transitions: int,
    window: int,
) -> Path:
    """Return the common-cohort dataset path for one observation window."""

    return (
        output_directory
        / (
            "brp_stress_common"
            f"{minimum_transitions}_visited_state_dataset_k{window}.csv"
        )
    )


def generate_common_cohort_datasets(
    raw_dataset_path: Path,
    output_directory: Path,
    manifest_path: Path,
    minimum_transitions: int = DEFAULT_MINIMUM_TRANSITIONS,
) -> dict[str, Any]:
    """Generate and validate all common-cohort fixed-window datasets."""

    source_provenance = capture_source_provenance()
    raw_dataset_path = raw_dataset_path.resolve()
    output_directory = output_directory.resolve()
    manifest_path = manifest_path.resolve()
    if not raw_dataset_path.is_file():
        raise FileNotFoundError(
            f"Raw trace dataset not found: {raw_dataset_path}"
        )

    raw_traces = pd.read_csv(raw_dataset_path)
    if raw_traces["trace_id"].duplicated().any():
        raise ValueError("Raw trace_id values must be unique.")
    cohort = select_trace_cohort(
        raw_traces,
        minimum_transitions,
    )
    excluded = raw_traces.loc[
        ~raw_traces.index.isin(cohort.index)
    ]
    if cohort.empty:
        raise ValueError("The common trace cohort is empty.")

    cohort_trace_hash = ordered_values_sha256(cohort["trace_id"])
    cohort_target_hash = ordered_values_sha256(
        cohort["reached_target"].astype(int)
    )
    reference_trace_ids: pd.Series | None = None
    reference_targets: pd.Series | None = None
    reference_split_hash: str | None = None
    window_entries = []
    generated_paths = []

    output_directory.mkdir(parents=True, exist_ok=True)
    for window in WINDOWS:
        dataset, summary = build_visited_state_dataset(
            cohort,
            prefix_length=window,
            include_trace_id=True,
        )
        if len(dataset) != len(cohort):
            raise RuntimeError(
                f"k{window} did not retain the complete common cohort."
            )
        if set(dataset["prefix_length"]) != {window + 1}:
            raise RuntimeError(
                f"k{window} does not have prefix_length={window + 1}."
            )
        if summary["terminal_leakage_count"] != 0:
            raise RuntimeError(
                f"k{window} contains terminal-state leakage."
            )

        trace_ids = dataset["trace_id"].reset_index(drop=True)
        targets = dataset["target"].astype(int).reset_index(drop=True)
        if reference_trace_ids is None:
            reference_trace_ids = trace_ids
            reference_targets = targets
        elif not trace_ids.equals(reference_trace_ids):
            raise RuntimeError(
                f"k{window} trace IDs differ from the common cohort order."
            )
        elif not targets.equals(reference_targets):
            raise RuntimeError(
                f"k{window} targets differ from the common cohort order."
            )

        trace_hash = ordered_values_sha256(trace_ids)
        target_hash = ordered_values_sha256(targets)
        if trace_hash != cohort_trace_hash:
            raise RuntimeError(
                f"k{window} ordered trace-ID hash does not match the cohort."
            )
        if target_hash != cohort_target_hash:
            raise RuntimeError(
                f"k{window} target hash does not match the cohort."
            )

        features = select_feature_columns(
            dataset,
            "visited_states_only",
        )
        if NON_FEATURE_COLUMNS.intersection(features):
            raise RuntimeError(
                f"k{window} metadata or target columns entered ML features."
            )
        if any(
            not feature.startswith("visited_state_")
            for feature in features
        ):
            raise RuntimeError(
                f"k{window} contains an unexpected ML feature."
            )

        split_hash = split_membership_sha256(dataset)
        if reference_split_hash is None:
            reference_split_hash = split_hash
        elif split_hash != reference_split_hash:
            raise RuntimeError(
                f"k{window} train/test membership differs."
            )

        output_path = output_path_for_window(
            output_directory,
            minimum_transitions,
            window,
        )
        dataset.to_csv(output_path, index=False)
        generated_paths.append(output_path)
        target_count = int(targets.sum())
        success_count = len(targets) - target_count
        window_entries.append(
            {
                "k": window,
                "output_path": repository_relative_path(
                    output_path,
                    "Common-cohort output path",
                ),
                "output_sha256": sha256_file(output_path),
                "row_count": len(dataset),
                "feature_count": len(features),
                "class_counts": {
                    "target": target_count,
                    "success": success_count,
                },
                "prefix_length": window + 1,
                "terminal_leakage_count": (
                    summary["terminal_leakage_count"]
                ),
                "ordered_trace_id_sha256": trace_hash,
                "target_column_sha256": target_hash,
                "train_test_membership_sha256": split_hash,
            }
        )

    retained_target_count = int(cohort["reached_target"].sum())
    retained_success_count = len(cohort) - retained_target_count
    excluded_target_count = int(
        (excluded["terminal_label"] == "target").sum()
    )
    excluded_success_count = int(
        (excluded["terminal_label"] == "success").sum()
    )
    output_generation_timestamp = datetime.now(timezone.utc).isoformat()
    manifest = {
        **source_provenance,
        "output_generation_timestamp": output_generation_timestamp,
        "raw_dataset_path": repository_relative_path(
            raw_dataset_path,
            "Raw dataset path",
        ),
        "raw_dataset_sha256": sha256_file(raw_dataset_path),
        "cohort_rule": (
            f"number_of_transitions > {minimum_transitions}"
        ),
        "minimum_required_transitions": minimum_transitions,
        "raw_row_count": len(raw_traces),
        "retained_row_count": len(cohort),
        "excluded_row_count": len(excluded),
        "retained_target_count": retained_target_count,
        "retained_success_count": retained_success_count,
        "retained_positive_rate": (
            retained_target_count / len(cohort)
        ),
        "excluded_target_count": excluded_target_count,
        "excluded_success_count": excluded_success_count,
        "ordered_trace_id_sha256": cohort_trace_hash,
        "target_column_sha256": cohort_target_hash,
        "train_test_membership_sha256": reference_split_hash,
        "ordered_hash_encoding": (
            "UTF-8 newline-delimited scalar values in dataset row order"
        ),
        "windows": window_entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "Common cohort: "
        f"{len(cohort)} retained, {len(excluded)} excluded"
    )
    print(
        "Retained balance: "
        f"{retained_target_count} target, "
        f"{retained_success_count} success"
    )
    print(
        "Excluded balance: "
        f"{excluded_target_count} target, "
        f"{excluded_success_count} success"
    )
    for entry in window_entries:
        print(
            f"k{entry['k']}: {entry['feature_count']} features, "
            f"{entry['terminal_leakage_count']} leaked terminal states"
        )
    print(
        "Identical ordered trace IDs: yes "
        f"({cohort_trace_hash})"
    )
    print(
        "Identical ordered targets: yes "
        f"({cohort_target_hash})"
    )
    print(
        "Identical train/test membership: yes "
        f"({reference_split_hash})"
    )
    print("Generated datasets:")
    for output_path in generated_paths:
        print(f"- {output_path}")
    print(f"Manifest: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate common-cohort BRP datasets for k=5,10,20,50."
        )
    )
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        default=DEFAULT_RAW_DATASET,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--cohort-min-transitions",
        type=int,
        default=DEFAULT_MINIMUM_TRANSITIONS,
    )
    args = parser.parse_args()
    generate_common_cohort_datasets(
        raw_dataset_path=args.raw_dataset,
        output_directory=args.output_directory,
        manifest_path=args.manifest,
        minimum_transitions=args.cohort_min_transitions,
    )


if __name__ == "__main__":
    main()
