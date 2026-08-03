from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_brp_common_cohort_baselines import (  # noqa: E402
    create_shared_trace_split,
    prepare_window_data,
    summary_provenance_fields,
    validate_manifest_source_provenance,
)


def example_window(extra_feature: str) -> pd.DataFrame:
    """Return one common cohort with a window-specific visited feature."""

    return pd.DataFrame(
        {
            "trace_id": list(range(20)),
            "visited_state_0": [1] * 20,
            extra_feature: [index % 2 for index in range(20)],
            "prefix_length": [6] * 20,
            "last_state": list(range(20)),
            "target": [0, 1] * 10,
        }
    )


def test_shared_trace_split_is_reused_across_windows() -> None:
    k5 = example_window("visited_state_5")
    k10 = example_window("visited_state_10")
    train_ids, test_ids = create_shared_trace_split(k5)

    assert set(train_ids).isdisjoint(test_ids)
    assert len(train_ids) == 16
    assert len(test_ids) == 4

    prepared = [
        prepare_window_data(dataset, train_ids, test_ids)
        for dataset in (k5, k10)
    ]
    for x_train, x_test, y_train, y_test, features in prepared:
        assert len(x_train) == 16
        assert len(x_test) == 4
        assert y_train.sum() == 8
        assert y_test.sum() == 2
        assert all(
            feature.startswith("visited_state_")
            for feature in features
        )
        assert {
            "trace_id",
            "target",
            "prefix_length",
            "last_state",
        }.isdisjoint(features)

    assert prepared[0][2].reset_index(drop=True).equals(
        prepared[1][2].reset_index(drop=True)
    )
    assert prepared[0][3].reset_index(drop=True).equals(
        prepared[1][3].reset_index(drop=True)
    )


def source_provenance() -> dict[str, str | bool]:
    return {
        "source_git_commit_sha": "source-sha",
        "source_git_branch": "test-branch",
        "source_working_tree_dirty": False,
        "provenance_capture_timestamp": "2026-08-03T00:00:00+00:00",
    }


def test_summary_reuses_manifest_source_provenance() -> None:
    manifest = source_provenance()
    validated = validate_manifest_source_provenance(
        manifest,
        current_commit_sha="source-sha",
    )
    summary = summary_provenance_fields(
        validated,
        "2026-08-03T00:01:00+00:00",
    )

    assert summary["source_git_commit_sha"] == "source-sha"
    assert summary["source_working_tree_dirty"] is False
    assert summary["git_commit_sha"] == "source-sha"
    assert summary["working_tree_dirty"] is False
    assert summary["provenance_capture_timestamp"] != (
        summary["output_generation_timestamp"]
    )


def test_manifest_source_sha_mismatch_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="source SHA does not match the current checkout",
    ):
        validate_manifest_source_provenance(
            source_provenance(),
            current_commit_sha="different-sha",
        )
