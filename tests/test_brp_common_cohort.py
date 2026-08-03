from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.create_brp_visited_state_dataset import (
    build_visited_state_dataset,
    select_trace_cohort,
)
from src.ml.train_brp_baselines import select_feature_columns
import scripts.generate_brp_common_cohort_datasets as cohort_generator
import src.ml.train_brp_baselines as baseline_training


SOURCE_STATUS_ARGUMENTS = (
    "status",
    "--porcelain",
    "--untracked-files=all",
    "--",
    ".",
    ":(exclude)results/systematic/brp_stress_error",
    ":(exclude)results/systematic/brp_stress_error/**",
)


def example_traces() -> pd.DataFrame:
    """Return traces with one short row and two common-cohort rows."""

    return pd.DataFrame(
        {
            "trace_id": [10, 11, 12],
            "state_ids": [
                "0|1|90",
                "0|1|2|3|91",
                "0|4|5|6|7|92",
            ],
            "terminal_label": ["target", "target", "success"],
            "reached_target": [1, 1, 0],
            "number_of_transitions": [2, 4, 5],
        }
    )


def test_common_cohort_is_shared_across_windows_without_leakage() -> None:
    traces = example_traces()
    cohort = select_trace_cohort(traces, minimum_transitions=3)

    assert cohort["trace_id"].tolist() == [11, 12]

    datasets = {}
    for window in (1, 3):
        dataset, summary = build_visited_state_dataset(
            cohort,
            prefix_length=window,
            include_trace_id=True,
        )
        datasets[window] = dataset
        assert dataset["prefix_length"].tolist() == [
            window + 1,
            window + 1,
        ]
        assert summary["terminal_leakage_count"] == 0

    assert datasets[1]["trace_id"].equals(datasets[3]["trace_id"])
    assert datasets[1]["target"].equals(datasets[3]["target"])


def test_trace_metadata_is_excluded_from_visited_state_features() -> None:
    cohort = select_trace_cohort(
        example_traces(),
        minimum_transitions=3,
    )
    dataset, _ = build_visited_state_dataset(
        cohort,
        prefix_length=1,
        include_trace_id=True,
    )

    features = select_feature_columns(
        dataset,
        "visited_states_only",
    )

    assert features
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


def test_operational_window_behavior_remains_backward_compatible() -> None:
    dataset, summary = build_visited_state_dataset(
        example_traces(),
        prefix_length=3,
    )

    assert "trace_id" not in dataset.columns
    assert len(dataset) == 2
    assert summary["excluded_target_count"] == 1
    assert summary["excluded_success_count"] == 0
    assert dataset["prefix_length"].tolist() == [4, 4]
    assert dataset["target"].tolist() == [1, 0]


def test_provenance_is_captured_before_common_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_path = tmp_path / "data/raw/traces.csv"
    output_directory = tmp_path / "data/processed"
    manifest_path = tmp_path / "results/manifest.json"
    raw_path.parent.mkdir(parents=True)
    states = "|".join(str(state) for state in range(53))
    traces = pd.DataFrame(
        {
            "trace_id": list(range(10)),
            "state_ids": [states] * 10,
            "terminal_label": ["success", "target"] * 5,
            "reached_target": [0, 1] * 5,
            "number_of_transitions": [52] * 10,
        }
    )
    traces.to_csv(raw_path, index=False)

    def clean_git_output(*arguments: str) -> str:
        assert not output_directory.exists()
        assert not manifest_path.exists()
        responses = {
            ("branch", "--show-current"): "test-branch",
            ("rev-parse", "HEAD"): "clean-source-sha",
            SOURCE_STATUS_ARGUMENTS: "",
        }
        return responses[arguments]

    monkeypatch.setattr(
        baseline_training,
        "git_output",
        clean_git_output,
    )
    monkeypatch.setattr(
        baseline_training,
        "PROJECT_ROOT",
        tmp_path,
    )
    manifest = cohort_generator.generate_common_cohort_datasets(
        raw_dataset_path=raw_path,
        output_directory=output_directory,
        manifest_path=manifest_path,
        minimum_transitions=50,
    )

    assert manifest_path.is_file()
    assert manifest["source_working_tree_dirty"] is False
    assert manifest["source_git_commit_sha"] == "clean-source-sha"
    assert manifest["source_git_branch"] == "test-branch"
    assert manifest["provenance_capture_timestamp"]
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["source_git_commit_sha"] == "clean-source-sha"
    assert not Path(persisted["raw_dataset_path"]).is_absolute()
    assert all(
        not Path(window["output_path"]).is_absolute()
        for window in persisted["windows"]
    )
    assert all(
        (tmp_path / window["output_path"]).is_file()
        for window in persisted["windows"]
    )


def initialise_test_repository(repository: Path) -> str:
    """Create a minimal repository and return its initial commit SHA."""

    subprocess.run(
        ["git", "init", "--initial-branch=test-branch"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repository,
        check=True,
    )
    tracked_source = repository / "src/tracked.py"
    tracked_source.parent.mkdir(parents=True)
    tracked_source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", "src/tracked.py"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial source"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_generated_systematic_outputs_do_not_make_source_dirty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected_sha = initialise_test_repository(tmp_path)
    generated_output = (
        tmp_path
        / "results/systematic/brp_stress_error/metrics/result.json"
    )
    generated_output.parent.mkdir(parents=True)
    generated_output.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(baseline_training, "PROJECT_ROOT", tmp_path)

    provenance = baseline_training.capture_source_provenance(
        "2026-08-03T00:00:00+00:00"
    )

    assert provenance == {
        "source_git_commit_sha": expected_sha,
        "source_git_branch": "test-branch",
        "source_working_tree_dirty": False,
        "provenance_capture_timestamp": "2026-08-03T00:00:00+00:00",
    }


def test_unrelated_untracked_source_file_makes_source_dirty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    initialise_test_repository(tmp_path)
    (tmp_path / "src/untracked.py").write_text(
        "VALUE = 2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline_training, "PROJECT_ROOT", tmp_path)

    provenance = baseline_training.capture_source_provenance()

    assert provenance["source_working_tree_dirty"] is True


def test_modified_tracked_source_file_makes_source_dirty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected_sha = initialise_test_repository(tmp_path)
    (tmp_path / "src/tracked.py").write_text(
        "VALUE = 2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline_training, "PROJECT_ROOT", tmp_path)

    provenance = baseline_training.capture_source_provenance()

    assert provenance["source_git_commit_sha"] == expected_sha
    assert provenance["source_git_branch"] == "test-branch"
    assert provenance["provenance_capture_timestamp"]
    assert provenance["source_working_tree_dirty"] is True
