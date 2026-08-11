from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_brp_candidate_states import rank_candidate_states
from scripts.generate_brp_common_cohort_datasets import ordered_values_sha256
from scripts.run_brp_common_cohort_baselines import create_shared_trace_split
from scripts.run_brp_sample_size_experiment import (
    adaptive_minimum_support,
    ensure_outputs_absent,
    exact_cache_from_existing,
    experiment_conditions,
    load_config,
    ranking_stability,
    reconstruct_fixed_split,
    stratified_training_sample,
    verify_missing_exact_states,
)
from src.ml.train_brp_baselines import create_models


def example_dataset(rows: int = 100) -> pd.DataFrame:
    targets = np.asarray(([0] * 7 + [1] * 3) * (rows // 10), dtype=int)
    return pd.DataFrame(
        {
            "trace_id": np.arange(rows),
            "visited_state_0": np.ones(rows, dtype=int),
            "visited_state_1": (np.arange(rows) % 2 == 0).astype(int),
            "visited_state_2": (np.arange(rows) % 3 == 0).astype(int),
            "prefix_length": np.full(rows, 21),
            "last_state": np.arange(rows) % 4,
            "target": targets,
        }
    )


def support_config() -> dict[str, object]:
    return {
        "candidate_support_rule": {
            "minimum_count_floor": 5,
            "training_fraction": 0.005,
        }
    }


def test_fixed_test_set_is_reconstructed_without_overlap() -> None:
    dataset = example_dataset()
    train_ids, test_ids = create_shared_trace_split(dataset)
    summary = {
        "split_trace_id_hashes": {
            "train": ordered_values_sha256(train_ids),
            "test": ordered_values_sha256(test_ids),
        },
        "split_configuration": {"train_rows": 80, "test_rows": 20},
    }
    train, test, _, test_hash = reconstruct_fixed_split(
        dataset, summary, test_size=0.2, split_seed=42
    )

    assert set(train["trace_id"]).isdisjoint(test["trace_id"])
    assert ordered_values_sha256(test["trace_id"]) == test_hash
    assert test["trace_id"].tolist() == test_ids.tolist()


def test_reduced_sampling_is_stratified_and_deterministic() -> None:
    train_pool = example_dataset()
    first = stratified_training_sample(train_pool, 50, 123, 100)
    second = stratified_training_sample(train_pool, 50, 123, 100)
    different = stratified_training_sample(train_pool, 50, 456, 100)

    assert first["trace_id"].tolist() == second["trace_id"].tolist()
    assert first["trace_id"].tolist() != different["trace_id"].tolist()
    assert len(first) == 50
    assert first["target"].sum() == 15


def test_one_sample_hash_can_be_shared_by_all_models() -> None:
    sampled = stratified_training_sample(example_dataset(), 50, 42, 100)
    sample_hash = ordered_values_sha256(sampled["trace_id"])
    per_model_hashes = {
        model_name: ordered_values_sha256(sampled["trace_id"])
        for model_name in create_models(42)
    }

    assert set(per_model_hashes.values()) == {sample_hash}


def test_full_sample_has_one_seedless_condition_and_preserves_pool_order() -> None:
    conditions = experiment_conditions([50, 100], [42, 123], 100)
    full = stratified_training_sample(example_dataset(), 100, None, 100)

    assert conditions == [(50, 42), (50, 123), (100, None)]
    assert full["trace_id"].tolist() == list(range(100))


@pytest.mark.parametrize(
    ("sample_size", "expected"),
    [(500, 5), (1000, 5), (2500, 13), (5000, 25), (7341, 37)],
)
def test_adaptive_minimum_support(sample_size: int, expected: int) -> None:
    assert adaptive_minimum_support(sample_size, support_config()) == expected


def test_ranking_overlap_jaccard_and_displacement() -> None:
    reference = pd.DataFrame({"rank": range(1, 21), "state_id": range(1, 21)})
    current = pd.DataFrame(
        {"rank": range(1, 21), "state_id": [1, 2, *range(21, 39)]}
    )
    metrics = ranking_stability(current, reference)

    assert metrics["top5_overlap_count"] == 2
    assert metrics["top5_overlap_fraction"] == pytest.approx(0.4)
    assert metrics["top5_jaccard_similarity"] == pytest.approx(2 / 8)
    assert metrics["shared_candidate_count"] == 2
    assert metrics["mean_absolute_rank_displacement"] == 0


def test_rank_correlation_edge_cases_do_not_crash() -> None:
    reference = pd.DataFrame({"rank": range(1, 21), "state_id": range(1, 21)})
    no_overlap = pd.DataFrame({"rank": range(1, 21), "state_id": range(21, 41)})
    metrics = ranking_stability(no_overlap, reference)

    assert np.isnan(metrics["spearman_rank_correlation"])
    assert np.isnan(metrics["kendall_rank_correlation"])
    assert metrics["rank_correlation_explanation"].startswith("undefined")


def test_exact_cache_reuse_avoids_model_build(tmp_path: Path) -> None:
    exact_path = tmp_path / "exact.csv"
    pd.DataFrame(
        {
            "state_id": [1],
            "exact_candidate_reachability": [0.2],
            "baseline_target_probability": [0.1],
            "target_probability_from_candidate": [0.3],
            "probability_difference_from_baseline": [0.2],
            "probability_ratio_to_baseline": [3.0],
            "raises_probability_from_state": [True],
        }
    ).to_csv(exact_path, index=False)
    cache = exact_cache_from_existing(exact_path)
    selected, metadata = verify_missing_exact_states(
        cache, [1], tmp_path / "missing.pm", tmp_path / "missing.pctl"
    )

    assert selected["state_id"].tolist() == [1]
    assert metadata == {"missing_state_count_verified": 0, "model_built": False}


def test_candidate_metrics_use_only_supplied_training_rows() -> None:
    training = example_dataset()
    features = ["visited_state_0", "visited_state_1", "visited_state_2"]
    models = create_models(42)
    models["Logistic Regression"].fit(training[features], training["target"])
    models["Random Forest"].fit(training[features], training["target"])
    first = rank_candidate_states(
        training,
        features,
        models["Logistic Regression"],
        models["Random Forest"],
        5,
    )
    unrelated_test = example_dataset().assign(target=lambda frame: 1 - frame.target)
    second = rank_candidate_states(
        training,
        features,
        models["Logistic Regression"],
        models["Random Forest"],
        5,
    )

    pd.testing.assert_frame_equal(first, second)
    assert not training["target"].equals(unrelated_test["target"])


def test_config_rejects_absolute_paths(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = {
        "experiment_name": "test",
        "dataset": "/absolute/dataset.csv",
        "common_cohort_manifest": "manifest.json",
        "common_cohort_summary": "summary.json",
        "existing_exact_reachability": "exact.csv",
        "prism_model": "model.pm",
        "property_file": "property.pctl",
        "training_sample_sizes": [10],
        "sampling_seeds": [42],
        "full_training_size": 10,
        "split_seed": 42,
        "test_size": 0.2,
        "top_k": 2,
        "candidate_support_rule": {},
        "models": [],
        "reliability_thresholds": {},
    }
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="repository-relative"):
        load_config(path)


def test_existing_artifacts_are_never_overwritten(tmp_path: Path) -> None:
    existing = tmp_path / "prediction_per_run.csv"
    existing.write_text("protected\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ensure_outputs_absent(tmp_path)
    assert existing.read_text(encoding="utf-8") == "protected\n"
