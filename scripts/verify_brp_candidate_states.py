from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import stormpy

from scripts.map_brp_candidate_states import (
    OUTPUT_CANDIDATE_COLUMNS,
    load_candidates,
    map_candidates,
)
from src.ml.train_brp_baselines import (
    repository_relative_path,
    reproducibility_metadata,
)
from src.storm.model_utils import load_prism_model


REQUIRED_RANKING_COLUMNS = {
    "support_fraction",
    "combined_ranking_score",
}

OUTPUT_COLUMNS = [
    "original_rank",
    "state_id",
    "empirical_support_fraction",
    "exact_candidate_reachability",
    "support_reachability_absolute_gap",
    "support_reachability_relative_gap",
    "baseline_target_probability",
    "target_probability_from_candidate",
    "probability_difference_from_baseline",
    "probability_ratio_to_baseline",
    "raises_probability_from_state",
    "combined_ranking_score",
    "risk_weighted_coverage",
]


def load_candidate_experiment_context(
    candidate_path: Path,
    candidates: pd.DataFrame,
) -> tuple[int, int]:
    """Return and validate the candidate experiment's window and population."""

    metadata_path = candidate_path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Candidate metadata JSON not found: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing_fields = [
        field
        for field in ("observation_window", "retained_row_count")
        if field not in metadata
    ]
    if missing_fields:
        raise ValueError(
            "Candidate metadata is missing required empirical-context fields: "
            f"{missing_fields}"
        )

    observation_window = metadata["observation_window"]
    empirical_population_rows = metadata["retained_row_count"]
    if (
        not isinstance(observation_window, int)
        or isinstance(observation_window, bool)
        or observation_window <= 0
    ):
        raise ValueError(
            "Candidate metadata observation_window must be a positive integer."
        )
    if (
        not isinstance(empirical_population_rows, int)
        or isinstance(empirical_population_rows, bool)
        or empirical_population_rows <= 0
    ):
        raise ValueError(
            "Candidate metadata retained_row_count must be a positive integer."
        )

    if "total_retained_trace_count" not in candidates.columns:
        raise ValueError(
            "Candidate CSV is missing total_retained_trace_count."
        )
    candidate_population = pd.to_numeric(
        candidates["total_retained_trace_count"],
        errors="coerce",
    )
    invalid_population = (
        candidate_population.isna()
        | (candidate_population % 1 != 0)
        | (candidate_population <= 0)
    )
    if invalid_population.any():
        csv_rows = (candidates.index[invalid_population] + 2).tolist()
        raise ValueError(
            "Candidate total_retained_trace_count must contain positive "
            f"integers; invalid CSV rows: {csv_rows}"
        )
    if set(candidate_population.astype(int)) != {empirical_population_rows}:
        raise ValueError(
            "Candidate CSV trace counts do not match candidate metadata "
            f"retained_row_count={empirical_population_rows}."
        )

    return observation_window, empirical_population_rows


def unique_initial_state(model: Any) -> int:
    """Return the model's sole initial state or raise a clear error."""

    initial_states = [int(state_id) for state_id in model.initial_states]
    if len(initial_states) != 1:
        raise ValueError(
            "Exact candidate verification requires one unique initial state, "
            f"but the constructed model has {len(initial_states)}: "
            f"{initial_states}"
        )
    return initial_states[0]


def validate_ranking_columns(candidates: pd.DataFrame) -> None:
    """Validate empirical support and combined-ranking inputs."""

    missing_columns = sorted(
        REQUIRED_RANKING_COLUMNS - set(candidates.columns)
    )
    if missing_columns:
        raise ValueError(
            "Candidate CSV is missing required reachability columns: "
            f"{missing_columns}"
        )

    for column in sorted(REQUIRED_RANKING_COLUMNS):
        values = pd.to_numeric(candidates[column], errors="coerce")
        if values.isna().any():
            csv_rows = (candidates.index[values.isna()] + 2).tolist()
            raise ValueError(
                f"Candidate column {column!r} must be numeric; "
                f"invalid CSV rows: {csv_rows}"
            )
        candidates[column] = values.astype(float)

    invalid_support = (
        (candidates["support_fraction"] < 0.0)
        | (candidates["support_fraction"] > 1.0)
    )
    if invalid_support.any():
        csv_rows = (candidates.index[invalid_support] + 2).tolist()
        raise ValueError(
            "Candidate support_fraction must be between zero and one; "
            f"invalid CSV rows: {csv_rows}"
        )


def candidate_reachabilities(
    model: Any,
    initial_state_id: int,
    state_ids: pd.Series,
) -> list[float]:
    """Model-check exact initial-to-candidate reachability probabilities."""

    probabilities: list[float] = []
    for state_id_value in state_ids:
        state_id = int(state_id_value)
        label = f"exact_reachability_candidate_{state_id}"
        if model.labeling.contains_label(label):
            raise ValueError(f"Temporary candidate label already exists: {label}")

        # A fresh label denotes exactly one sparse-model state. This avoids
        # rebuilding the model for every candidate and is independent of any
        # coincident labels attached to that state in the PRISM model.
        model.labeling.add_label(label)
        model.labeling.add_label_to_state(label, state_id)
        candidate_property = stormpy.parse_properties(
            f'P=? [ F "{label}" ]'
        )[0]
        result = stormpy.model_checking(model, candidate_property)
        probabilities.append(float(result.at(initial_state_id)))

    return probabilities


def verify_candidates(
    candidates: pd.DataFrame,
    mapped_candidates: pd.DataFrame,
    model: Any,
    target_result: Any,
    initial_state_id: int,
) -> pd.DataFrame:
    """Calculate both requested exact probabilities for every candidate."""

    validate_ranking_columns(candidates)
    if list(candidates["state_id"]) != list(mapped_candidates["state_id"]):
        raise ValueError(
            "Mapped candidate ordering does not match the ranked candidates."
        )

    baseline_probability = float(target_result.at(initial_state_id))
    exact_reachability = candidate_reachabilities(
        model,
        initial_state_id,
        candidates["state_id"],
    )
    target_from_candidate = [
        float(target_result.at(int(state_id)))
        for state_id in candidates["state_id"]
    ]

    verified = pd.DataFrame(
        {
            "original_rank": candidates["rank"].astype(int),
            "state_id": candidates["state_id"].astype(int),
            "empirical_support_fraction": candidates[
                "support_fraction"
            ].astype(float),
            "exact_candidate_reachability": exact_reachability,
            "baseline_target_probability": baseline_probability,
            "target_probability_from_candidate": target_from_candidate,
            "combined_ranking_score": candidates[
                "combined_ranking_score"
            ].astype(float),
        }
    )
    verified["support_reachability_absolute_gap"] = (
        verified["empirical_support_fraction"]
        - verified["exact_candidate_reachability"]
    ).abs()
    verified["support_reachability_relative_gap"] = (
        verified["support_reachability_absolute_gap"]
        / verified["exact_candidate_reachability"].where(
            verified["exact_candidate_reachability"] != 0.0
        )
    )
    verified["probability_difference_from_baseline"] = (
        verified["target_probability_from_candidate"]
        - baseline_probability
    )
    if baseline_probability == 0.0:
        verified["probability_ratio_to_baseline"] = float("nan")
    else:
        verified["probability_ratio_to_baseline"] = (
            verified["target_probability_from_candidate"]
            / baseline_probability
        )
    verified["raises_probability_from_state"] = (
        verified["target_probability_from_candidate"]
        > baseline_probability
    )
    verified["risk_weighted_coverage"] = (
        verified["exact_candidate_reachability"]
        * verified["target_probability_from_candidate"]
    )

    valuation_columns = [
        column
        for column in mapped_candidates.columns
        if column not in OUTPUT_CANDIDATE_COLUMNS
        and column != "labels"
    ]
    for column in valuation_columns:
        verified[column] = mapped_candidates[column].to_numpy()
    verified["labels"] = mapped_candidates["labels"].to_numpy()

    return verified[[*OUTPUT_COLUMNS, *valuation_columns, "labels"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exactly verify initial-to-candidate and candidate-to-target "
            "reachability for ranked BRP states."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)
    started_timer = time.perf_counter()
    model_path = args.model.resolve()
    property_path = args.property.resolve()
    candidate_path = args.candidates.resolve()
    output_path = args.output.resolve()

    candidates = load_candidates(candidate_path, args.top_k)
    validate_ranking_columns(candidates)
    observation_window, empirical_population_rows = (
        load_candidate_experiment_context(candidate_path, candidates)
    )
    program, properties, model = load_prism_model(
        model_path,
        property_path,
    )
    if len(properties) != 1:
        raise ValueError(
            "Expected exactly one target property, but parsed "
            f"{len(properties)} properties from {property_path}."
        )

    initial_state_id = unique_initial_state(model)
    target_result = stormpy.model_checking(model, properties[0])
    if not target_result.result_for_all_states:
        raise RuntimeError(
            "Storm did not return target probabilities for all model states."
        )
    if len(target_result.get_values()) != model.nr_states:
        raise RuntimeError(
            "Storm returned a probability vector whose size does not match "
            f"the model state count ({len(target_result.get_values())} versus "
            f"{model.nr_states})."
        )

    mapped_candidates = map_candidates(candidates, program, model)
    verified = verify_candidates(
        candidates=candidates,
        mapped_candidates=mapped_candidates,
        model=model,
        target_result=target_result,
        initial_state_id=initial_state_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verified.to_csv(output_path, index=False)

    completed_at = datetime.now(timezone.utc)
    execution_time_seconds = time.perf_counter() - started_timer
    baseline_probability = float(target_result.at(initial_state_id))
    metadata = {
        "model_path": repository_relative_path(model_path, "Model path"),
        "property_path": repository_relative_path(
            property_path,
            "Property path",
        ),
        "candidate_path": repository_relative_path(
            candidate_path,
            "Candidate path",
        ),
        "model_state_count": int(model.nr_states),
        "transition_count": int(model.nr_transitions),
        "initial_state_id": initial_state_id,
        "target_baseline": baseline_probability,
        "stormpy_version": getattr(stormpy, "__version__", None),
        "selected_top_k": len(verified),
        "observation_window_transitions": observation_window,
        "empirical_population_rows": empirical_population_rows,
        "empirical_population_condition": (
            f"number_of_transitions > {observation_window}"
        ),
        "empirical_observation_horizon": (
            f"first {observation_window} transitions"
        ),
        "exact_reachability_horizon": "unbounded",
        "quantities_directly_comparable": False,
        "calculation_definitions": {
            "empirical_support_fraction": (
                f"Fraction of retained k{observation_window} traces in which "
                "the candidate appears within the initial state plus first "
                f"{observation_window} transitions. The retained population "
                "contains only traces with more than "
                f"{observation_window} transitions."
            ),
            "exact_candidate_reachability": (
                "Unbounded, unconditional Storm probability "
                "P_initial(F candidate)."
            ),
            "support_reachability_absolute_gap": (
                "Absolute descriptive difference between the bounded, "
                "survival-conditioned empirical support and unbounded, "
                "unconditional exact reachability."
            ),
            "support_reachability_relative_gap": (
                "The absolute descriptive gap divided by exact candidate "
                "reachability, when the denominator is nonzero."
            ),
            "target_probability_from_candidate": (
                "P_candidate(F target): Storm model-checking probability of "
                "eventually reaching the target from the candidate state."
            ),
            "probability_difference_from_baseline": (
                "target_probability_from_candidate - target_baseline."
            ),
            "probability_ratio_to_baseline": (
                "target_probability_from_candidate / target_baseline; "
                "undefined when the baseline is zero."
            ),
            "raises_probability_from_state": (
                "Whether target_probability_from_candidate is strictly "
                "greater than target_baseline."
            ),
            "risk_weighted_coverage": (
                "exact_candidate_reachability * "
                "target_probability_from_candidate. This is a descriptive "
                "heuristic, not a formal causality measure."
            ),
        },
        "execution": {
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": execution_time_seconds,
        },
        "methodological_limitations": [
            "The empirical and exact quantities correspond to different "
            "events: empirical support is bounded and survival-conditioned, "
            "whereas exact candidate reachability is unbounded and "
            "unconditional.",
            "The support-reachability gap combines sampling variation, finite "
            "observation horizon, and population conditioning.",
            "The support-reachability gap must not be interpreted as pure "
            "Monte Carlo estimation error.",
            "A matched comparison requires either full-trace empirical "
            "visitation over all raw traces versus unbounded exact "
            "reachability, or exact bounded and survival-conditioned "
            "reachability matching the k20 dataset.",
            "risk_weighted_coverage is a descriptive heuristic and is not a "
            "formal causality measure.",
            "P_candidate(F target) is state-based and is not the historical "
            "conditional probability P(target | candidate was visited).",
            "Empirical support comes from finite retained traces, while exact "
            "reachability comes from the specified PRISM model; their "
            "difference mixes sampling and model-data mismatch.",
            "Probability raising relative to the initial-state baseline does "
            "not by itself establish causality.",
            "Storm sparse-model state IDs depend on this model, property set, "
            "builder configuration, and Storm version.",
        ],
        **reproducibility_metadata(completed_at.isoformat()),
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    display = verified[
        [
            "original_rank",
            "state_id",
            "empirical_support_fraction",
            "exact_candidate_reachability",
            "support_reachability_absolute_gap",
            "target_probability_from_candidate",
            "probability_difference_from_baseline",
        ]
    ].head(20)
    display = display.rename(
        columns={
            "original_rank": "rank",
            "empirical_support_fraction": "empirical_support",
            "support_reachability_absolute_gap": (
                "support_reachability_gap"
            ),
            "target_probability_from_candidate": (
                "exact_target_probability_from_candidate"
            ),
            "probability_difference_from_baseline": (
                "exact_target_probability_increase"
            ),
        }
    )
    print(f"Initial state ID: {initial_state_id}")
    print(f"Target baseline: {baseline_probability:.12f}")
    print(f"\nExact reachability for top {len(verified)} candidates:")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.9f}"))
    print(f"\nReachability CSV written to: {output_path}")
    print(f"Reachability metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()
