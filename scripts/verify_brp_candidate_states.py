from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import stormpy

from src.ml.train_brp_baselines import (
    repository_relative_path,
    reproducibility_metadata,
)
from src.storm.model_utils import load_prism_model


CORE_CANDIDATE_COLUMNS = [
    "rank",
    "state_id",
    "combined_score",
    "logistic_coefficient",
    "random_forest_importance",
    "visited_trace_count",
    "p_target_given_visited",
    "p_target_given_not_visited",
    "probability_difference",
]

EXACT_VERIFICATION_COLUMNS = [
    "baseline_target_probability",
    "target_probability_from_candidate",
    "probability_difference_from_baseline",
    "probability_ratio_to_baseline",
    "raises_probability_from_state",
]

DISPLAY_PRISM_VARIABLES = [
    "s",
    "i",
    "nrtr",
    "r",
    "rrep",
    "k",
    "l",
]


def load_candidates(path: Path, top_k: int) -> pd.DataFrame:
    """Load and validate mapped candidates in original-rank order."""

    if not path.is_file():
        raise FileNotFoundError(f"Mapped candidate CSV not found: {path}")
    if top_k <= 0:
        raise ValueError("top-k must be a positive integer.")

    candidates = pd.read_csv(path)
    required_columns = {*CORE_CANDIDATE_COLUMNS, "labels"}
    missing_columns = sorted(required_columns - set(candidates.columns))
    if missing_columns:
        raise ValueError(
            "Mapped candidate CSV is missing required columns: "
            f"{missing_columns}"
        )
    if candidates.empty:
        raise ValueError("Mapped candidate CSV contains no candidate states.")
    if len(candidates) < top_k:
        raise ValueError(
            f"Requested top-k={top_k}, but the mapped candidate CSV contains "
            f"only {len(candidates)} rows."
        )

    for column in ("rank", "state_id"):
        numeric_values = pd.to_numeric(candidates[column], errors="coerce")
        invalid = numeric_values.isna() | (numeric_values % 1 != 0)
        if invalid.any():
            csv_rows = (candidates.index[invalid] + 2).tolist()
            raise ValueError(
                f"Candidate column {column!r} must contain integers; "
                f"invalid CSV rows: {csv_rows}"
            )
        candidates[column] = numeric_values.astype(int)

    if (candidates["rank"] <= 0).any():
        raise ValueError("Candidate ranks must be positive integers.")
    if candidates["rank"].duplicated().any():
        raise ValueError("Candidate ranks must be unique.")
    if candidates["state_id"].duplicated().any():
        raise ValueError("Candidate state IDs must be unique.")

    return candidates.sort_values("rank").head(top_k).copy()


def validate_state_ids(candidates: pd.DataFrame, model: Any) -> None:
    """Verify that every selected state ID exists in the sparse model."""

    invalid_state_ids = sorted(
        {
            int(state_id)
            for state_id in candidates["state_id"]
            if state_id < 0 or state_id >= model.nr_states
        }
    )
    if invalid_state_ids:
        raise ValueError(
            "Candidate Storm state IDs do not exist in the constructed "
            f"{model.nr_states}-state model: {invalid_state_ids}"
        )


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


def verify_candidates(
    candidates: pd.DataFrame,
    model: Any,
    result: Any,
    baseline_probability: float,
) -> pd.DataFrame:
    """Attach exact target reachability results to mapped candidates."""

    validate_state_ids(candidates, model)
    verified = candidates.copy()
    exact_probabilities = [
        float(result.at(int(state_id)))
        for state_id in verified["state_id"]
    ]
    verified["baseline_target_probability"] = baseline_probability
    verified["target_probability_from_candidate"] = exact_probabilities
    verified["probability_difference_from_baseline"] = (
        verified["target_probability_from_candidate"] - baseline_probability
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

    remaining_input_columns = [
        column
        for column in candidates.columns
        if column not in CORE_CANDIDATE_COLUMNS
    ]
    output_columns = [
        *CORE_CANDIDATE_COLUMNS,
        *EXACT_VERIFICATION_COLUMNS,
        *remaining_input_columns,
    ]
    return verified[output_columns].sort_values(
        ["probability_difference_from_baseline", "rank"],
        ascending=[False, True],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exactly verify target reachability from mapped BRP candidates."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    model_path = args.model.resolve()
    property_path = args.property.resolve()
    candidate_path = args.candidates.resolve()
    output_path = args.output.resolve()

    candidates = load_candidates(candidate_path, args.top_k)
    _, properties, model = load_prism_model(model_path, property_path)
    if len(properties) != 1:
        raise ValueError(
            "Expected exactly one target property, but parsed "
            f"{len(properties)} properties from {property_path}."
        )

    initial_state_id = unique_initial_state(model)
    result = stormpy.model_checking(model, properties[0])
    if not result.result_for_all_states:
        raise RuntimeError(
            "Storm did not return target probabilities for all model states."
        )
    if len(result.get_values()) != model.nr_states:
        raise RuntimeError(
            "Storm returned a probability vector whose size does not match "
            f"the model state count ({len(result.get_values())} versus "
            f"{model.nr_states})."
        )

    baseline_probability = float(result.at(initial_state_id))
    verified = verify_candidates(
        candidates=candidates,
        model=model,
        result=result,
        baseline_probability=baseline_probability,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verified.to_csv(output_path, index=False)

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
        "baseline_target_probability": baseline_probability,
        "stormpy_version": getattr(stormpy, "__version__", None),
        "top_k": len(verified),
        **reproducibility_metadata(
            datetime.now(timezone.utc).isoformat()
        ),
        "interpretation": (
            "This verifies the exact probability of eventually reaching the "
            "target when the system is currently in the candidate state."
        ),
        "limitation": (
            "This is not yet the path-conditioned probability "
            "P(target | candidate was visited earlier)."
        ),
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    display_columns = [
        "rank",
        "state_id",
        *[
            column
            for column in DISPLAY_PRISM_VARIABLES
            if column in verified.columns
        ],
        "probability_difference",
        "target_probability_from_candidate",
        "probability_difference_from_baseline",
        "raises_probability_from_state",
    ]
    print(f"Initial state ID: {initial_state_id}")
    print(f"Baseline target probability: {baseline_probability:.12f}")
    print(f"\nExact verification of top {len(verified)} candidates:")
    print(verified[display_columns].head(20).to_string(index=False))
    print(f"\nVerification CSV written to: {output_path}")
    print(f"Verification metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()
