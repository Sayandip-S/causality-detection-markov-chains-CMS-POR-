from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.train_brp_baselines import (
    repository_relative_path,
    reproducibility_metadata,
)
from src.storm.model_utils import load_prism_model


REQUIRED_CANDIDATE_COLUMNS = {
    "rank",
    "state_id",
    "combined_ranking_score",
    "logistic_regression_coefficient",
    "random_forest_feature_importance",
    "visited_trace_count",
    "target_probability_when_visited",
    "target_probability_when_not_visited",
    "probability_difference",
}

OUTPUT_CANDIDATE_COLUMNS = [
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

CANDIDATE_COLUMN_MAPPING = {
    "rank": "rank",
    "state_id": "state_id",
    "combined_ranking_score": "combined_score",
    "logistic_regression_coefficient": "logistic_coefficient",
    "random_forest_feature_importance": "random_forest_importance",
    "visited_trace_count": "visited_trace_count",
    "target_probability_when_visited": "p_target_given_visited",
    "target_probability_when_not_visited": "p_target_given_not_visited",
    "probability_difference": "probability_difference",
}


def load_candidates(path: Path, top_k: int) -> pd.DataFrame:
    """Load, validate, and select the top-ranked candidate states."""

    if not path.is_file():
        raise FileNotFoundError(f"Candidate CSV not found: {path}")
    if top_k <= 0:
        raise ValueError("top-k must be a positive integer.")

    candidates = pd.read_csv(path)
    missing_columns = sorted(
        REQUIRED_CANDIDATE_COLUMNS - set(candidates.columns)
    )
    if missing_columns:
        raise ValueError(
            "Candidate CSV is missing required columns: "
            f"{missing_columns}"
        )
    if candidates.empty:
        raise ValueError("Candidate CSV contains no candidate states.")
    if len(candidates) < top_k:
        raise ValueError(
            f"Requested top-k={top_k}, but the candidate CSV contains only "
            f"{len(candidates)} rows."
        )

    for column in ("rank", "state_id"):
        numeric_values = pd.to_numeric(candidates[column], errors="coerce")
        invalid = numeric_values.isna() | (numeric_values % 1 != 0)
        if invalid.any():
            rows = (candidates.index[invalid] + 2).tolist()
            raise ValueError(
                f"Candidate column {column!r} must contain integers; "
                f"invalid CSV rows: {rows}"
            )
        candidates[column] = numeric_values.astype(int)

    if (candidates["rank"] <= 0).any():
        raise ValueError("Candidate ranks must be positive integers.")
    if candidates["rank"].duplicated().any():
        raise ValueError("Candidate ranks must be unique.")
    if candidates["state_id"].duplicated().any():
        raise ValueError("Candidate state IDs must be unique.")

    return candidates.sort_values("rank").head(top_k).copy()


def get_prism_variables(program: Any) -> list[Any]:
    """Return all state variables in deterministic name order."""

    variables = [
        *program.global_boolean_variables,
        *program.global_integer_variables,
    ]
    for module in program.modules:
        variables.extend(module.boolean_variables)
        variables.extend(module.integer_variables)

    names = [variable.name for variable in variables]
    if len(names) != len(set(names)):
        raise ValueError(
            "PRISM program contains duplicate state-variable names, so a "
            "deterministic flat CSV schema cannot be created."
        )
    return sorted(variables, key=lambda variable: variable.name)


def python_scalar(value: Any) -> bool | int | float | str:
    """Convert one Storm valuation into a CSV-safe Python scalar."""

    if isinstance(value, (bool, int, float, str)):
        return value
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return str(value)


def validate_state_ids(candidates: pd.DataFrame, model: Any) -> None:
    """Verify that all candidate IDs index states in this sparse model."""

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


def map_candidates(
    candidates: pd.DataFrame,
    program: Any,
    model: Any,
) -> pd.DataFrame:
    """Attach structured PRISM valuations and labels to candidate rows."""

    validate_state_ids(candidates, model)
    prism_variables = get_prism_variables(program)
    valuation_columns = [variable.name for variable in prism_variables]
    rows: list[dict[str, Any]] = []

    for _, candidate in candidates.iterrows():
        state_id = int(candidate["state_id"])
        row = {
            output_name: candidate[input_name]
            for input_name, output_name in CANDIDATE_COLUMN_MAPPING.items()
        }
        row["rank"] = int(row["rank"])
        row["state_id"] = state_id
        row["visited_trace_count"] = int(row["visited_trace_count"])

        for variable in prism_variables:
            value = model.state_valuations.get_value(
                state_id,
                variable.expression_variable,
            )
            row[variable.name] = python_scalar(value)

        row["labels"] = "|".join(sorted(model.states[state_id].labels))
        rows.append(row)

    output_columns = [
        *OUTPUT_CANDIDATE_COLUMNS,
        *valuation_columns,
        "labels",
    ]
    return pd.DataFrame(rows, columns=output_columns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Map ranked BRP Storm state IDs to their PRISM valuations."
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

    all_candidates = pd.read_csv(candidate_path)
    selected_candidates = load_candidates(candidate_path, args.top_k)
    program, _, model = load_prism_model(model_path, property_path)

    # Validate the complete candidate file, not only the displayed top-k.
    if "state_id" not in all_candidates.columns:
        raise ValueError("Candidate CSV is missing required column: state_id")
    all_state_ids = pd.to_numeric(
        all_candidates["state_id"], errors="coerce"
    )
    if all_state_ids.isna().any() or (all_state_ids % 1 != 0).any():
        raise ValueError("Every candidate state_id must be an integer.")
    validate_state_ids(
        pd.DataFrame({"state_id": all_state_ids.astype(int)}),
        model,
    )

    mapped = map_candidates(selected_candidates, program, model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(output_path, index=False)

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
        "selected_top_k": len(mapped),
        "interpretation": (
            "Each row maps a ranked Storm sparse-model state ID to the full "
            "PRISM state-variable valuation and labels attached to that state."
        ),
        "limitations": [
            "Storm state IDs are internal indices for this exact parsed model, "
            "property set, builder configuration, and Storm version; they are "
            "not PRISM variable values and may change under another build.",
            "The mapping is descriptive and does not establish that a state "
            "causes the target outcome.",
            "Candidate scores and empirical probabilities retain the exploratory "
            "limitations documented by the candidate-extraction experiment.",
            "Labels report only labels attached directly to the mapped state.",
        ],
        **reproducibility_metadata(
            datetime.now(timezone.utc).isoformat()
        ),
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Mapped top {len(mapped)} candidate states:")
    print(mapped.head(20).to_string(index=False))
    print(f"\nMapped candidate CSV written to: {output_path}")
    print(f"Mapping metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()
