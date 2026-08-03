from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.storm.model_utils import PROJECT_ROOT


DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "brp_traces_10000.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "brp_visited_state_dataset.csv"
)


def parse_state_ids(
    state_id_text: object,
) -> list[int]:
    """
    Convert a pipe-separated state sequence into a list of integers.

    Example
    -------
    "0|12|25" -> [0, 12, 25]
    """

    if not isinstance(state_id_text, str) or not state_id_text.strip():
        raise ValueError("state_ids is empty or missing.")

    values = state_id_text.split("|")

    if any(not value.strip() for value in values):
        raise ValueError(
            "state_ids contains an empty state ID."
        )

    try:
        return [int(value) for value in values]
    except ValueError as error:
        raise ValueError(
            "state_ids must be a pipe-separated sequence of integers."
        ) from error


def get_fractional_prefix(
    states: list[int],
    prefix_fraction: float,
) -> list[int]:
    """Return the requested fractional prefix of one full trace."""

    if not states:
        raise ValueError(
            "Trace contains no states."
        )

    prefix_length = max(
        1,
        int(len(states) * prefix_fraction),
    )

    return states[:prefix_length]


def positive_integer(value: str) -> int:
    """Parse a strictly positive prefix length for argparse."""

    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "prefix-length must be a positive integer."
        ) from error

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            "prefix-length must be a positive integer."
        )

    return parsed_value


def positive_cohort_threshold(value: str) -> int:
    """Parse a strictly positive common-cohort threshold."""

    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "cohort-min-transitions must be a positive integer."
        ) from error

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            "cohort-min-transitions must be a positive integer."
        )

    return parsed_value


def parse_transition_count(
    value: object,
) -> int:
    """Parse a non-negative integer transition count from a CSV cell."""

    if pd.isna(value):
        raise ValueError("number_of_transitions is empty or missing.")

    try:
        transition_count = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "number_of_transitions must be a non-negative integer."
        ) from error

    if transition_count < 0 or float(value) != transition_count:
        raise ValueError(
            "number_of_transitions must be a non-negative integer."
        )

    return transition_count


def select_trace_cohort(
    traces: pd.DataFrame,
    minimum_transitions: int,
) -> pd.DataFrame:
    """Select traces strictly longer than a common transition threshold."""

    if minimum_transitions <= 0:
        raise ValueError("minimum_transitions must be a positive integer.")
    if "number_of_transitions" not in traces.columns:
        raise ValueError(
            "Trace dataset must contain number_of_transitions."
        )

    transition_counts = pd.to_numeric(
        traces["number_of_transitions"],
        errors="coerce",
    )
    invalid = (
        transition_counts.isna()
        | (transition_counts < 0)
        | (transition_counts % 1 != 0)
    )
    if invalid.any():
        csv_rows = (traces.index[invalid] + 2).tolist()
        raise ValueError(
            "number_of_transitions must contain non-negative integers; "
            f"invalid CSV rows: {csv_rows}"
        )

    return traces.loc[
        transition_counts > minimum_transitions
    ].copy()


def build_visited_state_dataset(
    traces: pd.DataFrame,
    *,
    prefix_fraction: float | None = None,
    prefix_length: int | None = None,
    include_trace_id: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build one visited-state prefix dataset without writing it."""

    if (prefix_fraction is None) == (prefix_length is None):
        raise ValueError(
            "Specify exactly one of prefix_fraction or prefix_length."
        )
    if prefix_fraction is not None and not 0.0 < prefix_fraction <= 1.0:
        raise ValueError("prefix_fraction must be in (0, 1].")
    if prefix_length is not None and prefix_length <= 0:
        raise ValueError("prefix_length must be a positive integer.")
    if include_trace_id and "trace_id" not in traces.columns:
        raise ValueError(
            "Trace dataset must contain trace_id when it is retained."
        )

    all_prefixes: list[list[int]] = []
    final_states: list[int] = []
    labels: list[int] = []
    trace_ids: list[object] = []
    all_states: set[int] = set()
    excluded_target_traces = 0
    excluded_success_traces = 0

    for row_index, row in traces.iterrows():
        try:
            states = parse_state_ids(row["state_ids"])
            transition_count = parse_transition_count(
                row["number_of_transitions"]
            )
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Invalid trace at CSV row {row_index + 2}: {error}"
            ) from error

        parsed_transition_count = len(states) - 1

        if transition_count != parsed_transition_count:
            raise ValueError(
                "Invalid trace at CSV row "
                f"{row_index + 2}: number_of_transitions "
                f"is {transition_count}, but state_ids contains "
                f"{parsed_transition_count} transitions."
            )

        terminal_label = row["terminal_label"]

        # Preserve the existing behavior of using known outcomes only.
        if terminal_label not in {"target", "success"}:
            continue

        if (
            prefix_length is not None
            and transition_count <= prefix_length
        ):
            if terminal_label == "target":
                excluded_target_traces += 1
            else:
                excluded_success_traces += 1
            continue

        if prefix_length is not None:
            prefix = states[:prefix_length + 1]
        else:
            prefix = get_fractional_prefix(
                states=states,
                prefix_fraction=prefix_fraction,
            )

        all_prefixes.append(prefix)
        final_states.append(states[-1])
        labels.append(int(row["reached_target"]))
        if include_trace_id:
            trace_ids.append(row["trace_id"])
        all_states.update(prefix)

    if not all_prefixes:
        raise ValueError(
            "No traces were retained for the requested observation window."
        )

    terminal_state_prefix_count = sum(
        final_state in prefix
        for prefix, final_state in zip(
            all_prefixes,
            final_states,
        )
    )

    if (
        prefix_length is not None
        and terminal_state_prefix_count != 0
    ):
        raise ValueError(
            "Fixed-prefix terminal-state leakage detected: "
            f"{terminal_state_prefix_count} retained prefixes contain "
            "the final state of their full trace."
        )

    sorted_states = sorted(all_states)
    rows = []

    for position, (prefix, label) in enumerate(
        zip(all_prefixes, labels)
    ):
        visited_states = set(prefix)
        row = {}
        if include_trace_id:
            row["trace_id"] = trace_ids[position]
        row.update(
            {
                f"visited_state_{state_id}": int(
                    state_id in visited_states
                )
                for state_id in sorted_states
            }
        )
        row["prefix_length"] = len(prefix)
        row["last_state"] = prefix[-1]
        row["target"] = label
        rows.append(row)

    dataset = pd.DataFrame(rows)
    summary = {
        "input_trace_count": len(traces),
        "retained_trace_count": len(dataset),
        "excluded_target_count": excluded_target_traces,
        "excluded_success_count": excluded_success_traces,
        "terminal_leakage_count": terminal_state_prefix_count,
        "feature_count": len(sorted_states),
    }
    return dataset, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a visited-state prefix dataset from "
            "BRP execution traces."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input trace CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output dataset CSV.",
    )
    prefix_group = parser.add_mutually_exclusive_group(
        required=True
    )
    prefix_group.add_argument(
        "--prefix-fraction",
        type=float,
        help=(
            "Fraction of each completed trace used as the "
            "observed prefix."
        ),
    )
    prefix_group.add_argument(
        "--prefix-length",
        type=positive_integer,
        metavar="K",
        help=(
            "Fixed number of transitions observed from the "
            "beginning of each trace."
        ),
    )
    parser.add_argument(
        "--cohort-min-transitions",
        type=positive_cohort_threshold,
        help=(
            "Optionally retain one common cohort with strictly more than "
            "this many transitions before constructing the prefix."
        ),
    )
    args = parser.parse_args()

    raw_traces = pd.read_csv(args.input)
    total_input_traces = len(raw_traces)
    cohort_excluded = raw_traces.iloc[0:0]
    selected_traces = raw_traces

    if args.cohort_min_transitions is not None:
        selected_traces = select_trace_cohort(
            raw_traces,
            args.cohort_min_transitions,
        )
        cohort_excluded = raw_traces.loc[
            ~raw_traces.index.isin(selected_traces.index)
        ]

    dataset, summary = build_visited_state_dataset(
        selected_traces,
        prefix_fraction=args.prefix_fraction,
        prefix_length=args.prefix_length,
        include_trace_id=args.cohort_min_transitions is not None,
    )
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dataset.to_csv(
        args.output,
        index=False,
    )

    cohort_excluded_target = int(
        (cohort_excluded["terminal_label"] == "target").sum()
    )
    cohort_excluded_success = int(
        (cohort_excluded["terminal_label"] == "success").sum()
    )
    excluded_target_traces = (
        cohort_excluded_target
        + summary["excluded_target_count"]
    )
    excluded_success_traces = (
        cohort_excluded_success
        + summary["excluded_success_count"]
    )
    retained_target_traces = int(dataset["target"].sum())
    retained_success_traces = len(dataset) - retained_target_traces
    excluded_traces = total_input_traces - len(dataset)

    print("Visited-state prefix dataset created.")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    if args.cohort_min_transitions is not None:
        print(
            "Cohort rule: number_of_transitions > "
            f"{args.cohort_min_transitions}"
        )
    if args.prefix_length is not None:
        print("Observation mode: fixed transition length")
        print(f"Fixed transition length: {args.prefix_length}")
    else:
        print("Observation mode: fractional prefix")
        print(f"Prefix fraction: {args.prefix_fraction}")

    print(f"Total input traces: {total_input_traces}")
    print(f"Retained traces: {len(dataset)}")
    print(f"Excluded traces: {excluded_traces}")
    print(f"Excluded target traces: {excluded_target_traces}")
    print(f"Excluded success traces: {excluded_success_traces}")
    print(f"Retained target traces: {retained_target_traces}")
    print(f"Retained success traces: {retained_success_traces}")
    print(f"Retained positive rate: {dataset['target'].mean():.6f}")
    print(
        "Number of visited-state feature columns: "
        f"{summary['feature_count']}"
    )
    print(
        "Minimum observed prefix length: "
        f"{dataset['prefix_length'].min()}"
    )
    print(
        "Maximum observed prefix length: "
        f"{dataset['prefix_length'].max()}"
    )
    print(
        "Retained prefixes containing the final state: "
        f"{summary['terminal_leakage_count']}"
    )


if __name__ == "__main__":
    main()
