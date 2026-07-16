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
    """
    Return the observed prefix of a full trace.
    """

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
    """Parse a strictly positive integer for argparse."""

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

    args = parser.parse_args()

    if (
        args.prefix_fraction is not None
        and not 0.0 < args.prefix_fraction <= 1.0
    ):
        raise ValueError(
            "prefix-fraction must be in (0, 1]."
        )

    traces = pd.read_csv(args.input)
    total_input_traces = len(traces)

    all_prefixes: list[list[int]] = []
    final_states: list[int] = []
    labels: list[int] = []
    all_states: set[int] = set()
    excluded_target_traces = 0
    excluded_success_traces = 0

    for row_index, row in traces.iterrows():
        try:
            states = parse_state_ids(row["state_ids"])
            transition_count = parse_transition_count(
                row["number_of_transitions"]
            )
        except ValueError as error:
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
            args.prefix_length is not None
            and transition_count <= args.prefix_length
        ):
            if terminal_label == "target":
                excluded_target_traces += 1
            else:
                excluded_success_traces += 1
            continue

        if args.prefix_length is not None:
            prefix = states[:args.prefix_length + 1]
        else:
            prefix = get_fractional_prefix(
                states=states,
                prefix_fraction=args.prefix_fraction,
            )

        all_prefixes.append(prefix)
        final_states.append(states[-1])
        labels.append(
            int(row["reached_target"])
        )

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
        args.prefix_length is not None
        and terminal_state_prefix_count != 0
    ):
        raise ValueError(
            "Fixed-prefix terminal-state leakage detected: "
            f"{terminal_state_prefix_count} retained prefixes contain "
            "the final state of their full trace."
        )

    sorted_states = sorted(all_states)

    rows = []

    for prefix, label in zip(
        all_prefixes,
        labels,
    ):
        visited_states = set(prefix)

        row = {
            f"visited_state_{state_id}": (
                1
                if state_id in visited_states
                else 0
            )
            for state_id in sorted_states
        }

        row["prefix_length"] = len(prefix)
        row["last_state"] = prefix[-1]
        row["target"] = label

        rows.append(row)

    dataset = pd.DataFrame(rows)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        args.output,
        index=False,
    )

    print("Visited-state prefix dataset created.")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")

    if args.prefix_length is not None:
        print("Observation mode: fixed transition length")
        print(f"Fixed transition length: {args.prefix_length}")
    else:
        print("Observation mode: fractional prefix")
        print(f"Prefix fraction: {args.prefix_fraction}")

    retained_target_traces = int(dataset["target"].sum())
    retained_success_traces = len(dataset) - retained_target_traces
    excluded_traces = total_input_traces - len(dataset)

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
        f"{len(sorted_states)}"
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
        f"{terminal_state_prefix_count}"
    )


if __name__ == "__main__":
    main()
