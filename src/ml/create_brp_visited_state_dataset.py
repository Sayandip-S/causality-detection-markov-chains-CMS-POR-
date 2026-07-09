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
    state_id_text: str,
) -> list[int]:
    """
    Convert a pipe-separated state sequence into a list of integers.

    Example
    -------
    "0|12|25" -> [0, 12, 25]
    """

    return [
        int(value)
        for value in state_id_text.split("|")
        if value.strip()
    ]


def get_prefix(
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

    parser.add_argument(
        "--prefix-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of each trace used as observed prefix "
            "(default: 0.5)."
        ),
    )

    args = parser.parse_args()

    if not 0.0 < args.prefix_fraction <= 1.0:
        raise ValueError(
            "prefix-fraction must be in (0, 1]."
        )

    traces = pd.read_csv(args.input)

    # Use only traces with known outcomes.
    traces = traces[
        traces["terminal_label"].isin(
            [
                "target",
                "success",
            ]
        )
    ].copy()

    all_prefixes: list[list[int]] = []
    labels: list[int] = []
    all_states: set[int] = set()

    for _, row in traces.iterrows():
        states = parse_state_ids(
            row["state_ids"]
        )

        prefix = get_prefix(
            states=states,
            prefix_fraction=args.prefix_fraction,
        )

        all_prefixes.append(prefix)
        labels.append(
            int(row["reached_target"])
        )

        all_states.update(prefix)

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
    print(f"Input traces: {args.input}")
    print(f"Output dataset: {args.output}")
    print(f"Rows: {len(dataset)}")
    print(f"Number of visited-state features: {len(sorted_states)}")
    print()
    print("Class distribution:")
    print(dataset["target"].value_counts())
    print()
    print("Positive rate:")
    print(dataset["target"].mean())
    print()
    print("Example columns:")
    print(list(dataset.columns[:20]))


if __name__ == "__main__":
    main()