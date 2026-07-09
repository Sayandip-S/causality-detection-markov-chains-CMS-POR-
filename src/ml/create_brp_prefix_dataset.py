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
    / "brp_prefix_dataset.csv"
)


def parse_state_ids(state_id_text: str) -> list[int]:
    """
    Convert a pipe-separated state sequence into integers.

    Example
    -------
    "0|1|5" -> [0, 1, 5]
    """

    return [
        int(value)
        for value in state_id_text.split("|")
        if value.strip()
    ]


def create_features_from_prefix(
    states: list[int],
    prefix_fraction: float,
) -> dict[str, int | float]:
    """
    Create simple prefix-based features from one trace.

    We only use the first part of the trace as the input,
    while the label says whether the full trace eventually
    reaches the target.
    """

    if not states:
        raise ValueError(
            "Trace contains no states."
        )

    prefix_length = max(
        1,
        int(len(states) * prefix_fraction),
    )

    prefix = states[:prefix_length]

    return {
        "first_state": prefix[0],
        "last_state": prefix[-1],
        "prefix_length": len(prefix),
        "unique_states": len(set(prefix)),
        "total_state_visits": len(prefix),
        "min_state_id": min(prefix),
        "max_state_id": max(prefix),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a simple prefix-feature dataset "
            "from generated BRP traces."
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
        help="Output ML dataset CSV.",
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

    # Keep only traces with a known final outcome.
    traces = traces[
        traces["terminal_label"].isin(
            [
                "target",
                "success",
            ]
        )
    ].copy()

    rows = []

    for _, row in traces.iterrows():
        states = parse_state_ids(
            row["state_ids"]
        )

        features = create_features_from_prefix(
            states=states,
            prefix_fraction=args.prefix_fraction,
        )

        features["target"] = int(
            row["reached_target"]
        )

        rows.append(features)

    dataset = pd.DataFrame(rows)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        args.output,
        index=False,
    )

    print("Prefix dataset created.")
    print(f"Input traces: {args.input}")
    print(f"Output dataset: {args.output}")
    print(f"Rows: {len(dataset)}")
    print()
    print("Class distribution:")
    print(dataset["target"].value_counts())
    print()
    print("Positive rate:")
    print(dataset["target"].mean())


if __name__ == "__main__":
    main()