from __future__ import annotations

import argparse
from pathlib import Path

from src.storm.generate_traces import (
    compute_exact_target_probability,
    generate_traces,
    save_traces_to_csv,
)
from src.storm.load_benchmark import (
    BRP_MODEL_PATH,
    BRP_PROPERTY_PATH,
)
from src.storm.model_utils import (
    PROJECT_ROOT,
    load_prism_model,
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "brp_traces.csv"
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for BRP trace generation.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate execution traces from the BRP "
            "benchmark using StormPy."
        )
    )

    parser.add_argument(
        "--num-traces",
        type=int,
        default=100,
        help=(
            "Number of traces to generate "
            "(default: 100)."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help=(
            "Maximum transitions per trace "
            "(default: 500)."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility "
            "(default: 42)."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path.",
    )

    return parser.parse_args()


def print_class_balance(
    target_count: int,
    success_count: int,
    truncated_count: int,
    total_count: int,
) -> None:
    """
    Print the class distribution of generated traces.
    """

    positive_rate = (
        target_count / total_count
        if total_count > 0
        else 0.0
    )

    success_rate = (
        success_count / total_count
        if total_count > 0
        else 0.0
    )

    truncated_rate = (
        truncated_count / total_count
        if total_count > 0
        else 0.0
    )

    print()
    print("Class balance")
    print("-------------")
    print(
        f"Positive target traces: "
        f"{target_count} "
        f"({positive_rate:.2%})"
    )
    print(
        f"Negative success traces: "
        f"{success_count} "
        f"({success_rate:.2%})"
    )
    print(
        f"Truncated traces: "
        f"{truncated_count} "
        f"({truncated_rate:.2%})"
    )

    print()

    if target_count == 0:
        print(
            "Warning: no positive target traces were generated. "
            "This dataset cannot be used for binary ML training."
        )

    elif success_count == 0:
        print(
            "Warning: no negative success traces were generated. "
            "This dataset cannot be used for binary ML training."
        )

    elif positive_rate < 0.01:
        print(
            "Warning: the positive class is extremely rare."
        )

    elif positive_rate < 0.10:
        print(
            "Notice: the dataset is strongly imbalanced."
        )

    elif positive_rate > 0.90:
        print(
            "Notice: the dataset is strongly imbalanced "
            "toward the target class."
        )

    else:
        print(
            "The generated traces contain examples "
            "from both classes."
        )


def main() -> None:
    """
    Load BRP, generate traces, save them, and report
    the class distribution.
    """

    arguments = parse_arguments()

    print("Loading BRP benchmark...")
    print(f"Model: {BRP_MODEL_PATH}")
    print(f"Property: {BRP_PROPERTY_PATH}")
    print()

    _, properties, model = load_prism_model(
        model_path=BRP_MODEL_PATH,
        property_path=BRP_PROPERTY_PATH,
    )

    print("BRP model loaded successfully.")
    print(f"States: {model.nr_states}")
    print(f"Transitions: {model.nr_transitions}")
    print()

    exact_probability = (
        compute_exact_target_probability(
            model,
            properties,
        )
    )

    print(
        "Exact Storm target probability: "
        f"{exact_probability:.10f}"
    )

    traces = generate_traces(
        model=model,
        number_of_traces=arguments.num_traces,
        maximum_steps=arguments.max_steps,
        seed=arguments.seed,
        target_label="target",
        negative_terminal_label="success",
        monitor_label=None,
    )

    save_traces_to_csv(
        traces=traces,
        output_path=arguments.output,
    )

    target_count = sum(
        trace.reached_target
        for trace in traces
    )

    success_count = sum(
        trace.terminal_label == "success"
        for trace in traces
    )

    truncated_count = sum(
        trace.terminal_label == "max_steps"
        for trace in traces
    )

    empirical_probability = (
        target_count / len(traces)
    )

    print()
    print("BRP trace-generation summary")
    print("----------------------------")
    print(f"Total traces: {len(traces)}")
    print(f"Target traces: {target_count}")
    print(f"Success traces: {success_count}")
    print(f"Truncated traces: {truncated_count}")
    print(
        "Empirical target probability: "
        f"{empirical_probability:.10f}"
    )
    print(
        "Absolute difference from exact value: "
        f"{abs(empirical_probability - exact_probability):.10f}"
    )
    print(f"Saved to: {arguments.output}")

    print_class_balance(
        target_count=target_count,
        success_count=success_count,
        truncated_count=truncated_count,
        total_count=len(traces),
    )


if __name__ == "__main__":
    main()