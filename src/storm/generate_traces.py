from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import stormpy

try:
    # Works when running as a package:
    # python -m src.storm.generate_traces
    from src.storm.model_utils import (
        DEFAULT_MODEL_PATH,
        DEFAULT_PROPERTY_PATH,
        PROJECT_ROOT,
        load_prism_model,
    )
except ModuleNotFoundError:
    # Works when running directly:
    # python src/storm/generate_traces.py
    from model_utils import (
        DEFAULT_MODEL_PATH,
        DEFAULT_PROPERTY_PATH,
        PROJECT_ROOT,
        load_prism_model,
    )


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "simple_dtmc_traces.csv"
)


@dataclass
class TraceResult:
    """
    Stores one simulated execution trace.

    Attributes
    ----------
    trace_id:
        Unique identifier of the trace.

    state_ids:
        Storm internal state IDs visited during the trace.

    valuations:
        Original PRISM variable valuations of visited states.

    terminal_label:
        Reason the trace stopped:
        target label, negative terminal label, or "max_steps".

    reached_target:
        True if the positive target state was reached.

    reached_monitor:
        True if the optional monitor/candidate label was visited.

    number_of_transitions:
        Number of transitions taken during the trace.
    """

    trace_id: int
    state_ids: list[int]
    valuations: list[str]
    terminal_label: str
    reached_target: bool
    reached_monitor: bool
    number_of_transitions: int


def normalize_optional_label(
    label: str | None,
) -> str | None:
    """
    Convert an empty label or the text 'none' into Python None.
    """

    if label is None:
        return None

    cleaned_label = label.strip()

    if cleaned_label.lower() in {
        "",
        "none",
        "null",
    }:
        return None

    return cleaned_label


def get_state_labels(
    model,
    state_id: int,
) -> set[str]:
    """
    Return all labels associated with one Storm state.
    """

    state = model.states[state_id]
    return set(state.labels)


def get_state_valuation(
    model,
    state_id: int,
) -> str:
    """
    Return the original PRISM valuation of one Storm state.

    Example
    -------
    [s=3]
    """

    return model.state_valuations.get_string(
        state_id
    )


def get_outgoing_transitions(
    model,
    state_id: int,
) -> list[tuple[int, float]]:
    """
    Extract all outgoing transitions from one DTMC state.

    Returns
    -------
    list[tuple[int, float]]
        Pairs of:

        (target Storm state ID, probability)
    """

    state = model.states[state_id]
    actions = list(state.actions)

    if len(actions) != 1:
        raise ValueError(
            "This trace generator currently supports DTMCs only. "
            f"State {state_id} has {len(actions)} actions. "
            "More than one action may indicate nondeterminism."
        )

    transitions: list[tuple[int, float]] = []

    for transition in actions[0].transitions:
        target_state = int(
            transition.column
        )

        probability = float(
            transition.value()
        )

        transitions.append(
            (
                target_state,
                probability,
            )
        )

    if not transitions:
        raise ValueError(
            f"State {state_id} has no outgoing transitions."
        )

    total_probability = sum(
        probability
        for _, probability in transitions
    )

    if abs(total_probability - 1.0) > 1e-10:
        raise ValueError(
            f"Outgoing probabilities from state {state_id} "
            f"sum to {total_probability}, not 1."
        )

    return transitions


def sample_next_state(
    transitions: list[tuple[int, float]],
    random_generator: random.Random,
) -> int:
    """
    Sample one successor according to transition probabilities.
    """

    random_value = random_generator.random()
    cumulative_probability = 0.0

    for target_state, probability in transitions:
        cumulative_probability += probability

        if random_value <= cumulative_probability:
            return target_state

    # Floating-point safety fallback.
    return transitions[-1][0]


def simulate_trace(
    model,
    trace_id: int,
    random_generator: random.Random,
    maximum_steps: int,
    target_label: str,
    negative_terminal_label: str | None = None,
    monitor_label: str | None = None,
) -> TraceResult:
    """
    Simulate one execution path through a Storm-built DTMC.

    The trace stops when:

    1. the positive target label is reached;
    2. the negative terminal label is reached;
    3. the maximum number of transitions is reached.

    Examples
    --------
    Simple DTMC:

        target_label="error"
        negative_terminal_label="safe"
        monitor_label="candidate"

    BRP:

        target_label="target"
        negative_terminal_label="success"
        monitor_label=None
    """

    initial_states = list(
        model.initial_states
    )

    if len(initial_states) != 1:
        raise ValueError(
            "This implementation expects exactly one initial "
            f"state, but found {len(initial_states)}."
        )

    current_state = int(
        initial_states[0]
    )

    state_ids = [
        current_state
    ]

    valuations = [
        get_state_valuation(
            model,
            current_state,
        )
    ]

    reached_target = False
    reached_monitor = False
    terminal_label = "max_steps"

    for _ in range(maximum_steps):
        labels = get_state_labels(
            model,
            current_state,
        )

        if (
            monitor_label is not None
            and monitor_label in labels
        ):
            reached_monitor = True

        if target_label in labels:
            reached_target = True
            terminal_label = target_label
            break

        if (
            negative_terminal_label is not None
            and negative_terminal_label in labels
        ):
            terminal_label = (
                negative_terminal_label
            )
            break

        transitions = get_outgoing_transitions(
            model,
            current_state,
        )

        current_state = sample_next_state(
            transitions,
            random_generator,
        )

        state_ids.append(
            current_state
        )

        valuations.append(
            get_state_valuation(
                model,
                current_state,
            )
        )

    else:
        # The maximum number of transitions was taken.
        # Check whether the final state itself is terminal.
        final_labels = get_state_labels(
            model,
            current_state,
        )

        if (
            monitor_label is not None
            and monitor_label in final_labels
        ):
            reached_monitor = True

        if target_label in final_labels:
            reached_target = True
            terminal_label = target_label

        elif (
            negative_terminal_label is not None
            and negative_terminal_label
            in final_labels
        ):
            terminal_label = (
                negative_terminal_label
            )

        else:
            terminal_label = "max_steps"

    return TraceResult(
        trace_id=trace_id,
        state_ids=state_ids,
        valuations=valuations,
        terminal_label=terminal_label,
        reached_target=reached_target,
        reached_monitor=reached_monitor,
        number_of_transitions=(
            len(state_ids) - 1
        ),
    )


def generate_traces(
    model,
    number_of_traces: int,
    maximum_steps: int,
    seed: int,
    target_label: str,
    negative_terminal_label: str | None = None,
    monitor_label: str | None = None,
) -> list[TraceResult]:
    """
    Generate multiple execution traces.

    The same random seed produces the same generated traces.
    """

    if number_of_traces <= 0:
        raise ValueError(
            "number_of_traces must be positive."
        )

    if maximum_steps <= 0:
        raise ValueError(
            "maximum_steps must be positive."
        )

    if not target_label.strip():
        raise ValueError(
            "target_label must not be empty."
        )

    random_generator = random.Random(
        seed
    )

    traces: list[TraceResult] = []

    for trace_id in range(
        number_of_traces
    ):
        trace = simulate_trace(
            model=model,
            trace_id=trace_id,
            random_generator=random_generator,
            maximum_steps=maximum_steps,
            target_label=target_label,
            negative_terminal_label=(
                negative_terminal_label
            ),
            monitor_label=monitor_label,
        )

        traces.append(trace)

    return traces


def save_traces_to_csv(
    traces: Iterable[TraceResult],
    output_path: Path,
) -> None:
    """
    Save generated traces to a CSV file.

    Variable-length state sequences are stored using '|'
    as the separator.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        fieldnames = [
            "trace_id",
            "state_ids",
            "valuations",
            "terminal_label",
            "reached_target",
            "reached_monitor",
            "number_of_transitions",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for trace in traces:
            writer.writerow(
                {
                    "trace_id": (
                        trace.trace_id
                    ),
                    "state_ids": "|".join(
                        str(state_id)
                        for state_id
                        in trace.state_ids
                    ),
                    "valuations": "|".join(
                        trace.valuations
                    ),
                    "terminal_label": (
                        trace.terminal_label
                    ),
                    "reached_target": int(
                        trace.reached_target
                    ),
                    "reached_monitor": int(
                        trace.reached_monitor
                    ),
                    "number_of_transitions": (
                        trace.number_of_transitions
                    ),
                }
            )


def compute_exact_target_probability(
    model,
    properties,
) -> float:
    """
    Compute the exact target-reachability probability
    using Storm model checking.
    """

    if not properties:
        raise ValueError(
            "No property was supplied for model checking."
        )

    result = stormpy.model_checking(
        model,
        properties[0],
    )

    initial_states = list(
        model.initial_states
    )

    if len(initial_states) != 1:
        raise ValueError(
            "Expected exactly one initial state."
        )

    return float(
        result.at(initial_states[0])
    )


def print_class_balance(
    target_count: int,
    negative_count: int,
    truncated_count: int,
    total_count: int,
    target_label: str,
    negative_terminal_label: str | None,
) -> None:
    """
    Print the class distribution and relevant imbalance notice.
    """

    target_rate = target_count / total_count
    truncated_rate = truncated_count / total_count

    print()
    print("Class balance")
    print("-------------")
    print(
        f"Positive {target_label} traces: "
        f"{target_count} ({target_rate:.2%})"
    )

    if negative_terminal_label is not None:
        negative_rate = (
            negative_count / total_count
        )
        print(
            f"Negative {negative_terminal_label} traces: "
            f"{negative_count} ({negative_rate:.2%})"
        )

    print(
        f"Truncated traces: "
        f"{truncated_count} ({truncated_rate:.2%})"
    )
    print()

    if target_count == 0:
        print(
            "Warning: no positive target traces were generated. "
            "This dataset cannot train a target classifier."
        )

    elif (
        negative_terminal_label is not None
        and negative_count == 0
    ):
        print(
            f"Warning: no negative "
            f"{negative_terminal_label} traces were generated. "
            "This dataset cannot train a binary classifier."
        )

    elif target_rate < 0.01:
        print(
            "Notice: the dataset is extremely imbalanced "
            "with very few target traces."
        )

    elif target_rate < 0.10:
        if negative_terminal_label == "success":
            print(
                "Notice: the dataset is strongly imbalanced "
                "toward the success class."
            )
        elif negative_terminal_label is not None:
            print(
                "Notice: the dataset is strongly imbalanced "
                f"toward the {negative_terminal_label} class."
            )
        else:
            print(
                "Notice: the dataset is strongly imbalanced "
                "toward the non-target class."
            )

    elif target_rate > 0.99:
        if negative_terminal_label == "success":
            print(
                "Notice: the dataset is extremely imbalanced "
                "with very few success traces."
            )
        elif negative_terminal_label is not None:
            print(
                "Notice: the dataset is extremely imbalanced "
                f"with very few {negative_terminal_label} traces."
            )
        else:
            print(
                "Notice: the dataset is extremely imbalanced "
                "with very few non-target traces."
            )

    elif target_rate > 0.90:
        print(
            "Notice: the dataset is strongly imbalanced "
            "toward the target class."
        )

    elif 0.20 <= target_rate <= 0.80:
        if negative_terminal_label is not None:
            class_split = (
                f"target/{negative_terminal_label}"
            )
        else:
            class_split = "target/non-target"

        print(
            "Notice: the dataset has a reasonably balanced "
            f"{class_split} split for baseline ML training."
        )


def print_summary(
    traces: list[TraceResult],
    exact_probability: float,
    seed: int,
    maximum_steps: int,
    output_path: Path,
    target_label: str,
    negative_terminal_label: str | None,
    monitor_label: str | None,
) -> None:
    """
    Print summary statistics for generated traces.
    """

    number_of_traces = len(traces)

    if number_of_traces == 0:
        raise ValueError(
            "No traces were generated."
        )

    target_count = sum(
        trace.reached_target
        for trace in traces
    )

    if negative_terminal_label is None:
        negative_count = 0
    else:
        negative_count = sum(
            trace.terminal_label
            == negative_terminal_label
            for trace in traces
        )

    truncated_count = sum(
        trace.terminal_label == "max_steps"
        for trace in traces
    )

    monitor_count = sum(
        trace.reached_monitor
        for trace in traces
    )

    empirical_probability = (
        target_count / number_of_traces
    )

    absolute_error = abs(
        empirical_probability
        - exact_probability
    )

    average_trace_length = (
        sum(
            trace.number_of_transitions
            for trace in traces
        )
        / number_of_traces
    )

    print()
    print("Trace-generation summary")
    print("------------------------")
    print(
        f"Number of traces: "
        f"{number_of_traces}"
    )
    print(
        f"Random seed: {seed}"
    )
    print(
        f"Maximum steps per trace: "
        f"{maximum_steps}"
    )
    print(
        f"Target label: "
        f"{target_label}"
    )
    print(
        f"Target traces: "
        f"{target_count}"
    )

    if negative_terminal_label is not None:
        negative_label_heading = (
            negative_terminal_label[:1].upper()
            + negative_terminal_label[1:]
        )
        print(
            f"{negative_label_heading} traces: "
            f"{negative_count}"
        )

    print(
        f"Truncated traces: "
        f"{truncated_count}"
    )

    if monitor_label is not None:
        print(
            f"Traces visiting "
            f"'{monitor_label}': "
            f"{monitor_count}"
        )

    print(
        f"Average transitions per trace: "
        f"{average_trace_length:.4f}"
    )

    print()
    print(
        f"Exact Storm target probability: "
        f"{exact_probability:.6f}"
    )
    print(
        f"Empirical target probability: "
        f"{empirical_probability:.6f}"
    )
    print(
        f"Absolute estimation error: "
        f"{absolute_error:.6f}"
    )

    print()
    print(
        f"Traces saved to: "
        f"{output_path}"
    )

    print_class_balance(
        target_count=target_count,
        negative_count=negative_count,
        truncated_count=truncated_count,
        total_count=number_of_traces,
        target_label=target_label,
        negative_terminal_label=(
            negative_terminal_label
        ),
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate execution traces from a "
            "Storm-built DTMC."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Path to the PRISM model file. "
            "Defaults to the simple DTMC."
        ),
    )

    parser.add_argument(
        "--property",
        type=Path,
        default=DEFAULT_PROPERTY_PATH,
        help=(
            "Path to the property file. "
            "Defaults to the simple DTMC property."
        ),
    )

    parser.add_argument(
        "--num-traces",
        type=int,
        default=10_000,
        help=(
            "Number of traces to generate "
            "(default: 10000)."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help=(
            "Maximum number of transitions per trace "
            "(default: 100)."
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
        help=(
            "Path for the generated CSV file."
        ),
    )

    parser.add_argument(
        "--target-label",
        type=str,
        default="error",
        help=(
            "Label representing the positive target "
            "(default: error)."
        ),
    )

    parser.add_argument(
        "--negative-terminal-label",
        type=str,
        default="safe",
        help=(
            "Label representing the negative terminal "
            "outcome (default: safe). "
            "Use 'none' when no such label exists."
        ),
    )

    parser.add_argument(
        "--monitor-label",
        type=str,
        default="candidate",
        help=(
            "Optional label to track during traces "
            "(default: candidate). "
            "Use 'none' to disable."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Load a PRISM model, generate traces, save them,
    and compare simulation with exact model checking.
    """

    arguments = parse_arguments()

    target_label = arguments.target_label.strip()

    negative_terminal_label = (
        normalize_optional_label(
            arguments.negative_terminal_label
        )
    )

    monitor_label = normalize_optional_label(
        arguments.monitor_label
    )

    _, properties, model = load_prism_model(
        model_path=arguments.model,
        property_path=arguments.property,
    )

    exact_probability = (
        compute_exact_target_probability(
            model,
            properties,
        )
    )

    traces = generate_traces(
        model=model,
        number_of_traces=(
            arguments.num_traces
        ),
        maximum_steps=(
            arguments.max_steps
        ),
        seed=arguments.seed,
        target_label=target_label,
        negative_terminal_label=(
            negative_terminal_label
        ),
        monitor_label=monitor_label,
    )

    save_traces_to_csv(
        traces=traces,
        output_path=arguments.output,
    )

    print_summary(
        traces=traces,
        exact_probability=exact_probability,
        seed=arguments.seed,
        maximum_steps=arguments.max_steps,
        output_path=arguments.output,
        target_label=target_label,
        negative_terminal_label=(
            negative_terminal_label
        ),
        monitor_label=monitor_label,
    )


if __name__ == "__main__":
    main()
