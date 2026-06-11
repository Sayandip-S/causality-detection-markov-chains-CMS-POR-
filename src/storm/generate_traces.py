from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import stormpy

from model_utils import (
    DEFAULT_PROPERTY_PATH,
    PROJECT_ROOT,
    load_prism_model,
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "raw" / "simple_dtmc_traces.csv"
)


@dataclass
class TraceResult:
    """
    Stores one simulated execution trace.
    """

    trace_id: int
    state_ids: list[int]
    valuations: list[str]
    terminal_label: str
    reached_error: bool
    reached_candidate: bool
    number_of_transitions: int


def get_state_labels(model, state_id: int) -> set[str]:
    """
    Return all labels associated with one Storm state.
    """
    state = model.states[state_id]
    return set(state.labels)


def get_state_valuation(model, state_id: int) -> str:
    """
    Return the original PRISM valuation of one Storm state.

    Example:
        [s=3]
    """
    return model.state_valuations.get_string(state_id)


def get_outgoing_transitions(
    model,
    state_id: int,
) -> list[tuple[int, float]]:
    """
    Extract outgoing transitions from one DTMC state.

    Returns
    -------
    list of tuple
        Each tuple is:
        (target_state_id, transition_probability)
    """
    state = model.states[state_id]

    actions = list(state.actions)

    if len(actions) != 1:
        raise ValueError(
            "This trace generator currently supports DTMCs only. "
            f"State {state_id} has {len(actions)} actions."
        )

    transitions: list[tuple[int, float]] = []

    for transition in actions[0].transitions:
        target_state = int(transition.column)
        probability = float(transition.value())

        transitions.append(
            (target_state, probability)
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
    Sample one target state according to transition probabilities.
    """
    random_value = random_generator.random()
    cumulative_probability = 0.0

    for target_state, probability in transitions:
        cumulative_probability += probability

        if random_value <= cumulative_probability:
            return target_state

    # Protect against very small floating-point rounding errors.
    return transitions[-1][0]


def simulate_trace(
    model,
    trace_id: int,
    random_generator: random.Random,
    maximum_steps: int,
) -> TraceResult:
    """
    Simulate one path through the Storm-built DTMC.

    The trace stops when:
    - an error-labelled state is reached;
    - a safe-labelled state is reached;
    - maximum_steps is reached.
    """
    initial_states = list(model.initial_states)

    if len(initial_states) != 1:
        raise ValueError(
            "This first implementation expects exactly one "
            f"initial state, but found {len(initial_states)}."
        )

    current_state = int(initial_states[0])

    state_ids = [current_state]
    valuations = [
        get_state_valuation(model, current_state)
    ]

    reached_error = False
    reached_candidate = False
    terminal_label = "max_steps"

    for _ in range(maximum_steps + 1):
        labels = get_state_labels(
            model,
            current_state,
        )

        if "candidate" in labels:
            reached_candidate = True

        if "error" in labels:
            reached_error = True
            terminal_label = "error"
            break

        if "safe" in labels:
            terminal_label = "safe"
            break

        transitions = get_outgoing_transitions(
            model,
            current_state,
        )

        current_state = sample_next_state(
            transitions,
            random_generator,
        )

        state_ids.append(current_state)
        valuations.append(
            get_state_valuation(
                model,
                current_state,
            )
        )
    else:
        terminal_label = "max_steps"

    return TraceResult(
        trace_id=trace_id,
        state_ids=state_ids,
        valuations=valuations,
        terminal_label=terminal_label,
        reached_error=reached_error,
        reached_candidate=reached_candidate,
        number_of_transitions=len(state_ids) - 1,
    )


def generate_traces(
    model,
    number_of_traces: int,
    maximum_steps: int,
    seed: int,
) -> list[TraceResult]:
    """
    Generate multiple traces using one reproducible random seed.
    """
    if number_of_traces <= 0:
        raise ValueError(
            "number_of_traces must be positive."
        )

    if maximum_steps <= 0:
        raise ValueError(
            "maximum_steps must be positive."
        )

    random_generator = random.Random(seed)

    traces = []

    for trace_id in range(number_of_traces):
        trace = simulate_trace(
            model=model,
            trace_id=trace_id,
            random_generator=random_generator,
            maximum_steps=maximum_steps,
        )

        traces.append(trace)

    return traces


def save_traces_to_csv(
    traces: Iterable[TraceResult],
    output_path: Path,
) -> None:
    """
    Save traces in a CSV file.

    Variable-length paths are stored using the | separator.
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
            "reached_error",
            "reached_candidate",
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
                    "trace_id": trace.trace_id,
                    "state_ids": "|".join(
                        str(state_id)
                        for state_id in trace.state_ids
                    ),
                    "valuations": "|".join(
                        trace.valuations
                    ),
                    "terminal_label": (
                        trace.terminal_label
                    ),
                    "reached_error": int(
                        trace.reached_error
                    ),
                    "reached_candidate": int(
                        trace.reached_candidate
                    ),
                    "number_of_transitions": (
                        trace.number_of_transitions
                    ),
                }
            )


def compute_exact_error_probability(
    model,
    properties,
) -> float:
    """
    Compute the exact error-reachability probability using Storm.
    """
    result = stormpy.model_checking(
        model,
        properties[0],
    )

    initial_states = list(model.initial_states)

    if len(initial_states) != 1:
        raise ValueError(
            "Expected one initial state."
        )

    return float(
        result.at(initial_states[0])
    )


def print_summary(
    traces: list[TraceResult],
    exact_probability: float,
    seed: int,
    maximum_steps: int,
    output_path: Path,
) -> None:
    """
    Print summary statistics for the generated traces.
    """
    number_of_traces = len(traces)

    error_count = sum(
        trace.reached_error
        for trace in traces
    )

    safe_count = sum(
        trace.terminal_label == "safe"
        for trace in traces
    )

    truncated_count = sum(
        trace.terminal_label == "max_steps"
        for trace in traces
    )

    candidate_count = sum(
        trace.reached_candidate
        for trace in traces
    )

    empirical_probability = (
        error_count / number_of_traces
    )

    absolute_error = abs(
        empirical_probability - exact_probability
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
        f"Number of traces: {number_of_traces}"
    )
    print(f"Random seed: {seed}")
    print(
        f"Maximum steps per trace: {maximum_steps}"
    )
    print(f"Error traces: {error_count}")
    print(f"Safe traces: {safe_count}")
    print(
        f"Truncated traces: {truncated_count}"
    )
    print(
        f"Traces visiting candidate: "
        f"{candidate_count}"
    )
    print(
        f"Average transitions per trace: "
        f"{average_trace_length:.4f}"
    )
    print()
    print(
        f"Exact Storm error probability: "
        f"{exact_probability:.6f}"
    )
    print(
        f"Empirical error probability: "
        f"{empirical_probability:.6f}"
    )
    print(
        f"Absolute estimation error: "
        f"{absolute_error:.6f}"
    )
    print()
    print(f"Traces saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate execution traces from a "
            "Storm-built DTMC."
        )
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
            "CSV output path."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    _, properties, model = load_prism_model()

    exact_probability = (
        compute_exact_error_probability(
            model,
            properties,
        )
    )

    traces = generate_traces(
        model=model,
        number_of_traces=arguments.num_traces,
        maximum_steps=arguments.max_steps,
        seed=arguments.seed,
    )

    save_traces_to_csv(
        traces,
        arguments.output,
    )

    print_summary(
        traces=traces,
        exact_probability=exact_probability,
        seed=arguments.seed,
        maximum_steps=arguments.max_steps,
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()