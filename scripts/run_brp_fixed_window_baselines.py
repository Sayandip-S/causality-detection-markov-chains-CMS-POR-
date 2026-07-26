from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ml.train_brp_baselines import run_experiment
from src.storm.model_utils import PROJECT_ROOT


WINDOWS = (5, 10, 20, 50)
DATASET_DIRECTORY = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "brp_fixed_windows"
)
MODEL_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "brp_fixed_windows"
)


def dataset_path_for_window(window: int) -> Path:
    """Return the fixed-window dataset path for one window."""

    return (
        DATASET_DIRECTORY
        / f"brp_stress_tuned_visited_state_dataset_k{window}.csv"
    )


def combined_output_for_windows(windows: tuple[int, ...]) -> Path:
    """Keep the canonical output exclusive to complete default runs."""

    if windows == WINDOWS:
        filename = "combined_metrics.csv"
    else:
        window_suffix = "_".join(f"k{window}" for window in windows)
        filename = f"combined_metrics_{window_suffix}.csv"

    return OUTPUT_DIRECTORY / filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BRP baselines for fixed observation windows."
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        choices=WINDOWS,
        default=list(WINDOWS),
        help="Windows to train (default: 5 10 20 50).",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Persist each selected window's fitted baseline models.",
    )
    parser.add_argument(
        "--model-output-root",
        type=Path,
        default=MODEL_OUTPUT_DIRECTORY,
        help="Root directory used with --save-models.",
    )
    args = parser.parse_args()
    selected_windows = tuple(
        window for window in WINDOWS if window in args.windows
    )
    combined_output = combined_output_for_windows(selected_windows)
    print(f"Combined metrics output: {combined_output}")

    dataset_paths = {
        window: dataset_path_for_window(window)
        for window in selected_windows
    }
    missing_datasets = [
        path
        for path in dataset_paths.values()
        if not path.is_file()
    ]

    if missing_datasets:
        missing_list = "\n".join(
            f"- {path}" for path in missing_datasets
        )
        raise FileNotFoundError(
            "Missing expected fixed-window datasets:\n"
            f"{missing_list}"
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    combined_rows = []

    for window, dataset_path in dataset_paths.items():
        print()
        print("#" * 78)
        print(f"Fixed transition window: k{window}")
        print("#" * 78)
        results = run_experiment(
            dataset_path=dataset_path,
            feature_set="visited_states_only",
            test_size=0.2,
            random_seed=42,
            metrics_output=OUTPUT_DIRECTORY / f"k{window}.json",
            model_output_dir=(
                args.model_output_root / f"k{window}"
                if args.save_models
                else None
            ),
            observation_window=window,
        )

        for model_name, metrics in results["metrics"].items():
            combined_rows.append(
                {
                    "window": window,
                    "model": model_name,
                    **metrics,
                }
            )

    combined_metrics = pd.DataFrame(combined_rows)
    combined_metrics.to_csv(combined_output, index=False)

    print()
    print("Combined metrics:")
    print(combined_metrics.to_string(index=False))
    print()
    print(f"Combined metrics written to: {combined_output}")


if __name__ == "__main__":
    main()
