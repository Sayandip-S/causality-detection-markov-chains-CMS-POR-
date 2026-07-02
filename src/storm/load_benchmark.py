from pathlib import Path

from src.storm.model_utils import (
    PROJECT_ROOT,
    load_prism_model,
)


BRP_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "prism"
    / "brp"
    / "brp.pm"
)

BRP_PROPERTY_PATH = (
    PROJECT_ROOT
    / "models"
    / "properties"
    / "brp"
    / "brp_failure.pctl"
)


def main() -> None:
    print("Loading BRP benchmark...")
    print(f"Model: {BRP_MODEL_PATH}")
    print(f"Property: {BRP_PROPERTY_PATH}")
    print()

    _, properties, model = load_prism_model(
        model_path=BRP_MODEL_PATH,
        property_path=BRP_PROPERTY_PATH,
    )

    print("BRP benchmark loaded successfully.")
    print(f"Model type: {model.model_type}")
    print(f"Number of states: {model.nr_states}")
    print(f"Number of transitions: {model.nr_transitions}")
    print(f"Initial states: {list(model.initial_states)}")
    print(f"Properties: {len(properties)}")


if __name__ == "__main__":
    main()