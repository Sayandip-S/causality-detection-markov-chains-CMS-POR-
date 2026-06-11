from model_utils import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROPERTY_PATH,
    load_prism_model,
)


def main() -> None:
    print("Loading PRISM model...")
    print(f"Model file: {DEFAULT_MODEL_PATH}")
    print(f"Property file: {DEFAULT_PROPERTY_PATH}")
    print()

    _, properties, model = load_prism_model()

    print("Model loaded successfully.")
    print(f"Model type: {model.model_type}")
    print(f"Number of states: {model.nr_states}")
    print(f"Number of transitions: {model.nr_transitions}")
    print(f"Initial states: {list(model.initial_states)}")
    print(f"Number of properties: {len(properties)}")


if __name__ == "__main__":
    main()