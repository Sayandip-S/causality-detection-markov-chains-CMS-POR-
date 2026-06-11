import stormpy

from model_utils import (
    DEFAULT_PROPERTY_PATH,
    load_prism_model,
)


def main() -> None:
    _, properties, model = load_prism_model()

    property_text = DEFAULT_PROPERTY_PATH.read_text(
        encoding="utf-8"
    ).strip()

    selected_property = properties[0]

    result = stormpy.model_checking(
        model,
        selected_property,
    )

    print("Model-checking property:")
    print(property_text)
    print()

    for initial_state in model.initial_states:
        probability = result.at(initial_state)

        print(f"Initial state ID: {initial_state}")
        print(
            "Exact probability of eventually reaching error: "
            f"{float(probability):.6f}"
        )


if __name__ == "__main__":
    main()