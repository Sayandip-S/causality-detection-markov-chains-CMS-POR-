import stormpy

from src.storm.load_benchmark import (
    BRP_MODEL_PATH,
    BRP_PROPERTY_PATH,
)
from src.storm.model_utils import load_prism_model


def main() -> None:
    _, properties, model = load_prism_model(
        model_path=BRP_MODEL_PATH,
        property_path=BRP_PROPERTY_PATH,
    )

    result = stormpy.model_checking(
        model,
        properties[0],
    )

    initial_states = list(model.initial_states)

    if len(initial_states) != 1:
        raise ValueError(
            "Expected exactly one initial state."
        )

    initial_state = initial_states[0]
    exact_probability = float(
        result.at(initial_state)
    )

    print("BRP property:")
    print(
        BRP_PROPERTY_PATH.read_text(
            encoding="utf-8"
        ).strip()
    )
    print()

    print(
        "Exact probability of reaching "
        f"the BRP target: {exact_probability:.10f}"
    )


if __name__ == "__main__":
    main()