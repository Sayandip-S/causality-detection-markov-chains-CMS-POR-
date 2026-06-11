from pathlib import Path

import stormpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "prism" / "simple_dtmc.prism"
)

DEFAULT_PROPERTY_PATH = (
    PROJECT_ROOT / "models" / "properties" / "simple_dtmc.props"
)


def load_prism_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    property_path: Path = DEFAULT_PROPERTY_PATH,
):
    """
    Parse a PRISM program and build its Storm sparse model.

    Returns
    -------
    tuple
        program, properties, model
    """

    model_path = Path(model_path)
    property_path = Path(property_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"PRISM model file does not exist: {model_path}"
        )

    if not property_path.exists():
        raise FileNotFoundError(
            f"Property file does not exist: {property_path}"
        )

    property_text = property_path.read_text(
        encoding="utf-8"
    ).strip()

    if not property_text:
        raise ValueError(
            f"Property file is empty: {property_path}"
        )

    program = stormpy.parse_prism_program(
        str(model_path)
    )

    properties = stormpy.parse_properties_for_prism_program(
        property_text,
        program,
    )

    if not properties:
        raise ValueError(
            "No valid Storm properties were parsed."
        )

    model = stormpy.build_model(
        program,
        properties,
    )

    return program, properties, model