from pathlib import Path

import stormpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "prism"
    / "simple_dtmc.prism"
)

DEFAULT_PROPERTY_PATH = (
    PROJECT_ROOT
    / "models"
    / "properties"
    / "simple_dtmc.props"
)


def load_prism_model(
    model_path: str | Path | None = None,
    property_path: str | Path | None = None,
):
    """
    Load a PRISM model and construct its reachable Storm model.

    Parameters
    ----------
    model_path:
        Path to a .prism or .pm model file.

    property_path:
        Path to a .props or .pctl property file.

    Returns
    -------
    tuple
        parsed PRISM program,
        parsed properties,
        constructed Storm model
    """

    selected_model_path = (
        Path(model_path)
        if model_path is not None
        else DEFAULT_MODEL_PATH
    )

    selected_property_path = (
        Path(property_path)
        if property_path is not None
        else DEFAULT_PROPERTY_PATH
    )

    if not selected_model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {selected_model_path}"
        )

    if not selected_property_path.exists():
        raise FileNotFoundError(
            f"Property file not found: {selected_property_path}"
        )

    property_text = selected_property_path.read_text(
        encoding="utf-8"
    ).strip()

    if not property_text:
        raise ValueError(
            f"Property file is empty: {selected_property_path}"
        )

    program = stormpy.parse_prism_program(
        str(selected_model_path)
    )

    properties = (
        stormpy.parse_properties_for_prism_program(
            property_text,
            program,
        )
    )

    if not properties:
        raise ValueError(
            "No valid properties were parsed."
        )

    builder_options = stormpy.BuilderOptions(
        [
            property_object.raw_formula
            for property_object in properties
        ]
    )

    builder_options.set_build_state_valuations()
    builder_options.set_build_all_labels()

    model = stormpy.build_sparse_model_with_options(
        program,
        builder_options,
    )

    return program, properties, model