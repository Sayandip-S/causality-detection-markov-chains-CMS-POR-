import sys
from pathlib import Path

import stormpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORM_SOURCE_DIR = PROJECT_ROOT / "src" / "storm"

sys.path.insert(0, str(STORM_SOURCE_DIR))

from model_utils import load_prism_model  # noqa: E402


def test_simple_dtmc_structure() -> None:
    _, _, model = load_prism_model()

    assert model.nr_states == 6
    assert list(model.initial_states) == [0]


def test_error_reachability_probability() -> None:
    _, properties, model = load_prism_model()

    result = stormpy.model_checking(
        model,
        properties[0],
    )

    initial_state = list(model.initial_states)[0]
    probability = float(result.at(initial_state))

    assert abs(probability - 0.4) < 1e-10