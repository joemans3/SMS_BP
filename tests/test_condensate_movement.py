import numpy as np
import pytest

from SMS_BP.condensate_movement import Condensate

# Define sample values to use across tests
SAMPLE_POSITION = np.array([1, 2, 3])
SAMPLE_CELL_SPACE = np.array([[-10, 10], [-10, 10], [-5, 5]])


@pytest.fixture
def sample_condensate_params():
    return {
        "initial_position": SAMPLE_POSITION,
        "initial_time": 0,
        "diffusion_coefficient": 0.5,
        "hurst_exponent": 0.7,
        "units_time": "ms",
        "units_position": "um",
        "condensate_id": 1,
        "initial_scale": 1.0,
        "cell_space": SAMPLE_CELL_SPACE,
        "cell_axial_range": 5,
    }


@pytest.fixture
def sample_condensate(sample_condensate_params):
    return Condensate(**sample_condensate_params)


@pytest.fixture
def zero_position_condensate(sample_condensate_params):
    params = sample_condensate_params.copy()
    params.update(
        {
            "initial_position": np.array([0, 0, 0]),
            "diffusion_coefficient": 0.1,
            "hurst_exponent": 0.5,
        }
    )
    return Condensate(**params)


def test_condensate_initialization(sample_condensate, sample_condensate_params):
    for key, value in sample_condensate_params.items():
        assert getattr(sample_condensate, key) == pytest.approx(value)


def test_condensate_call(zero_position_condensate):
    result = zero_position_condensate(0, "ms")
    assert "Position" in result
    assert "Scale" in result
    assert np.all(result["Position"] == np.array([0, 0, 0]))
    assert result["Scale"] == 1.0


def test_condensate_call_time_mismatch(sample_condensate):
    with pytest.raises(ValueError, match="Time units do not match to the condensate."):
        sample_condensate(1, "s")


def test_condensate_generate_positions(zero_position_condensate):
    zero_position_condensate(10, "ms")
    assert len(zero_position_condensate.times) == 11
    assert zero_position_condensate.condensate_positions.shape == (11, 3)
    assert len(zero_position_condensate.scale) == 11


def test_condensate_add_positions(sample_condensate):
    new_times = np.array([1, 2, 3])
    new_positions = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    new_scales = np.array([1.1, 1.2, 1.3])

    sample_condensate.add_positions(new_times, new_positions, new_scales)

    assert np.all(sample_condensate.times == np.array([0, 1, 2, 3]))
    assert np.all(
        sample_condensate.condensate_positions
        == np.vstack([SAMPLE_POSITION, new_positions])
    )
    assert np.all(sample_condensate.scale == np.array([1.0, 1.1, 1.2, 1.3]))
