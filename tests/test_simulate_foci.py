import numpy as np
import pytest

from SMS_BP.simulate_foci import (
    get_lengths,
)


# Test for the `get_lengths` function
@pytest.mark.parametrize(
    "distribution, mean, total_tracks, expected_length",
    [
        ("exponential", 10, 5, 5),
        ("uniform", 10, 5, 5),
        ("constant", 10, 5, 5),
    ],
)
def test_get_lengths(distribution, mean, total_tracks, expected_length):
    lengths = get_lengths(distribution, mean, total_tracks)
    assert len(lengths) == expected_length
    assert isinstance(lengths, np.ndarray)
    assert np.all(lengths >= 1)  # Ensure all track lengths are >= 1


def test_invalid_distribution():
    with pytest.raises(ValueError):
        get_lengths("invalid_distribution", 10, 5)
