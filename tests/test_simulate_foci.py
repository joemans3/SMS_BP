import numpy as np
import pytest

from SMS_BP.simulate_foci import (
    create_condensate_dict,
    generate_points,
    get_lengths,
    tophat_function_2d,
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


# Test for `create_condensate_dict` function
def test_create_condensate_dict():
    initial_centers = np.array([[0, 0], [1, 1]])
    initial_scale = np.array([[1, 1], [1, 1]])
    diffusion_coefficient = np.array([[0.1, 0.1], [0.2, 0.2]])
    hurst_exponent = np.array([[0.5, 0.5], [0.6, 0.6]])
    cell_space = np.array([[0, 0], [2, 2]])
    cell_axial_range = 1.0

    condensates = create_condensate_dict(
        initial_centers,
        initial_scale,
        diffusion_coefficient,
        hurst_exponent,
        cell_space,
        cell_axial_range,
    )

    assert isinstance(condensates, dict)
    assert len(condensates) == 2
    assert "0" in condensates and "1" in condensates


# Test for `tophat_function_2d`
def test_tophat_function_2d_inside():
    center = [0, 0]
    radius = 1.0
    bias_subspace = 0.9
    space_prob = 0.1
    var_inside = [0.5, 0.5]

    prob = tophat_function_2d(var_inside, center, radius, bias_subspace, space_prob)
    assert prob == bias_subspace  # Should return bias_subspace probability


def test_tophat_function_2d_outside():
    center = [0, 0]
    radius = 1.0
    bias_subspace = 0.9
    space_prob = 0.1
    var_outside = [2, 2]

    prob = tophat_function_2d(var_outside, center, radius, bias_subspace, space_prob)
    assert prob == space_prob  # Should return space_prob probability


# Test for `generate_points`
def test_generate_points():
    def mock_pdf(var, center, radius, bias_subspace, space_prob):
        return tophat_function_2d(var, center, radius, bias_subspace, space_prob)

    center = [0, 0]
    radius = 1.0
    bias_subspace = 0.9
    space_prob = 0.1
    total_points = 100
    min_x = -2
    max_x = 2

    points = generate_points(
        pdf=mock_pdf,
        total_points=total_points,
        min_x=min_x,
        max_x=max_x,
        center=center,
        radius=radius,
        bias_subspace_x=bias_subspace,
        space_prob=space_prob,
        density_dif=0.1,
    )

    assert len(points) == total_points
    assert all(len(p) == 2 for p in points)  # Check if all points have 2 coordinates
