import numpy as np
import pytest

from SMS_BP.probability_functions import multiple_top_hat_probability


# Fixtures to provide commonly used data
@pytest.fixture
def prob_func_fixture():
    """Fixture to initialize the multiple_top_hat_probability object."""
    num_subspace = 2
    subspace_centers = np.array([[0, 0, 0], [2, 2, 0]])
    subspace_radius = np.array([1, 1])
    density_dif = 0.1
    space_size = np.array([6, 6, 4])

    return multiple_top_hat_probability(
        num_subspace, subspace_centers, subspace_radius, density_dif, space_size
    )


# Test probability calculation within subspaces
def test_prob_in_subspaces(prob_func_fixture):
    prob_func = prob_func_fixture
    # Inside subspace 1
    assert np.isclose(
        prob_func(np.array([0.5, 0.5, 0])), prob_func.subspace_probability, atol=1e-6
    )
    # Inside subspace 2
    assert np.isclose(
        prob_func(np.array([2.5, 2.5, 0])), prob_func.subspace_probability, atol=1e-6
    )


# Test probability calculation outside subspaces
def test_prob_outside_subspaces(prob_func_fixture):
    prob_func = prob_func_fixture
    # Outside of both subspaces
    assert np.isclose(
        prob_func(np.array([3.5, 3.5, 0])),
        prob_func.non_subspace_probability,
        atol=1e-6,
    )


# Test updating parameters
def test_update_parameters(prob_func_fixture):
    prob_func = prob_func_fixture
    new_num_subspace = 1
    new_subspace_centers = np.array([[1, 1, 1]])
    new_subspace_radius = np.array([2])
    new_density_dif = 0.2
    new_space_size = np.array([10, 10, 5])

    prob_func.update_parameters(
        num_subspace=new_num_subspace,
        subspace_centers=new_subspace_centers,
        subspace_radius=new_subspace_radius,
        density_dif=new_density_dif,
        space_size=new_space_size,
    )

    assert prob_func.num_subspace == new_num_subspace
    assert np.allclose(prob_func.subspace_centers, new_subspace_centers)
    assert np.allclose(prob_func.subspace_radius, new_subspace_radius)
    assert np.isclose(prob_func.density_dif, new_density_dif)
    assert np.allclose(prob_func.space_size, new_space_size)


# Test for invalid input types
def test_invalid_input_types(prob_func_fixture):
    prob_func = prob_func_fixture
    with pytest.raises(TypeError):
        prob_func("invalid input")  # Position must be a numpy array
    with pytest.raises(TypeError):
        prob_func.num_subspace = "invalid type"  # num_subspace must be an int


# Test calculation of subspace probability
def test_subspace_probability_calculation(prob_func_fixture):
    prob_func = prob_func_fixture
    space_size = prob_func.space_size
    density_dif = prob_func.density_dif
    expected_subspace_probability = density_dif / np.prod(space_size)

    assert np.isclose(
        prob_func.subspace_probability, expected_subspace_probability, atol=1e-6
    )


# Test calculation of non-subspace probability
def test_non_subspace_probability_calculation(prob_func_fixture):
    prob_func = prob_func_fixture
    space_size = prob_func.space_size
    expected_non_subspace_probability = 1.0 / np.prod(space_size)

    assert np.isclose(
        prob_func.non_subspace_probability, expected_non_subspace_probability, atol=1e-6
    )
