import numpy as np
import pytest

from SMS_BP.fbm_BP import FBM_BP, MCMC_state_selection, _boundary_conditions

# Set a fixed seed for reproducibility
np.random.seed(42)


@pytest.fixture
def transition_matrix():
    return np.array([[0.4, 0.6], [0.2, 0.8]])


@pytest.fixture
def possible_states():
    return np.array([1, 2])


@pytest.fixture
def fbm_bp_instance():
    n = 1000
    dt = 1
    diffusion_parameters = np.array([0.1, 0.2])
    hurst_parameters = np.array([0.3, 0.7])
    diffusion_parameter_transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
    hurst_parameter_transition_matrix = np.array([[0.8, 0.2], [0.2, 0.8]])
    state_probability_diffusion = np.array([0.5, 0.5])
    state_probability_hurst = np.array([0.5, 0.5])
    space_lim = np.array([-10, 10])

    np.random.seed(42)  # Reset seed before creating instance
    return FBM_BP(
        n,
        dt,
        diffusion_parameters,
        hurst_parameters,
        diffusion_parameter_transition_matrix,
        hurst_parameter_transition_matrix,
        state_probability_diffusion,
        state_probability_hurst,
        space_lim,
    )


def test_MCMC_state_selection(transition_matrix, possible_states):
    n = 10000
    initial_state_index = 0

    np.random.seed(42)  # Reset seed before the test
    state_selection = MCMC_state_selection(
        initial_state_index, transition_matrix, possible_states, n
    )

    assert len(state_selection) == n
    assert all(state in possible_states for state in state_selection)

    # Check if the distribution of states is as expected
    state_counts = np.bincount(state_selection.astype(int))[1:]
    expected_prob = np.array([0.25, 0.75])  # Steady-state probabilities
    actual_prob = state_counts / n

    np.testing.assert_allclose(actual_prob, expected_prob, atol=0.01)


def test_FBM_BP_initialization(fbm_bp_instance):
    assert fbm_bp_instance.n == 1000
    assert fbm_bp_instance.dt == 1
    np.testing.assert_array_equal(fbm_bp_instance.diffusion_parameter, [0.1, 0.2])
    np.testing.assert_array_equal(fbm_bp_instance.hurst_parameter, [0.3, 0.7])
    np.testing.assert_array_equal(fbm_bp_instance.space_lim, [-10, 10])


def test_FBM_BP_fbm(fbm_bp_instance):
    np.random.seed(42)  # Reset seed before generating FBM
    fbm = fbm_bp_instance.fbm()

    assert len(fbm) == fbm_bp_instance.n
    assert fbm[0] == 0  # FBM should start at 0
    assert np.all(fbm >= fbm_bp_instance.space_lim[0]) and np.all(
        fbm <= fbm_bp_instance.space_lim[1]
    )


def test_boundary_conditions():
    space_lim = np.array([-10, 10])

    # Test reflecting boundary
    assert _boundary_conditions(9, 11, space_lim, "reflecting") == 9
    assert _boundary_conditions(-9, -11, space_lim, "reflecting") == -9

    # Test absorbing boundary
    assert _boundary_conditions(9, 11, space_lim, "absorbing") == 10
    assert _boundary_conditions(-9, -11, space_lim, "absorbing") == -10

    # Test invalid boundary condition
    with pytest.raises(ValueError):
        _boundary_conditions(0, 1, space_lim, "invalid")


def test_FBM_BP_singular_parameters():
    n = 1000
    dt = 1
    diffusion_parameters = np.array([0.1])
    hurst_parameters = np.array([0.5])
    diffusion_parameter_transition_matrix = np.array([[1]])
    hurst_parameter_transition_matrix = np.array([[1]])
    state_probability_diffusion = np.array([1])
    state_probability_hurst = np.array([1])
    space_lim = np.array([-10, 10])

    np.random.seed(42)  # Reset seed before creating instance
    fbm_bp = FBM_BP(
        n,
        dt,
        diffusion_parameters,
        hurst_parameters,
        diffusion_parameter_transition_matrix,
        hurst_parameter_transition_matrix,
        state_probability_diffusion,
        state_probability_hurst,
        space_lim,
    )

    np.random.seed(42)  # Reset seed before generating FBM
    fbm = fbm_bp.fbm()

    assert len(fbm) == n
    assert fbm[0] == 0
    assert np.all(fbm >= space_lim[0]) and np.all(fbm <= space_lim[1])

    # For Hurst = 0.5, check if the increments are uncorrelated
    increments = np.diff(fbm)
    autocorr = np.correlate(increments, increments, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    autocorr /= autocorr[0]
    np.testing.assert_allclose(autocorr[1:10], np.zeros(9), atol=0.1)
