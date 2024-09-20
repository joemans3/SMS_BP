import numpy as np
import matplotlib.pyplot as plt
from SMS_BP.boundary_conditions import _refecting_boundary, _absorbing_boundary

BOUNDARY_CONDITIONS = {
    "reflecting": _refecting_boundary,
    "absorbing": _absorbing_boundary,
}


def MCMC_state_selection(
    initial_state_index: int,
    transition_matrix: np.ndarray,
    possible_states: np.ndarray,
    n: int,
):
    """Markov Chain Monte Carlo state selection

    Parameters:
    -----------
    initial_state_index : int
        Initial state index, this is the index of the initial state in the possible states
    transition_matrix : np.ndarray
        Transition matrix, this is the prbability at a time step. (time step is 1)
    possible_states : np.ndarray
        possible states
    n : int
        Number of iterations

    Returns:
    --------
    np.ndarray
        State selection at each iteration
    """
    # initialize the state selection
    state_selection = np.zeros(n)
    # initialize the current state
    current_state = possible_states[initial_state_index]
    current_state_index = initial_state_index
    # iterate through the number of iterations
    for i in range(n):
        # find the probability of switching to each state
        state_probability = transition_matrix[current_state_index]
        # find the next state
        next_state_index = np.random.choice(
            np.arange(len(possible_states)), p=state_probability
        )
        next_state = possible_states[next_state_index]
        # update the current state
        current_state = next_state
        current_state_index = next_state_index
        state_selection[i] = current_state
    return state_selection


class FBM_BP:
    def __init__(
        self,
        n: int,
        dt: float,
        diffusion_parameters: np.ndarray,
        hurst_parameters: np.ndarray,
        diffusion_parameter_transition_matrix: np.ndarray,
        hurst_parameter_transition_matrix: np.ndarray,
        state_probability_diffusion: np.ndarray,
        state_probability_hurst: np.ndarray,
        space_lim: np.ndarray,
    ):
        self.n = int(n)
        self.dt = dt  # ms
        self.diffusion_parameter = diffusion_parameters
        self.hurst_parameter = hurst_parameters
        # state probability of the diffusion parameter
        self.diffusion_parameter_transition_matrix = (
            diffusion_parameter_transition_matrix
        )
        # state probability of the hurst parameter
        self.hurst_parameter_transition_matrix = hurst_parameter_transition_matrix
        # probability of the initial state, this approximates the population distribution
        self.state_probability_diffusion = state_probability_diffusion
        # probability of the initial state, this approximates the population distribution
        self.state_probability_hurst = state_probability_hurst
        # space lim (min, max) for the FBM
        self.space_lim = np.array(space_lim, dtype=float)
        # initialize the autocovariance matrix and the diffusion parameter
        self._setup()

    def _autocovariance(self, k, hurst):
        """Autocovariance function for fGn

        Parameters:
        -----------
        k : int
            Lag
        dt : float
            Time step
        hurst : float
            Hurst parameter
        diff_a : float
            Diffusion coefficient related to the Hurst parameter

        Returns:
        --------
        float
            Autocovariance function
        """
        return 0.5 * (
            abs(k - 1) ** (2 * hurst)
            - 2 * abs(k) ** (2 * hurst)
            + abs(k + 1) ** (2 * hurst)
        )

    def _setup(self) -> None:
        "setup to avoid recomputation of the autocovariance matrix and diffusion parameter"
        self._cov = np.zeros(self.n)
        self._diff_a_n = np.zeros(self.n)
        self._hurst_n = np.zeros(self.n)
        # catch if the diffusion or hurst parameter sets are singular
        if len(self.diffusion_parameter) == 1:
            self._diff_a_n = np.full(self.n, self.diffusion_parameter[0])
        else:
            diff_a_start = np.random.choice(
                self.diffusion_parameter, p=self.state_probability_diffusion
            )
            self._diff_a_n[0] = diff_a_start
            self._diff_a_n[1:] = MCMC_state_selection(
                np.where(self.diffusion_parameter == diff_a_start)[0][0],
                self.diffusion_parameter_transition_matrix,
                self.diffusion_parameter,
                self.n - 1,
            )
        if len(self.hurst_parameter) == 1:
            self._hurst_n = np.full(self.n, self.hurst_parameter[0])
        else:
            hurst_start = np.random.choice(
                self.hurst_parameter, p=self.state_probability_hurst
            )
            self._hurst_n[0] = hurst_start
            self._hurst_n[1:] = MCMC_state_selection(
                np.where(self.hurst_parameter == hurst_start)[0][0],
                self.hurst_parameter_transition_matrix,
                self.hurst_parameter,
                self.n - 1,
            )
        for i in range(self.n):
            self._cov[i] = self._autocovariance(i, self._hurst_n[i])

    def fbm(self):
        fgn = np.zeros(self.n)
        fbm_store = np.zeros(self.n)
        phi = np.zeros(self.n)
        psi = np.zeros(self.n)
        # construct a gaussian noise vector
        gn = (
            np.random.normal(0, 1, self.n)
            * np.sqrt(self.dt * 2 * self._diff_a_n)
            * (self.dt ** (2 * self._hurst_n))
        )
        # catch is all hurst are 0.5 then use the gaussian noise vector corresponding to the scale defined by the diffusion parameter
        if np.all(self._hurst_n == 0.5):
            # each gn is then pulled from a normal distribution with mean 0 and standard deviation diff_a_n
            # ignore the fbm calculations but keep the reflection
            for i in range(1, self.n):
                fbm_candidate = fbm_store[i - 1] + gn[i]
                # check if this is outside the space limit by using the reflecting boundary condition
                fbm_store[i] = _boundary_conditions(
                    fbm_store[i - 1], fbm_candidate, self.space_lim, "reflecting"
                )
            return fbm_store

        fbm_store[0] = 0
        fgn[0] = gn[0]
        v = 1
        phi[0] = 0

        for i in range(1, self.n):
            phi[i - 1] = self._cov[i]
            for j in range(i - 1):
                psi[j] = phi[j]
                phi[i - 1] -= psi[j] * self._cov[i - j - 1]
            phi[i - 1] /= v
            for j in range(i - 1):
                phi[j] = psi[j] - phi[i - 1] * psi[i - j - 2]
            v *= 1 - phi[i - 1] * phi[i - 1]
            for j in range(i):
                fgn[i] += phi[j] * fgn[i - j - 1]
            fgn[i] += np.sqrt(np.abs(v)) * gn[i]
            # add to the fbm
            fbm_candidate = fbm_store[i - 1] + fgn[i]

            # check if this is outside the space limit by using the reflecting boundary condition
            fbm_store[i] = _boundary_conditions(
                fbm_store[i - 1], fbm_candidate, self.space_lim, "reflecting"
            )
            if fbm_store[i] != fbm_candidate:
                # update the fgn based on the new difference
                fgn[i] = fbm_store[i] - fbm_store[i - 1]
        return fbm_store


def _boundary_conditions(
    fbm_store_last: float,
    fbm_candidate: float,
    space_lim: np.ndarray,
    condition_type: str,
):
    """Boundary conditions for the FBM

    Parameters:
    -----------
    fbm_store_last : float
        Last value of the FBM
    fbm_candidate : float
        Candidate value of the FBM
    space_lim : np.ndarray
        Space limit (min, max) for the FBM\
    condition_type : str
        Type of boundary condition takes values in REFLECTING_CONDITIONS
    Returns:
    --------
    float
        New value of the FBM
    """
    # check if the condition type is valid
    if condition_type not in BOUNDARY_CONDITIONS:
        raise ValueError(
            "Invalid condition type: "
            + condition_type
            + "! Must be one of: "
            + str(BOUNDARY_CONDITIONS.keys())
        )
    return BOUNDARY_CONDITIONS[condition_type](fbm_store_last, fbm_candidate, space_lim)


# run tests if this is the main module


if __name__ == "__main__":
    # test the MCMC_state_selection function
    # initialize the transition matrix
    transition_matrix = np.array([[0.4, 0.6], [0.2, 0.8]])
    # initialize the possible states
    possible_states = np.array([1, 2])
    # initialize the number of iterations
    n = 50000
    # initialize the initial state index
    initial_state_index = 1
    state_select = MCMC_state_selection(
        initial_state_index, transition_matrix, possible_states, n
    )
    # find the probability of each state
    state_probability = np.zeros(len(possible_states))
    for i in range(len(possible_states)):
        state_probability[i] = np.sum(state_select == possible_states[i]) / n
    # compare the population distribution with the state probability distribution
    total_rate = np.sum(transition_matrix)
    # add the column of the transition matrix and divide by the total rate
    true_state_probability = np.sum(transition_matrix, axis=0) / total_rate
    plt.bar(
        possible_states,
        state_probability,
        label="State probability distribution",
        alpha=0.9,
    )
    # plt.bar(possible_states, true_state_probability, label='Population distribution', alpha=0.5)
    plt.xlabel("State")
    plt.ylabel("Probability")
    plt.title("State probability distribution")
    plt.legend()
    plt.show()
    plt.plot(state_select)
    plt.show()
    # find the probability of switching from state 1 to state 2 at each iteration
    state_1_to_2 = np.zeros(n) - 1
    for i in range(n - 1):
        if state_select[i] == 1 and state_select[i + 1] == 2:
            state_1_to_2[i] = 1
        elif state_select[i] == 1 and state_select[i + 1] == 1:
            state_1_to_2[i] = 0
    # find the probability of switching from state 2 to state 1 at each iteration
    state_2_to_1 = np.zeros(n) - 1
    for i in range(n - 1):
        if state_select[i] == 2 and state_select[i + 1] == 1:
            state_2_to_1[i] = 1
        elif state_select[i] == 2 and state_select[i + 1] == 2:
            state_2_to_1[i] = 0
    # print the probability of switching from state 1 to state 2 ignore 0s
    print(
        "Probability of switching from state 1 to state 2: "
        + str(np.sum(state_1_to_2[state_1_to_2 != -1]) / np.sum(state_1_to_2 != -1))
    )
    # print the probability of switching from state 2 to state 1 ignore 0s
    print(
        "Probability of switching from state 2 to state 1: "
        + str(np.sum(state_2_to_1[state_2_to_1 != -1]) / np.sum(state_2_to_1 != -1))
    )

    # # test the FBM_BP class
    # n = 2000
    # dt = 1
    # diffusion_parameters = np.array([0.1,0.1])
    # hurst_parameters = np.array([0.8,0.2])
    # diffusion_parameter_transition_matrix = np.array([[0.01, 0.01],
    #                                                 [0.01, 0.01]])
    # hurst_parameter_transition_matrix = np.array([[0.9, 0.1],
    #                                             [0.1, 0.9]])
    # state_probability_diffusion = np.array([0.5,0.5])
    # state_probability_hurst = np.array([0.5,0.5])
    # space_lim = [-10,10]
    # fbm_bp = FBM_BP(n, dt, diffusion_parameters, hurst_parameters, diffusion_parameter_transition_matrix, hurst_parameter_transition_matrix, state_probability_diffusion, state_probability_hurst, space_lim)
    # # test the fbm method
    # fbm = fbm_bp.fbm()
    # # plot the fbm
    # plt.plot(fbm, linestyle='--')
    # plt.xlabel('Iteration')
    # plt.ylabel('Value')
    # plt.title('Fractional Brownian motion')
    # plt.show()

    # # test the MCMC_state_selection function
    # # initialize the transition matrix
    # transition_matrix = np.array([[500, 6.9],
    #                             [6.9, 500]])
    # # initialize the possible states
    # possible_states = np.array([1, 2])
    # # initialize the number of iterations
    # n = 100
    # # initialize the initial state index
    # initial_state_index = 1

    # # test the MCMC_state_selection function
    # state_selection = MCMC_state_selection(initial_state_index, transition_matrix, possible_states, n)
    # # plot the state selection
    # plt.plot(state_selection)
    # plt.xlabel('Iteration')
    # plt.ylabel('State')
    # plt.title('State selection')
    # plt.show()

    # # plot the probability of each state
    # state_probability = np.zeros(len(possible_states))
    # for i in range(len(possible_states)):
    #     state_probability[i] = np.sum(state_selection == possible_states[i])/n

    # # compare the population distribution with the state probability distribution
    # total_rate = np.sum(transition_matrix)
    # # add the column of the transition matrix and divide by the total rate
    # true_state_probability = np.sum(transition_matrix, axis=0)/total_rate
    # plt.bar(possible_states, state_probability, label='State probability distribution', alpha=0.9)
    # plt.bar(possible_states, true_state_probability, label='Population distribution', alpha=0.5)
    # plt.xlabel('State')
    # plt.ylabel('Probability')
    # plt.title('State probability distribution')
    # plt.legend()
    # plt.show()

    # #test for singular diffusion and hurst parameter sets
    # n = 100
    # dt = 1
    # diffusion_parameters = np.array([1])
    # hurst_parameters = np.array([0.5])
    # diffusion_parameter_transition_matrix = np.array([[1]])
    # hurst_parameter_transition_matrix = np.array([[1]])
    # state_probability_diffusion = np.array([1])
    # state_probability_hurst = np.array([1])
    # space_lim = [-1000,1000]
    # fbm_bp = FBM_BP(n, dt, diffusion_parameters, hurst_parameters, diffusion_parameter_transition_matrix, hurst_parameter_transition_matrix, state_probability_diffusion, state_probability_hurst, space_lim)
    # # test the fbm method
    # fbm = fbm_bp.fbm()
    # # plot the fbm
    # plt.plot(fbm, linestyle='--')
    # plt.xlabel('Iteration')
    # plt.ylabel('Value')
    # plt.title('Fractional Brownian motion')
    # plt.show()

    # # #test the MSD calculation
    # import sys
    # sys.path.append('/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/Scripts')
    # sys.path.append('/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/Scripts/src')
    # from SMT_Analysis_BP.helpers.analysisFunctions.MSD_Utils import MSD_Calculations_Track_Dict
    # #make a 2D FBM by making two 1D FBM and then combining them
    # n = 1000
    # dt = 1
    # #singular
    # diffusion_parameters = np.array([2])
    # hurst_parameters = np.array([0.5])
    # diffusion_parameter_transition_matrix = np.array([[1]])
    # hurst_parameter_transition_matrix = np.array([[1]])
    # state_probability_diffusion = np.array([1])
    # state_probability_hurst = np.array([1])
    # space_lim = [-1000,1000]
    # fbm_bp = FBM_BP(n, dt, diffusion_parameters, hurst_parameters, diffusion_parameter_transition_matrix, hurst_parameter_transition_matrix, state_probability_diffusion, state_probability_hurst, space_lim)
    # # test the fbm method
    # fbm_x = fbm_bp.fbm()
    # fbm_bp = FBM_BP(n, dt, diffusion_parameters, hurst_parameters, diffusion_parameter_transition_matrix, hurst_parameter_transition_matrix, state_probability_diffusion, state_probability_hurst, space_lim)
    # fbm_y = fbm_bp.fbm()
    # #plot the fbm
    # plt.plot(fbm_x,fbm_y,'.-')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Fractional Brownian motion')
    # plt.show()

    # #combine the 1D FBM to make a 2D FBM in the form {track_ID: [[x0, y0], [x1, y1], ...]}
    # track_dict = {0: np.zeros((n,2))}
    # track_dict[0][:,0] = fbm_x
    # track_dict[0][:,1] = fbm_y
    # #calculate the MSD
    # MSD_calced = MSD_Calculations_Track_Dict(track_dict,pixel_to_um=1,frame_to_seconds=1,min_track_length=1, max_track_length=10000)
    # #plot the MSD
    # plt.plot(MSD_calced.combined_store.ensemble_MSD.keys(), MSD_calced.combined_store.ensemble_MSD.values(), linestyle='--')
    # #fit the MSD with a line to find the slope in log-log space
    # #do a linear fit
    # x = np.log(list(MSD_calced.combined_store.ensemble_MSD.keys())[:5])
    # y = np.log(list(MSD_calced.combined_store.ensemble_MSD.values())[:5])
    # A = np.vstack([x, np.ones(len(x))]).T
    # m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    # plt.plot(np.exp(x), np.exp(m*x + c), 'r', label='Fitted line')
    # #annotate the slope
    # plt.text(0.1, 0.1, 'Slope: ' + str(m), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    # #annotate the intercept
    # plt.text(0.1, 0.2, 'Intercept: ' + str(0.25*np.exp(c)), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    # plt.xlabel('Time')
    # plt.ylabel('MSD')
    # plt.title('MSD')
    # #log axis
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    pass
