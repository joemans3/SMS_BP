"""
simulate_foci.py
================
This file contains the necessary classes and functions to simulate foci dynamics in space, particularly within cell simulations.

Author: Baljyot Singh Parmar

Classes:
--------
- Track_generator: A class to generate tracks of foci movements in a cell space with or without transitions.

Functions:
----------
- get_lengths: Generates an array of track lengths based on a chosen distribution.
- create_condensate_dict: Creates a dictionary of condensates for simulation.
- tophat_function_2d: Defines a circular top-hat probability distribution in 2D.
- generate_points: Generates random points following a given probability distribution.
- generate_points_from_cls: Generates 3D points using the accept/reject method based on a given distribution.
- generate_radial_points: Generates uniformly distributed points in a circle.
- generate_sphere_points: Generates uniformly distributed points in a sphere.
- radius_spherical_cap: Computes the radius of a spherical cap given the sphere's radius and a z-slice.
- get_gaussian: Returns a 2D Gaussian distribution over a given domain.
- axial_intensity_factor: Computes the axial intensity factor based on axial position.
- generate_map_from_points: Generates a spatial map from given points and intensities.
"""

from typing import Callable, Tuple, overload

import numpy as np
from boundedfbm.motion.FBM import FBM_BP
from scipy.stats import multivariate_normal

from .cells import BaseCell


def get_lengths(
    track_distribution: str, track_length_mean: int, total_tracks: int
) -> np.ndarray:
    """
    Returns track lengths based on the specified distribution.

    Parameters:
    -----------
    track_distribution : str
        The distribution of track lengths. Options are "exponential", "uniform", and "constant".
    track_length_mean : int
        The mean length of the tracks.
    total_tracks : int
        The total number of tracks to generate.

    Returns:
    --------
    np.ndarray
        An array of track lengths (shape: (total_tracks,)).

    Raises:
    -------
    ValueError
        If the distribution type is not recognized.
    """
    if track_distribution == "exponential":
        # make sure each of the lengths is an integer and is greater than or equal to 1
        return np.array(
            np.ceil(np.random.exponential(scale=track_length_mean, size=total_tracks)),
            dtype=int,
        )
    elif track_distribution == "uniform":
        # make sure each of the lengths is an integer
        return np.array(
            np.ceil(
                np.random.uniform(
                    low=1, high=2 * (track_length_mean) - 1, size=total_tracks
                )
            ),
            dtype=int,
        )
    elif track_distribution == "constant":
        return np.array(np.ones(total_tracks) * int(track_length_mean), dtype=int)
    else:
        raise ValueError("Distribution not recognized")


def tophat_function_2d(
    var: np.ndarray,
    center: np.ndarray,
    radius: float,
    bias_subspace: float,
    space_prob: float,
    **kwargs,
) -> float:
    """
    Defines a circular top-hat probability distribution in 2D.

    Parameters:
    -----------
    var : np.ndarray
        [x, y] coordinates for sampling the distribution.
    center : np.ndarray
        [c1, c2] coordinates representing the center of the top-hat region.
    radius : float
        Radius of the circular top-hat.
    bias_subspace : float
        Probability at the center of the top-hat.
    space_prob : float
        Probability outside the top-hat region.

    Returns:
    --------
    float
        The probability value at the given coordinates.
    """
    x = var[0]
    y = var[1]
    if ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius**2:
        return bias_subspace
    else:
        return space_prob


def generate_points(
    pdf: callable,
    total_points: int,
    min_x: float,
    max_x: float,
    center: np.ndarray,
    radius: float,
    bias_subspace_x: float,
    space_prob: float,
    density_dif: float,
) -> np.ndarray:
    """
    Generates random (x, y) points using the accept/reject method based on a given distribution.

    Parameters:
    -----------
    pdf : callable
        Probability density function to sample from.
    total_points : int
        Number of points to generate.
    min_x : float
        Minimum x value for sampling.
    max_x : float
        Maximum x value for sampling.
    center : np.ndarray
        Coordinates of the center of the top-hat distribution.
    radius : float
        Radius of the top-hat region.
    bias_subspace_x : float
        Probability at the top of the top-hat.
    space_prob : float
        Probability outside the top-hat region.
    density_dif : float
        Scaling factor for density differences.

    Returns:
    --------
    np.ndarray
        Array of generated points.
    """
    xy_coords = []
    while len(xy_coords) < total_points:
        # generate candidate variable
        var = np.random.uniform([min_x, min_x], [max_x, max_x])
        # generate varibale to condition var1
        var2 = np.random.uniform(0, 1)
        # apply condition
        pdf_val = pdf(var, center, radius, bias_subspace_x, space_prob)
        if var2 < ((1.0 / density_dif) * (max_x - min_x) ** 2) * pdf_val:
            xy_coords.append(var)
    return np.array(xy_coords)


def generate_points_from_cls(
    pdf: Callable,
    total_points: int,
    volume: float,
    bounds: Tuple[float, float, float, float, float, float],
    density_dif: float,
) -> np.ndarray:
    """
    Generates random (x, y, z) points using the accept/reject method based on a given distribution.

    Parameters:
    -----------
    pdf : callable
        Probability density function to sample from.
    total_points : int
        Number of points to generate.
    bound : list with the following
        min_x : float
            Minimum x value for sampling.
        max_x : float
            Maximum x value for sampling.
        min_y : float
            Minimum y value for sampling.
        max_y : float
            Maximum y value for sampling.
        min_z : float
            Minimum z value for sampling.
        max_z : float
            Maximum z value for sampling.
    volume : float,
        volume of region sampling
    density_dif : float
        Scaling factor for density differences.

    Returns:
    --------
    np.ndarray
        Array of generated (x, y, z) points.
    """
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    xyz_coords = []
    while len(xyz_coords) < total_points:
        # generate candidate variable
        var = np.random.uniform([min_x, min_y, min_z], [max_x, max_y, max_z])
        # generate varibale to condition var1
        var2 = np.random.uniform(0, 1)
        # apply condition
        pdf_val = pdf(var)
        if var2 < ((1.0 / density_dif) * volume) * pdf_val:
            xyz_coords.append(var)
    return np.array(xyz_coords)


def generate_radial_points(
    total_points: int, center: np.ndarray, radius: float
) -> np.ndarray:
    """
    Generates uniformly distributed points in a circle of a given radius.

    Parameters:
    -----------
    total_points : int
        Number of points to generate.
    center : np.ndarray
        Coordinates of the center of the circle.
    radius : float
        Radius of the circle.

    Returns:
    --------
    np.ndarray
        Array of generated (x, y) coordinates.
    """
    theta = 2.0 * np.pi * np.random.random(size=total_points)
    rad = radius * np.sqrt(np.random.random(size=total_points))
    x = rad * np.cos(theta) + center[0]
    y = rad * np.sin(theta) + center[1]
    return np.stack((x, y), axis=-1)


def generate_sphere_points(
    total_points: int, center: np.ndarray, radius: float
) -> np.ndarray:
    """
    Generates uniformly distributed points in a sphere of a given radius.

    Parameters:
    -----------
    total_points : int
        Number of points to generate.
    center : np.ndarray
        Coordinates of the center of the sphere.
    radius : float
        Radius of the sphere.

    Returns:
    --------
    np.ndarray
        Array of generated (x, y, z) coordinates.
    """
    # check to see if the center is an array of size 3
    if len(center) != 3:
        # make it an array of size 3 with the last element being 0
        center = np.array([center[0], center[1], 0])

    theta = 2.0 * np.pi * np.random.random(size=total_points)
    phi = np.arccos(2.0 * np.random.random(size=total_points) - 1.0)
    rad = radius * np.cbrt(np.random.random(size=total_points))
    x = rad * np.cos(theta) * np.sin(phi) + center[0]
    y = rad * np.sin(theta) * np.sin(phi) + center[1]
    z = rad * np.cos(phi) + center[2]
    return np.stack((x, y, z), axis=-1)


def radius_spherical_cap(R: float, center: np.ndarray, z_slice: float) -> float:
    """
    Calculates the radius of a spherical cap at a given z-slice.

    Parameters:
    -----------
    R : float
        Radius of the sphere.
    center : np.ndarray
        [x, y, z] coordinates of the center of the sphere.
    z_slice : float
        Z-coordinate of the slice relative to the sphere's center.

    Returns:
    --------
    float
        Radius of the spherical cap at the given z-slice.

    Raises:
    -------
    ValueError
        If the z-slice is outside the sphere.
    """
    # check if z_slice is within the sphere
    if z_slice > R:
        raise ValueError("z_slice is outside the sphere")
    # check if z_slice is at the edge of the sphere
    if z_slice == R:
        return 0
    # check if z_slice is at the center of the sphere
    if z_slice == 0:
        return R
    # calculate the radius of the spherical cap
    return np.sqrt(R**2 - (z_slice) ** 2)


# numpy version of get_gaussian
def get_gaussian(
    mu: np.ndarray,
    sigma: float | np.ndarray,
    domain: list[list[int]] = [list(range(10)), list(range(10))],
) -> np.ndarray:
    """
    Generates a 2D Gaussian distribution over a given domain.

    Parameters:
    -----------
    mu : np.ndarray
        Center position of the Gaussian (x, y).
    sigma : float | np.ndarray
        Standard deviation(s) of the Gaussian.
    domain : list[list[int]], optional
        Domain over which to compute the Gaussian (default is 0-9 for x and y).

    Returns:
    --------
    np.ndarray
        2D array representing the Gaussian distribution over the domain.
    """
    # generate a multivariate normal distribution with the given mu and sigma over the domain using scipy stats
    # generate the grid
    x = domain[0]
    y = domain[1]
    xx, yy = np.meshgrid(x, y)
    # generate the multivariate normal distribution
    rv = multivariate_normal(mu, sigma)
    # generate the probability distribution
    gauss = rv.pdf(np.dstack((xx, yy)))
    # reshape the distribution on the grid
    return gauss


def axial_intensity_factor(
    abs_axial_pos: float | np.ndarray, detection_range: float, **kwargs
) -> float | np.ndarray:
    """Docstring
    Calculate the factor for the axial intensity of the PSF given the absolute axial position from the 0 position of
    the focal plane. This is the factor that is multiplied by the intensity of the PSF

    For now this is a negative exponential decay i.e:
        I = I_0*e^(-|z-z_0|)
    This function returns the factor e^(-|z-z_0|**2 / (2*2.2**2)) only.

    Parameters:
    -----------
    abs_axial_pos : float|np.ndarray
        absolute axial position from the 0 position of the focal plane
    detection_range : float
        detection range of the function. This is the standard deviation of the gaussian function describing the axial intensity decay assuming a gaussian function.
    kwargs : dict

    Returns:
    --------
    float|np.ndarray
        factor for the axial intensity of the PSF
    """
    func_type = kwargs.get("func", "ones")
    if func_type == "ones":
        try:
            return np.ones(len(abs_axial_pos))
        except Exception:
            return 1
    elif func_type == "exponential":
        # for now this uses a negative exponential decay
        return np.exp(-(abs_axial_pos**2) / (2 * detection_range**2))


def generate_map_from_points(
    points: np.ndarray,
    point_intensity: float | np.ndarray,
    map: np.ndarray,
    movie: bool,
    base_noise: float,
    psf_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a 2D spatial map from a set of points and their intensities.

    Parameters:
    -----------
    points : np.ndarray
        Array of points of shape (total_points, 2).
    point_intensity : float | np.ndarray
        Intensity of the points.
    map : np.ndarray
        Pre-defined space map to update. If None, a new map is generated.
    movie : bool
        If True, noise is added to the whole image at once; otherwise, noise is added per point.
    base_noise : float
        Base noise level to add to the spatial map.
    psf_sigma : float
        Sigma of the PSF (in pixel units).

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        The updated spatial map and the points.
    """

    space_map = map
    x = np.arange(0, np.shape(map)[0], 1.0)
    y = np.arange(0, np.shape(map)[1], 1.0)

    if np.isscalar(point_intensity):
        point_intensity *= np.ones(len(points))

    if point_intensity is None:
        for i, j in enumerate(points):
            space_map += get_gaussian(j, np.ones(2) * psf_sigma, domain=[x, y])
    else:
        for i, j in enumerate(points):
            gauss_probability = get_gaussian(j, np.ones(2) * psf_sigma, domain=[x, y])
            # normalize
            gauss_probability = gauss_probability / np.max(gauss_probability)

            # generate poisson process over this space using the gaussian probability as means
            if not movie:
                space_map += np.random.poisson(
                    gauss_probability * point_intensity[i] + base_noise,
                    size=(len(x), len(y)),
                )
            else:
                space_map += gauss_probability * point_intensity[i]
        if movie:
            intensity = np.random.poisson(space_map + base_noise, size=(len(x), len(y)))
            space_map = intensity
    return space_map, points


class Track_generator:
    """
    A class to generate tracks of foci movements in a simulated cell space.

    Parameters:
    -----------
    cell : BaseCell
        Cell object defining the space for track generation
    oversample_motion_time : int | float
        Time for oversampling motion in milliseconds.
    """

    def __init__(
        self,
        cell: BaseCell,
        total_time: int | float,
        oversample_motion_time: int | float,
    ) -> None:
        self.cell = cell
        self._allowable_cell_types()

        self.oversample_motion_time = oversample_motion_time  # in ms
        # total time in ms is the exposure time + interval time * (cycle_count) / oversample_motion_time
        # in ms
        self.total_time = total_time

    def _allowable_cell_types(self):
        # only allow rectangular cells for now
        # if not isinstance(self.cell, RectangularCell):
        #     raise ValueError(
        #         "Only rectangular cells are supported for track generation"
        #     )
        pass

    def track_generation_no_transition(
        self,
        diffusion_coefficient: float,  # um^2/s
        hurst_exponent: float,
        track_length: int,
        initials: np.ndarray,
        start_time: int | float,
    ) -> dict:
        """
        Simulates the track generation with no transition between the diffusion coefficients and the hurst exponents
        namely, this means each track has a unique diffusion coefficient and hurst exponent
        This simulation is confined to the cell space and the axial range of the cell

        Parameters:
        -----------
        diffusion_coefficient : float
            diffusion coefficient for the track
        hurst_exponent : float
            hurst exponent for the track
        track_length : int
            track_length for the track
        initials : array-like
            [[x,y,z]] coordinates of the initial positions of the track
        start_time : int
            time at which the track start (this is not the frame, and needs to be converted to the frame using the exposure time and interval time and the oversample motion time)
        Returns:
        --------
        dict-like with format: {"xy":xyz,"times":times,"diffusion_coefficient":diffusion_coefficient,"hurst":hurst_exponent,"initial":initial}
        """
        # initialize the fbm class
        # make self.space_lim relative to the initial position, using self.space_lim define the 0 to be initial position
        if np.shape(initials) == (2,):
            # change the shape to (3,)
            initials = np.array([initials[0], initials[1], 0])
        # convert the diffusion_coefficients
        # diffusion_coefficient = self._convert_diffcoef_um2s_um2xms(
        #     diffusion_coefficient
        # )
        fbm = FBM_BP(
            n=track_length,
            dt=self.oversample_motion_time / 1000.0,
            hurst_parameters=[hurst_exponent],
            diffusion_parameters=[diffusion_coefficient],
            diffusion_parameter_transition_matrix=[1],
            hurst_parameter_transition_matrix=[1],
            state_probability_diffusion=[1],
            state_probability_hurst=[1],
            cell=self.cell,
            initial_position=initials,
        )
        xyz = fbm.fbm(dims=3)
        # make the times starting from the starting time
        track_times = np.arange(
            start_time,
            (track_length + start_time),
            1,
        )
        track_xyz = xyz
        # create the dict
        track_data = {
            "xy": track_xyz,
            "times": track_times,
            "diffusion_coefficient": fbm._diff_a_n,
            "hurst": fbm._hurst_n,
            "initial": initials,
        }
        # construct the dict
        return track_data

    def track_generation_with_transition(
        self,
        diffusion_transition_matrix: np.ndarray | list,
        hurst_transition_matrix: np.ndarray | list,
        diffusion_parameters: np.ndarray | list,  # um^2/s
        hurst_parameters: np.ndarray | list,
        diffusion_state_probability: np.ndarray | list,
        hurst_state_probability: np.ndarray | list,
        track_length: int,
        initials: np.ndarray,
        start_time: int | float,
    ) -> dict:
        """
        Genereates the track data with transition between the diffusion coefficients and the hurst exponents

        Parameters:
        -----------
        diffusion_transition_matrix : array-like
            transition matrix for the diffusion coefficients
        hurst_transition_matrix : array-like
            transition matrix for the hurst exponents
        diffusion_parameters : array-like
            diffusion coefficients for the tracks
        hurst_parameters : array-like
            hurst exponents for the tracks
        diffusion_state_probability : array-like
            probabilities for the diffusion coefficients
        hurst_state_probability : array-like
            probabilities for the hurst exponents
        track_length : int
            track_length for the track
        initials : array-like
            [[x,y,z]] coordinates of the initial positions of the track
        start_time : int
            time at which the track start (this is not the frame, and needs to be converted to the frame using the exposure time and interval time and the oversample motion time)

        Returns:
        --------
        dict-like with format: {"xy":xyz,"times":times,"diffusion_coefficient":diffusion_coefficient,"hurst":hurst_exponent,"initial":initial}
        """
        # make self.space_lim relative to the initial position, using self.space_lim define the 0 to be initial position
        # self.space_lim is in general shape (3,2) while the initials is in shape (3,)
        # make sure the - operator is broadcasted correctly
        if np.shape(initials) == (2,):
            # change the shape to (3,)
            initials = np.array([initials[0], initials[1], 0])
        # subtract each element of the first dimension of self.space_lim by the first element of initials

        # convert the diffusion_coefficients
        # diffusion_parameters = self._convert_diffcoef_um2s_um2xms(diffusion_parameters)
        # initialize the fbm class
        fbm = FBM_BP(
            n=track_length,
            dt=self.oversample_motion_time / 1000.0,
            hurst_parameters=hurst_parameters,
            diffusion_parameters=diffusion_parameters,
            diffusion_parameter_transition_matrix=diffusion_transition_matrix,
            hurst_parameter_transition_matrix=hurst_transition_matrix,
            state_probability_diffusion=diffusion_state_probability,
            state_probability_hurst=hurst_state_probability,
            cell=self.cell,
            initial_position=initials,
        )
        xyz = fbm.fbm(dims=3)
        # make the times starting from the starting time
        track_times = np.arange(
            start_time,
            track_length + start_time,
            1,
        )
        track_xyz = xyz
        # create the dict
        track_data = {
            "xy": track_xyz,
            "times": track_times,
            "diffusion_coefficient": fbm._diff_a_n,
            "hurst": fbm._hurst_n,
            "initial": initials,
        }
        # construct the dict
        return track_data

    def track_generation_constant(
        self, track_length: int, initials: np.ndarray, start_time: int
    ) -> dict:
        """
        Generate a constant track (no movement).

        Parameters:
        -----------
        track_length : int
            mean track length, in this case the track length is constant with this mean
        initials : array-like
            [[x,y,z]] coordinates of the initial positions of the track
        starting_time : int
            time at which the track start (this is not the frame, and needs to be converted to the frame using the exposure time and interval time and the oversample motion time)

        Returns:
        --------
        dict-like with format: {"xy":xyz,"times":times,"diffusion_coefficient":diffusion_coefficient,"hurst":hurst_exponent,"initial":initial}
        """
        # make the times starting from the starting time
        track_times = np.arange(
            start_time,
            track_length + start_time,
            1,
        )
        # make the track x,y,z from the initial positions
        track_xyz = np.tile(initials, (len(track_times), 1))
        # construct the dict
        track_data = {
            "xy": track_xyz,
            "times": track_times,
            "diffusion_coefficient": 0,
            "hurst": 0,
            "initial": initials,
        }
        return track_data

    @overload
    def _convert_diffcoef_um2s_um2xms(self, diffusion_coefficient: float) -> float: ...
    @overload
    def _convert_diffcoef_um2s_um2xms(
        self, diffusion_coefficient: np.ndarray
    ) -> np.ndarray: ...
    @overload
    def _convert_diffcoef_um2s_um2xms(self, diffusion_coefficient: list) -> list: ...
    def _convert_diffcoef_um2s_um2xms(
        self, diffusion_coefficient: float | np.ndarray | list
    ) -> float | np.ndarray | list:
        """converts um^2/s diffusion_coefficient into um^2/ x ms
        x = amount of ms
        ms = milliseconds

        x ms = self.oversample_motion_time (in ms, int)"""
        if isinstance(diffusion_coefficient, (np.ndarray, float)):
            return (
                1.0 / (1000.0 / self.oversample_motion_time)
            ) * diffusion_coefficient
        elif isinstance(diffusion_coefficient, list):
            return [
                (1.0 / (1000.0 / self.oversample_motion_time)) * i
                for i in diffusion_coefficient
            ]
        else:
            raise TypeError(f"Unsupported type: {type(diffusion_coefficient)}")

    def _convert_time_to_frame(
        self, time: int, exposure_time: int, interval_time: int
    ) -> int:
        """
        Parameters:
        -----------
        time : int
            time in ms

        Returns:
        --------
        int: frame number
        """
        return int(
            (time * self.oversample_motion_time) / (exposure_time + interval_time)
        )

    def _convert_frame_to_time(
        self, frame: int, exposure_time: int, interval_time: int
    ) -> int:
        """
        Parameters:
        -----------
        frame : int
            frame number

        Returns:
        --------
        int: time in ms
        """
        return int((frame * (exposure_time + interval_time)))


def _initialize_points_per_time(total_time: int, oversample_motion_time: int) -> dict:
    """Initialize empty points per time dictionary.

    Returns
    -------
    dict
        Empty dictionary with keys for each time point
    """
    return {
        str(i): []
        for i in np.arange(
            0, total_time + oversample_motion_time, oversample_motion_time
        )
    }


def _update_points_per_time(points_per_time: dict, track: dict) -> None:
    """Update points per time dictionary with new track data.

    Parameters
    ----------
    points_per_time : dict
        Dictionary to update
    track : dict
        Track data to add
    """
    for frame, position in zip(track["times"], track["xy"]):
        points_per_time[str(frame)].append(position)


def _generate_constant_tracks(
    track_generator: Track_generator,
    track_lengths: list | np.ndarray | int,
    initial_positions: np.ndarray,
    starting_times: int = 0,
) -> tuple[dict, dict]:
    """Generate tracks with constant parameters."""
    if isinstance(track_lengths, int):
        track_lengths = np.full(len(initial_positions), track_lengths)
    if isinstance(starting_times, int):
        starting_times = np.full(len(initial_positions), starting_times)

    tracks = {}
    points_per_time = _initialize_points_per_time(
        track_generator.total_time, track_generator.oversample_motion_time
    )
    for i in range(len(track_lengths)):
        tracks[i] = track_generator.track_generation_constant(
            track_length=track_lengths[i],
            initials=initial_positions[i],
            start_time=starting_times[i],
        )
        _update_points_per_time(points_per_time, tracks[i])

    return tracks, points_per_time


def _generate_no_transition_tracks(
    track_generator: Track_generator,
    track_lengths: list | np.ndarray | int,
    initial_positions: np.ndarray,
    starting_times: int,
    diffusion_parameters: np.ndarray,
    hurst_parameters: np.ndarray,
) -> tuple[dict, dict]:
    """Generate tracks without state transitions.

    Parameters
    ----------
    track_generator : sf.Track_generator
        Track generator instance
    track_lengths : list | np.ndarray | int
        Track lengths
    initial_positions : np.ndarray
        Initial positions
    starting_times : int
        Starting times
    diffusion_parameters : np.ndarray
        Diffusion parameters
    hurst_parameters : np.ndarray
        Hurst parameters

    Returns
    -------
    tuple[dict, dict]
        Tracks dictionary and points per time dictionary
    """
    if isinstance(track_lengths, int):
        track_lengths = np.full(len(initial_positions), track_lengths)
    if isinstance(starting_times, int):
        starting_times = np.full(len(initial_positions), starting_times)

    tracks = {}
    points_per_time = _initialize_points_per_time(
        track_generator.total_time, track_generator.oversample_motion_time
    )

    for i in range(len(track_lengths)):
        # Randomly select diffusion coefficient and hurst exponent indices
        diff_idx = np.random.randint(0, len(diffusion_parameters) - 1)
        hurst_idx = np.random.randint(0, len(hurst_parameters) - 1)

        # Generate track with selected parameters
        tracks[i] = track_generator.track_generation_no_transition(
            track_length=track_lengths[i],
            initials=initial_positions[i],
            start_time=starting_times[i],
            diffusion_coefficient=diffusion_parameters[diff_idx],
            hurst_exponent=hurst_parameters[hurst_idx],
        )
        _update_points_per_time(points_per_time, tracks[i])

    return tracks, points_per_time


def _generate_transition_tracks(
    track_generator: Track_generator,
    track_lengths: list | np.ndarray | int,
    initial_positions: np.ndarray,
    starting_times: int,
    diffusion_parameters: np.ndarray,
    hurst_parameters: np.ndarray,
    diffusion_transition_matrix: np.ndarray,
    hurst_transition_matrix: np.ndarray,
    diffusion_state_probability: np.ndarray,
    hurst_state_probability: np.ndarray,
) -> tuple[dict, dict]:
    """Generate tracks with state transitions.

    Parameters
    ----------
    track_generator : sf.Track_generator
        Track generator instance
    track_lengths : list | np.ndarray | int
        Track lengths
    initial_positions : np.ndarray
        Initial positions
    starting_times : int
        Starting times
    diffusion_parameters : np.ndarray
        Diffusion parameters
    hurst_parameters : np.ndarray
        Hurst parameters
    diffusion_transition_matrix : np.ndarray
        Diffusion transition matrix
    hurst_transition_matrix : np.ndarray
        Hurst transition matrix
    diffusion_state_probability : np.ndarray
        Diffusion state probability
    hurst_state_probability : np.ndarray
        Hurst state probability

    Returns
    -------
    tuple[dict, dict]
        Tracks dictionary and points per time dictionary
    """
    if isinstance(track_lengths, int):
        track_lengths = np.full(len(initial_positions), track_lengths)
    if isinstance(starting_times, int):
        starting_times = np.full(len(initial_positions), starting_times)

    tracks = {}
    points_per_time = _initialize_points_per_time(
        track_generator.total_time, track_generator.oversample_motion_time
    )

    for i in range(len(track_lengths)):
        # Generate track with transitions
        tracks[i] = track_generator.track_generation_with_transition(
            diffusion_transition_matrix=diffusion_transition_matrix,
            hurst_transition_matrix=hurst_transition_matrix,
            diffusion_parameters=diffusion_parameters,
            hurst_parameters=hurst_parameters,
            diffusion_state_probability=diffusion_state_probability,
            hurst_state_probability=hurst_state_probability,
            track_length=track_lengths[i],
            initials=initial_positions[i],
            start_time=starting_times[i],
        )
        _update_points_per_time(points_per_time, tracks[i])

    return tracks, points_per_time
