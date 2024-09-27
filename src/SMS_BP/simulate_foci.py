"""
Documentation for the simulate_foci.py file.
This file contains the class for simulating foci in space.

Author: Baljyot Singh Parmar
"""

import numpy as np
from scipy.stats import multivariate_normal

import SMS_BP.condensate_movement as condensate_movement
import SMS_BP.fbm_BP as fbm_BP


def get_lengths(track_distribution: str, track_length_mean: int, total_tracks: int):
    """
    Returns the track lengths from the distribution track_distribution. The lengths are returned as the closest integer

    Parameters:
    -----------
    track_distribution: distribution of track lengths. Options are "exponential" and "uniform"
    track_length_mean: mean track length
    total_tracks: total number of tracks to be generated

    Returns:
    --------
    track lengths as a numpy array of shape (total_tracks,1)

    Notes:
    ------
    1. If the distribution is exponential, then the track lengths are generated using exponential distribution.
    2. If the distribution is uniform, then the track lengths are generated using uniform distribution between 0 and 2*track_length_mean.
    3. If the distribution is constant, then all the track lengths are set to the mean track length. (track_length_mean)

    Exceptions:
    -----------
    ValueError: if the distribution is not recognized.
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


def create_condensate_dict(
    initial_centers: np.ndarray,
    initial_scale: np.ndarray,
    diffusion_coefficient: np.ndarray,
    hurst_exponent: np.ndarray,
    cell_space: np.ndarray,
    cell_axial_range: float,
    **kwargs,
) -> dict:
    """
    Docstring for create_condensate_dict:

    Parameters:
    -----------
    inital_centers: numpy array of shape (num_condensates,2) with the initial centers of the condensates
    initial_scale: numpy array of shape (num_condensates,2) with the initial scales of the condensates
    diffusion_coefficient: numpy array of shape (num_condensates,2) with the diffusion coefficients of the condensates
    hurst_exponent: numpy array of shape (num_condensates,2) with the hurst exponents of the condensates
    cell_space: numpy array of shape (2,2) with the cell space
    cell_axial_range: float
    **kwargs: additional arguments to be passed to the condensate_movement.Condensate class
    """
    # check the length of diffusion_coefficient to find the number of condensates
    num_condensates = len(diffusion_coefficient)
    condensates = {}
    units_time = kwargs.get("units_time", ["ms"] * num_condensates)
    for i in range(num_condensates):
        condensates[str(i)] = condensate_movement.Condensate(
            inital_position=initial_centers[i],
            initial_scale=initial_scale[i],
            diffusion_coefficient=diffusion_coefficient[i],
            hurst_exponent=hurst_exponent[i],
            condensate_id=int(str(i)),
            units_time=units_time[i],
            cell_space=cell_space,
            cell_axial_range=cell_axial_range,
        )
    return condensates


def tophat_function_2d(var, center, radius, bias_subspace, space_prob, **kwargs):
    """
    Defines a circular top hat probability distribution with a single biased region defining the hat.
    The rest of the space is uniformly distrubuted in 2D

    Parameters
    ----------
    var : array-like, float
        [x,y] defining sampling on the x,y span of this distribution
    center : array-like, float
        [c1,c2] defining the center coordinates of the top hat region
    radius : float
        defines the radius of the circular tophat from the center
    bias_subspace : float
        probability at the top position of the top hat
    space_prob : float
        probability everywhere not in the bias_subspace

    Returns
    -------
    float, can be array-like if var[0],var[1] is array-like
        returns the value of bias_subspace or space_prob depending on where the [x,y] data lies

    """
    x = var[0]
    y = var[1]
    if ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius**2:
        return bias_subspace
    else:
        return space_prob


def generate_points(
    pdf,
    total_points,
    min_x,
    max_x,
    center,
    radius,
    bias_subspace_x,
    space_prob,
    density_dif,
):
    """
    genereates random array of (x,y) points given a distribution using accept/reject method

    Parameters
    ----------
    pdf : function
        function which defines the distribution to sample from
    total_points : int
        total points to sample
    min_x : float
        lower bound to the support of the distribution
    max_x : float
        upper bound to the support of the distribution
    center : array-like of float
        coordinates of the center of the top hat
    redius : float
        raidus of the top hat
    bias_subspace : float
        probability at the top hat
    space_prob : float
        probaility everywhere not at the top hat

    Returns
    -------
    array-like
        [x,y] coordinates of the points sampled from the distribution defined in pdf
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
    pdf, total_points, min_x, max_x, min_y, max_y, min_z, max_z, density_dif
):
    xyz_coords = []
    area = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
    while len(xyz_coords) < total_points:
        # generate candidate variable
        var = np.random.uniform([min_x, min_y, min_z], [max_x, max_y, max_z])
        # generate varibale to condition var1
        var2 = np.random.uniform(0, 1)
        # apply condition
        pdf_val = pdf(var)
        if var2 < ((1.0 / density_dif) * area) * pdf_val:
            xyz_coords.append(var)
    return np.array(xyz_coords)


def generate_radial_points(total_points, center, radius):
    """Genereate uniformly distributed points in a circle of radius.

    Parameters
    ----------
    total_points : int
        total points from this distribution
    center : array-like or tuple
        coordinate of the center of the radius. [x,y,...]
    radius : float-like
        radius of the region on which to

    Returns
    -------
    (n,2) size array
        array of coordinates of points genereated (N,3) N = # of points, 2 = dimentions
    """
    theta = 2.0 * np.pi * np.random.random(size=total_points)
    rad = radius * np.sqrt(np.random.random(size=total_points))
    x = rad * np.cos(theta) + center[0]
    y = rad * np.sin(theta) + center[1]
    return np.stack((x, y), axis=-1)


def generate_sphere_points(total_points, center, radius):
    """Genereate uniformly distributed points in a sphere of radius.

    Parameters
    ----------
    total_points : int
        total points from this distribution
    center : array-like or tuple
        coordinate of the center of the radius. [x,y,...]
    radius : float-like
        radius of the region on which to

    Returns
    -------
    (n,2) size array
        array of coordinates of points genereated (N,3) N = # of points, 2 = dimentions
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


def radius_spherical_cap(R, center, z_slice):
    """Find the radius of a spherical cap given the radius of the sphere and the z coordinate of the slice
    Theory: https://en.wikipedia.org/wiki/Spherical_cap, https://mathworld.wolfram.com/SphericalCap.html

    Parameters:
    -----------
    R : float,int
        radius of the sphere
    center : array-like
        [x,y,z] coordinates of the center of the sphere
    z_slice : float,int
        z coordinate of the slice relative to the center of the sphere, z_slice = 0 is the center of the sphere

    Returns:
    --------
    float
        radius of the spherical cap at the z_slice

    Notes:
    ------
    1. This is a special case of the spherical cap equation where the center of the sphere is at the origin
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
def get_gaussian(mu, sigma, domain=[list(range(10)), list(range(10))]):
    """
    Parameters
    ----------
    mu : array-like or float of floats
        center position of gaussian (x,y) or collection of (x,y)
    sigma : float or array-like of floats of shape mu
        sigma of the gaussian
    domain : array-like, Defaults to 0->9 for x,y
        x,y domain over which this gassuain is over


    Returns
    -------
    array-like 2D
        values of the gaussian centered at mu with sigma across the (x,y) points defined in domain

    Notes:
    ------
    THIS IS IMPORTANT: MAKE SURE THE TYPES IN EACH PARAMETER ARE THE SAME!!!!
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
) -> np.ndarray:
    """
    Docstring for generate_map_from_points:
    ---------------------------
    Generates the space map from the points. 2D

    Parameters:
    -----------
    points: array-like
        points numpy array of shape (total_points,2)
    point_intensity: array-like
        intensity of the points, if None, then self.point_intensity is used.
    map: array-like
        space map, if None, then a new space map is generated.
    movie: bool
        if True, then don't add the gaussian+noise for each point. Rather add the gaussians and then to the whole add the noise.
    base_noise: float
        base noise to add to the space map
    psf_sigma: float
        sigma of the psf (pix units)


    Returns:
    --------
    1. space map as a numpy array of shape (max_x,max_x)
    2. points as a numpy array of shape (total_points,2)


    Notes:
    ------
    1. The space map is generated using get_gaussian function.
    2. For movie: In the segmented experimental images you are adding the noise of each frame to the whole subframe,
        so for this (movie=False) add each gaussian point to the image with the noise per point.
        (movie=True) add the gaussians together and then add the noise to the final image.
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
    def __init__(
        self,
        cell_space: np.ndarray | list,
        cell_axial_range: int | float,
        frame_count: int,
        exposure_time: int | float,
        interval_time: int | float,
        oversample_motion_time: int | float,
    ) -> None:
        self.cell_space = cell_space
        self.min_x = self.cell_space[0][0]
        self.max_x = self.cell_space[0][1]
        self.min_y = self.cell_space[1][0]
        self.max_y = self.cell_space[1][1]
        self.cell_axial_range = cell_axial_range
        self.space_lim = np.array(
            [
                [self.min_x, self.max_x],
                [self.min_y, self.max_y],
                [-self.cell_axial_range, self.cell_axial_range],
            ]
        )
        self.frame_count = frame_count  # count of frames
        self.exposure_time = exposure_time  # in ms
        self.interval_time = interval_time  # in ms
        self.oversample_motion_time = oversample_motion_time  # in ms
        # total time in ms is the exposure time + interval time * (frame_count) / oversample_motion_time
        # in ms
        self.total_time = self._convert_frame_to_time(self.frame_count)

    def track_generation_no_transition(
        self,
        diffusion_coefficient: float,
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
        dict-like with format: {"xy":xyz,"frames":frames,"diffusion_coefficient":diffusion_coefficient,"hurst":hurst_exponent,"initial":initial}
        """
        # initialize the fbm class
        # make self.space_lim relative to the initial position, using self.space_lim define the 0 to be initial position
        if np.shape(initials) == (2,):
            # change the shape to (3,)
            initials = np.array([initials[0], initials[1], 0])
        # subtract each element of the first dimension of self.space_lim by the first element of initials
        rel_space_lim = np.zeros((3, 2))
        for i in range(3):
            rel_space_lim[i] = self.space_lim[i] - initials[i]

        fbm = fbm_BP.FBM_BP(
            n=track_length,
            dt=1,
            hurst_parameters=[hurst_exponent],
            diffusion_parameters=[diffusion_coefficient],
            diffusion_parameter_transition_matrix=[1],
            hurst_parameter_transition_matrix=[1],
            state_probability_diffusion=[1],
            state_probability_hurst=[1],
            space_lim=rel_space_lim[0],
        )
        x = fbm.fbm()
        # repeat for y,z
        fbm.space_lim = rel_space_lim[1]
        y = fbm.fbm()
        fbm.space_lim = rel_space_lim[2]
        z = fbm.fbm()
        # convert to format [[x1,y1,z1],[x2,y2,z2],...]
        xyz = np.stack((x, y, z), axis=-1)
        # make the times starting from the starting time
        track_times = np.arange(start_time, track_length + start_time, 1)
        # add back the initial position to the track
        track_xyz = xyz + initials
        # create the dict
        track_data = {
            "xy": track_xyz,
            "frames": track_times,
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
        diffusion_parameters: np.ndarray | list,
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
        dict-like with format: {"xy":xyz,"frames":frames,"diffusion_coefficient":diffusion_coefficient,"hurst":hurst_exponent,"initial":initial}
        """
        # make self.space_lim relative to the initial position, using self.space_lim define the 0 to be initial position
        # self.space_lim is in general shape (3,2) while the initials is in shape (3,)
        # make sure the - operator is broadcasted correctly
        if np.shape(initials) == (2,):
            # change the shape to (3,)
            initials = np.array([initials[0], initials[1], 0])
        # subtract each element of the first dimension of self.space_lim by the first element of initials
        rel_space_lim = np.zeros((3, 2))
        for i in range(3):
            rel_space_lim[i] = self.space_lim[i] - initials[i]
        # initialize the fbm class
        fbm = fbm_BP.FBM_BP(
            n=track_length,
            dt=1,
            hurst_parameters=hurst_parameters,
            diffusion_parameters=diffusion_parameters,
            diffusion_parameter_transition_matrix=diffusion_transition_matrix,
            hurst_parameter_transition_matrix=hurst_transition_matrix,
            state_probability_diffusion=diffusion_state_probability,
            state_probability_hurst=hurst_state_probability,
            space_lim=rel_space_lim[0],
        )
        x = fbm.fbm()
        # repeat for y,z
        fbm.space_lim = rel_space_lim[1]
        y = fbm.fbm()
        fbm.space_lim = rel_space_lim[2]
        z = fbm.fbm()
        # convert to format [[x1,y1,z1],[x2,y2,z2],...]
        xyz = np.stack((x, y, z), axis=-1)
        # make the times starting from the starting time
        track_times = np.arange(start_time, track_length + start_time, 1)
        # add back the initial position to the track
        track_xyz = xyz + initials
        # create the dict
        track_data = {
            "xy": track_xyz,
            "frames": track_times,
            "diffusion_coefficient": fbm._diff_a_n,
            "hurst": fbm._hurst_n,
            "initial": initials,
        }
        # construct the dict
        return track_data

    def track_generation_constant(
        self, track_length: int, initials: np.ndarray, starting_time: int
    ) -> dict:
        """
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
        np.ndarray
            track data for the constant track, {"xy":xyz,"frames":frames,"diffusion_coefficient":diffusion_coefficient,"hurst":hurst_exponent,"initial":initial}
        """
        # make the times starting from the starting time
        track_times = np.arange(starting_time, track_length + starting_time, 1)
        # make the track x,y,z from the initial positions
        track_xyz = np.tile(initials, (len(track_times), 1))
        # construct the dict
        track_data = {
            "xy": track_xyz,
            "frames": track_times,
            "diffusion_coefficient": 0,
            "hurst": 0,
            "initial": initials,
        }
        return track_data

    def _convert_time_to_frame(self, time: int) -> int:
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
            (time * self.oversample_motion_time)
            / (self.exposure_time + self.interval_time)
        )

    def _convert_frame_to_time(self, frame: int) -> int:
        """
        Parameters:
        -----------
        frame : int
            frame number

        Returns:
        --------
        int: time in ms
        """
        return int(
            (frame * (self.exposure_time + self.interval_time))
            / self.oversample_motion_time
        )
