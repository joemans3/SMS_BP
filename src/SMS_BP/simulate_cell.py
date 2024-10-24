import json
import os
import pickle
import random

import numpy as np
import skimage as skimage
from PIL import Image
from scipy.linalg import fractional_matrix_power

import SMS_BP.decorators as decorators
import SMS_BP.probability_functions as pf
import SMS_BP.simulate_foci as sf
from SMS_BP.config_schema import SimulationConfig
from SMS_BP.json_validator_converter import (
    load_validate_and_convert,
    validate_and_convert,
)


def save_tiff(image: np.ndarray, path: str, img_name: str | None = None) -> None:
    """
    Save the image as a TIFF file.

    Parameters:
    -----------
    image : np.ndarray
        Image to be saved.
    path : str
        Path where the image will be saved.
    img_name : str, optional
        Name of the image file (without extension), by default None.

    Returns:
    --------
    None
    """
    if img_name is None:
        skimage.io.imsave(path, image)
    else:
        skimage.io.imsave(os.path.join(path, img_name + ".tiff"), image)
    return


# function to perform the subsegmentation
def sub_segment(
    img: np.ndarray,
    subsegment_num: int,
    img_name: str | None = None,
    subsegment_type: str = "mean",
) -> list[np.ndarray]:
    """
    Perform subsegmentation on the image.

    Parameters:
    -----------
    img : np.ndarray
        Image to be subsegmented.
    subsegment_num : int
        Number of subsegments to be created.
    img_name : str, optional
        Name of the image, by default None.
    subsegment_type : str, optional
        Type of subsegmentation to be performed. Options are "mean", "max", "std". Default is "mean".

    Returns:
    --------
    list[np.ndarray]
        List of subsegmented images.

    Raises:
    -------
    ValueError
        If the subsegment type is not supported.
    """
    supported_subsegment_types = ["mean", "max", "std"]
    if subsegment_type not in supported_subsegment_types:
        raise ValueError(
            f"Subsegment type {subsegment_type} is not supported. Supported types are {supported_subsegment_types}"
        )
    # get the dimensions of the image
    dims = img.shape
    # get the number of frames
    num_frames = dims[0]
    # find the number of frames per subsegment
    frames_per_subsegment = int(num_frames / subsegment_num)
    hold_img = []
    for j in np.arange(subsegment_num):
        if subsegment_type == "mean":
            hold_img.append(
                np.mean(
                    img[
                        int(j * frames_per_subsegment) : int(
                            (j + 1) * frames_per_subsegment
                        )
                    ],
                    axis=0,
                )
            )
        elif subsegment_type == "max":
            hold_img.append(
                np.max(
                    img[
                        int(j * frames_per_subsegment) : int(
                            (j + 1) * frames_per_subsegment
                        )
                    ],
                    axis=0,
                )
            )
        elif subsegment_type == "std":
            hold_img.append(
                np.std(
                    img[
                        int(j * frames_per_subsegment) : int(
                            (j + 1) * frames_per_subsegment
                        )
                    ],
                    axis=0,
                )
            )
    return hold_img


def make_directory_structure(
    cd: str,
    img_name: str,
    img: np.ndarray,
    subsegment_type: str,
    subsegment_num: int,
    **kwargs,
) -> list[np.ndarray]:
    """
    Create the directory structure for the simulation, save the image, and perform subsegmentation.

    Parameters:
    -----------
    cd : str
        Directory where the simulation will be saved.
    img_name : str
        Name of the image.
    img : np.ndarray
        Image to be subsegmented.
    subsegment_type : str
        Type of subsegmentation to be performed.
    subsegment_num : int
        Number of subsegments to be created.
    **kwargs : dict
        Additional keyword arguments, including:
        - data : dict (optional)
            Dictionary of data to be saved, keys are "map", "tracks", "points_per_frame".
        - parameters : dict (optional)
            Parameters of the simulation to be saved.

    Returns:
    --------
    list[np.ndarray]
        List of subsegmented images.

    Raises:
    -------
    None
    """
    # make the directory if it does not exist
    if not os.path.exists(cd):
        os.makedirs(cd)
    # track_pickle
    track_pickle = os.path.join(cd, "Track_dump.pkl")
    # params_pickle
    params_pickle = os.path.join(cd, "params_dump.pkl")
    # params_json
    params_json = os.path.join(cd, "params_dump.json")

    # saves the data if it is passed as a keyword argument (map,tracks,points_per_frame)
    with open(track_pickle, "wb+") as f:
        pickle.dump(kwargs.get("data", {}), f)
    # saves the parameters used to generate the simulation
    with open(params_pickle, "wb+") as f:
        pickle.dump(kwargs.get("parameters", {}), f)

    # in this directory, dump the parameters into a json file
    with open(params_json, "w") as f:
        # dump the parameters into a json file
        # json.dump(convert_arrays_to_lists(kwargs.get("parameters", {})), f)
        json.dump({}, f)

    # make a diretory inside cd called Analysis if it does not exist
    if not os.path.exists(os.path.join(cd, "Analysis")):
        os.makedirs(os.path.join(cd, "Analysis"))
    # save the img file with its name in the cd directory
    save_tiff(img, cd, img_name=img_name)
    # make a directory inside cd called segmented if it does not exist
    if not os.path.exists(os.path.join(cd, "segmented")):
        os.makedirs(os.path.join(cd, "segmented"))
    # perform subsegmentation on the image
    hold_img = sub_segment(
        img, subsegment_num, img_name=img_name, subsegment_type=subsegment_type
    )
    # create the names for the subsegmented images
    hold_name = []
    for i in np.arange(subsegment_num):
        hold_name.append(
            os.path.join(cd, "segmented", str(int(i) + 1) + "_" + img_name + ".tif")
        )
    # save the subsegmented images
    for i in np.arange(subsegment_num):
        img = Image.fromarray(hold_img[i])
        img.save(hold_name[i])
    return hold_img


# Function to recursively convert lists to NumPy arrays
def convert_lists_to_arrays(obj: list | dict) -> np.ndarray | dict:
    """
    Recursively convert lists to NumPy arrays.

    Parameters:
    -----------
    obj : list | dict
        Object to be converted.

    Returns:
    --------
    np.ndarray | dict
        Converted object with lists replaced by NumPy arrays.
    """
    if isinstance(obj, list):
        return np.array(obj)
    elif isinstance(obj, dict):
        return {k: convert_lists_to_arrays(v) for k, v in obj.items()}
    else:
        return obj


def convert_arrays_to_lists(obj: np.ndarray | dict) -> list | dict:
    """
    Recursively convert NumPy arrays to lists.

    Parameters:
    -----------
    obj : np.ndarray | dict
        Object to be converted.

    Returns:
    --------
    list | dict
        Converted object with NumPy arrays replaced by lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_arrays_to_lists(v) for k, v in obj.items()}
    else:
        return obj


class Simulate_cells:
    def __init__(self, init_dict_json: dict | str | SimulationConfig):
        """Docstring for Simulate_cells: Class for simulating cells in space.
        Parameters:
        -----------
        init_dict_json : dict|str
            dictionary of parameters or path to the json file containing the parameters
            see sim_config.md for more details
        Returns:
        --------
        None
        """
        if isinstance(init_dict_json, str):
            self.simulation_config = load_validate_and_convert(init_dict_json)
        elif isinstance(init_dict_json, dict):
            self.simulation_config = validate_and_convert(init_dict_json)
        else:
            self.simulation_config = init_dict_json

        self.simulation_config.make_array()
        # store the times
        self.frame_count = self.simulation_config.Global_Parameters.frame_count
        self.interval_time = int(self.simulation_config.Global_Parameters.interval_time)
        self.oversample_motion_time = int(
            self.simulation_config.Global_Parameters.oversample_motion_time
        )
        self.exposure_time = int(self.simulation_config.Global_Parameters.exposure_time)
        self.total_time = self._convert_frame_to_time(
            self.frame_count,
            self.exposure_time,
            self.interval_time,
            self.oversample_motion_time,
        )
        # convert the track_length_mean from frame to time
        self.track_length_mean = self._convert_frame_to_time(
            self.simulation_config.Track_Parameters.track_length_mean,
            self.exposure_time,
            self.interval_time,
            self.oversample_motion_time,
        )

        # update the diffusion coefficients from um^2/s to pix^2/ms
        self.track_diffusion_updated = self._update_units(
            self.simulation_config.Track_Parameters.diffusion_coefficient,
            "um^2/s",
            "pix^2/(oversample_motion_time)ms)",
        )
        self.condensate_diffusion_updated = self._update_units(
            self.simulation_config.Condensate_Parameters.diffusion_coefficient,
            "um^2/s",
            "pix^2/(oversample_motion_time)ms)",
        )
        # update the pixel_size,axial_detection_range,psf_sigma from um to pix
        self.pixel_size_pix = self._update_units(
            self.simulation_config.Global_Parameters.pixel_size, "um", "pix"
        )
        self.axial_detection_range_pix = self._update_units(
            self.simulation_config.Global_Parameters.axial_detection_range, "um", "pix"
        )
        self.psf_sigma_pix = self._update_units(
            self.simulation_config.Global_Parameters.psf_sigma, "um", "pix"
        )

        if self.simulation_config.Track_Parameters.allow_transition_probability:
            # convert the transition matrix from the time given to the oversample_motion_time
            # store the transition_matrix_time_step
            self.transition_matrix_time_step = (
                self.simulation_config.Track_Parameters.transition_matrix_time_step
            )

            # check if the diffusion_coefficient and hurst_exponent are of length n, and then check if the length of the transition matrix is the same as the length of the diffusion_coefficient and hurst_exponent
            if len(
                self.simulation_config.Track_Parameters.diffusion_coefficient
            ) != len(
                self.simulation_config.Track_Parameters.diffusion_transition_matrix
            ):
                raise ValueError(
                    "The length of the diffusion_coefficient and the diffusion_transition_matrix are not the same"
                )
            if len(self.simulation_config.Track_Parameters.hurst_exponent) != len(
                self.simulation_config.Track_Parameters.hurst_transition_matrix
            ):
                raise ValueError(
                    "The length of the hurst_exponent and the hurst_transition_matrix are not the same"
                )
            # compare to the oversample_motion_time and scale to the appropriate time step
            if len(self.simulation_config.Track_Parameters.diffusion_coefficient) != 1:
                self.diffusion_transition_matrix = np.real(
                    fractional_matrix_power(
                        self.simulation_config.Track_Parameters.diffusion_transition_matrix,
                        self.oversample_motion_time / self.transition_matrix_time_step,
                    )
                )
            else:
                self.diffusion_transition_matrix = (
                    self.simulation_config.Track_Parameters.diffusion_transition_matrix
                )
            if len(self.simulation_config.Track_Parameters.hurst_exponent) != 1:
                self.hurst_transition_matrix = np.real(
                    fractional_matrix_power(
                        self.simulation_config.Track_Parameters.hurst_transition_matrix,
                        self.oversample_motion_time / self.transition_matrix_time_step,
                    )
                )
            else:
                self.hurst_transition_matrix = (
                    self.simulation_config.Track_Parameters.hurst_transition_matrix
                )
        return

    def _convert_frame_to_time(
        self,
        frame: int,
        exposure_time: int,
        interval_time: int,
        oversample_motion_time: int,
    ) -> int:
        """Docstring for _convert_frame_to_time: convert the frame number to time
        Parameters:
        -----------
        frame : int
            frame number
        exposure_time : int
            exposure time
        interval_time : int
            interval time
        oversample_motion_time : int
            oversample motion time
        Returns:
        --------
        int
            time in the appropriate units
        """
        return int((frame * (exposure_time + interval_time)) / oversample_motion_time)

    def _update_units(
        self, unit: np.ndarray | float | int, orig_type: str, update_type: str
    ) -> float | np.ndarray | None:
        """Docstring for _update_units: update the unit from one type to another
        Parameters:
        -----------
        unit : float|np.ndarray
            unit to be updated
        orig_type : str
            original type of unit
        update_type : str
            type to update unit to
        """
        unit = np.array(unit)
        if orig_type == "nm":
            if update_type == "pix":
                return unit / self.simulation_config.Global_Parameters.pixel_size
        elif orig_type == "pix":
            if update_type == "nm":
                return unit * self.simulation_config.Global_Parameters.pixel_size
        elif orig_type == "ms":
            if update_type == "s":
                return unit / 1000.0
        elif orig_type == "s":
            if update_type == "ms":
                return unit * 1000.0
        elif orig_type == "um^2/s":
            if update_type == "pix^2/(oversample_motion_time)ms)":
                return (
                    unit
                    * (1.0 / (self.simulation_config.Global_Parameters.pixel_size**2))
                    * (
                        self.simulation_config.Global_Parameters.oversample_motion_time
                        / 1000.0
                    )
                )
        if orig_type == "um":
            if update_type == "pix":
                return unit / self.simulation_config.Global_Parameters.pixel_size

    def _check_init_dict(self) -> bool:
        """Docstring for _check_init_dict: check the init_dict for the required keys, and if they are consistent with other keys

        Parameters:
        -----------
        None

        Returns:
        --------
        bool: True if the init_dict is correct

        Raises:
        -------
        InitializationKeys: if the init_dict does not have the required keys
        InitializationValues: if the init_dict values are not consistent with each other
        """
        # check if the init_dict has the required keys
        # TODO
        return True

    def _read_json(self, json_file: str) -> dict:
        """Docstring for _read_json: read the json file and return the dictionary
        Parameters:
        -----------
        json_file : str
            path to the json file

        Returns:
        --------
        dict
            dictionary of parameters
        """
        # Open the json file
        with open(json_file) as f:
            # Load the json file
            data = json.load(f)
        # Function to recursively convert lists to NumPy arrays

        def convert_lists_to_arrays(obj):
            if isinstance(obj, list):
                return np.array(obj)
            elif isinstance(obj, dict):
                return {k: convert_lists_to_arrays(v) for k, v in obj.items()}
            else:
                return obj

        # Convert lists to NumPy arrays
        data = convert_lists_to_arrays(data)
        return data

    def _define_space(
        self, dims: tuple[int, int] = (100, 100), movie_frames: int = 500
    ) -> np.ndarray:
        """Docstring for _define_space: make the empty space for simulation
        Parameters:
        -----------
        dims : tuple, Default = (100,100)
            dimensions of the space to be simulated
        movie_frames : int, Default = 500
            number of frames to be simulated
        Returns:
        --------
        space : array-like, shape = (movie_frames,dims[0],dims[1])
            empty space for simulation
        """
        space = np.zeros((movie_frames, dims[0], dims[1]))
        return space

    def _convert_track_dict_points_per_frame(
        self, tracks: dict, movie_frames: int
    ) -> dict:
        """
        Convert the track dictionary into a dictionary of points per frame.

        Parameters:
        -----------
        tracks : dict
            Dictionary of tracks where keys are track numbers and values are track data.
        movie_frames : int
            Number of frames in the movie.

        Returns:
        --------
        dict
            Dictionary where keys are frame numbers and values are lists of (x, y, z) tuples representing points.
        """
        points_per_frame = dict(
            zip(
                [str(i) for i in range(movie_frames)], [[] for i in range(movie_frames)]
            )
        )
        for i, j in tracks.items():
            for k in range(len(j["frames"])):
                points_per_frame[str(j["frames"][k])].append(j["xy"][k])

        return points_per_frame

    def _convert_track_dict_msd(self, tracks: dict) -> dict:
        """
        Convert the track dictionary to a format required for the MSD (mean square displacement) function.

        Parameters:
        -----------
        tracks : dict
            Dictionary of tracks where keys are track numbers and values are track data.

        Returns:
        --------
        dict
            Dictionary where keys are track numbers and values are lists of (x, y, T) tuples representing track points.
        """
        track_msd = {}
        for i, j in tracks.items():
            # make a (x,y,T) tuple for each point
            track_msd[i] = []
            for k in range(len(j["xy"])):
                track_msd[i].append((j["xy"][k][0], j["xy"][k][1], j["frames"][k]))
            # add this to the dictionary
            track_msd[i] = np.array(track_msd[i])
        return track_msd

    def _create_track_pop_dict(self, simulation_cube: np.ndarray) -> tuple[dict, dict]:
        """
        Create the tracks for the cell simulation and return tracks and points per time.

        Parameters:
        -----------
        simulation_cube : np.ndarray
            Empty space for the simulation.

        Returns:
        --------
        tuple[dict, dict]
            A tuple where the first element is a dictionary of tracks, and the second element is a dictionary of points per time.
        """
        # get the lengths of the tracks given a distribution
        track_lengths = sf.get_lengths(
            track_distribution=self.simulation_config.Track_Parameters.track_distribution,
            track_length_mean=self.track_length_mean,
            total_tracks=self.simulation_config.Track_Parameters.num_tracks,
        )
        # if track_lengths is larger than the number of frames then set that to the number of frames -1
        track_lengths = np.array(
            [i if i < self.total_time else self.total_time - 1 for i in track_lengths]
        )
        # for each track_lengths find the starting frame
        starting_frames = np.array(
            [random.randint(0, self.total_time - i) for i in track_lengths]
        )

        # initialize the Condensates. Assuming box shaped.
        # find area assuming cell_space is [[min_x,max_x],[min_y,max_y]]
        vol_cell = (
            np.abs(np.diff(self.simulation_config.Cell_Parameters.cell_space[0]))
            * np.abs(np.diff(self.simulation_config.Cell_Parameters.cell_space[1]))
            * 2.0
            * self.simulation_config.Cell_Parameters.cell_axial_radius
        )

        self.condensates = sf.create_condensate_dict(
            initial_centers=np.array(
                self.simulation_config.Condensate_Parameters.initial_centers
            ),
            initial_scale=np.array(
                self.simulation_config.Condensate_Parameters.initial_scale
            ),
            diffusion_coefficient=np.array(self.condensate_diffusion_updated),
            hurst_exponent=np.array(
                self.simulation_config.Condensate_Parameters.hurst_exponent
            ),
            units_time=np.array(
                [
                    str(self.simulation_config.Global_Parameters.oversample_motion_time)
                    + self.simulation_config.time_unit
                ]
                * len(self.condensate_diffusion_updated)
            ),
            cell_space=self.simulation_config.Cell_Parameters.cell_space,
            cell_axial_range=self.simulation_config.Cell_Parameters.cell_axial_radius,
        )

        # define the top_hat class that will be used to sample the condensates
        top_hat_func = pf.multiple_top_hat_probability(
            num_subspace=len(self.condensate_diffusion_updated),
            subspace_centers=self.simulation_config.Condensate_Parameters.initial_centers,
            subspace_radius=self.simulation_config.Condensate_Parameters.initial_scale,
            density_dif=self.simulation_config.Condensate_Parameters.density_dif,
            space_size=np.array(vol_cell),
        )
        # make a placeholder for the initial position array with all 0s
        initials = np.zeros((self.simulation_config.Track_Parameters.num_tracks, 3))
        # lets use the starting frames to find the inital position based on the position of the condensates
        for i in range(self.simulation_config.Track_Parameters.num_tracks):
            # get the starting time from the frame, oversample_motion_time, and interval_time
            starting_frame = starting_frames[i]
            # condensate positions
            condensate_positions = np.zeros((len(self.condensates), 3))
            # loop through the condensates
            for ID, cond in self.condensates.items():
                condensate_positions[int(ID)] = cond(
                    int(starting_frame),
                    str(self.simulation_config.Global_Parameters.oversample_motion_time)
                    + self.simulation_config.time_unit,
                )["Position"]
            # update the top_hat_func with the new condensate positions
            top_hat_func.update_parameters(subspace_centers=condensate_positions)
            # sample the top hat to get the initial position
            initials[i] = sf.generate_points_from_cls(
                top_hat_func,
                total_points=1,
                min_x=self.simulation_config.Cell_Parameters.cell_space[0][0],
                max_x=self.simulation_config.Cell_Parameters.cell_space[0][1],
                min_y=self.simulation_config.Cell_Parameters.cell_space[1][0],
                max_y=self.simulation_config.Cell_Parameters.cell_space[1][1],
                min_z=-self.simulation_config.Cell_Parameters.cell_axial_radius,
                max_z=self.simulation_config.Cell_Parameters.cell_axial_radius,
                density_dif=self.simulation_config.Condensate_Parameters.density_dif,
            )[0]
        # check to see if there is 2 or 3 values in the second dimension of initials
        if initials.shape[1] == 2:
            # add a third dimension of zeros so that the final shape is (num_tracks,3) with (:,3) being 0s
            initials = np.hstack(
                (
                    initials,
                    np.zeros((self.simulation_config.Track_Parameters.num_tracks, 1)),
                )
            )
        # create tracks
        tracks = {}
        points_per_time = dict(
            zip(
                [str(i) for i in range(int(self.total_time))],
                [[] for i in range(int(self.total_time))],
            )
        )
        # initialize the Track_generator class
        track_generator = sf.Track_generator(
            cell_space=self.simulation_config.Cell_Parameters.cell_space,
            cell_axial_range=self.simulation_config.Cell_Parameters.cell_axial_radius,
            frame_count=self.frame_count,
            exposure_time=self.exposure_time,
            interval_time=self.interval_time,
            oversample_motion_time=self.oversample_motion_time,
        )
        if self.simulation_config.Track_Parameters.track_type == "constant":
            for i in range(self.simulation_config.Track_Parameters.num_tracks):
                # make a constant track
                tracks[i] = track_generator.track_generation_constant(
                    track_length=track_lengths[i],
                    initials=initials[i],
                    starting_time=starting_frames[i],
                )
                # add the number of points per frame to the dictionary
                for j in range(len(tracks[i]["frames"])):
                    points_per_time[str(int(tracks[i]["frames"][j]))].append(
                        tracks[i]["xy"][j]
                    )
        elif not self.simulation_config.Track_Parameters.allow_transition_probability:
            # for the amount of tracks make a choice from the diffusion and hurst parameters based on the probability from diffusion_track_amount, hurst_track_amount
            # make an index of the track_diffusion_updated and hurst_exponent
            index_diffusion = np.arange(len(self.track_diffusion_updated))
            index_hurst = np.arange(
                len(self.simulation_config.Track_Parameters.hurst_exponent)
            )
            track_diffusion_choice = np.random.choice(
                index_diffusion,
                size=self.simulation_config.Track_Parameters.num_tracks,
                p=self.simulation_config.Track_Parameters.diffusion_track_amount,
            )
            track_hurst_choice = np.random.choice(
                index_hurst,
                size=self.simulation_config.Track_Parameters.num_tracks,
                p=self.simulation_config.Track_Parameters.hurst_track_amount,
            )
            for i in range(self.simulation_config.Track_Parameters.num_tracks):
                tracks_diffusion = self.track_diffusion_updated[
                    track_diffusion_choice[i]
                ]
                tracks_hurst = self.simulation_config.Track_Parameters.hurst_exponent[
                    track_hurst_choice[i]
                ]
                # make a track with no transition probability
                tracks[i] = track_generator.track_generation_no_transition(
                    diffusion_coefficient=tracks_diffusion,
                    hurst_exponent=tracks_hurst,
                    track_length=track_lengths[i],
                    initials=initials[i],
                    start_time=starting_frames[i],
                )
                # add the number of points per frame to the dictionary
                for j in range(len(tracks[i]["frames"])):
                    points_per_time[str(int(tracks[i]["frames"][j]))].append(
                        tracks[i]["xy"][j]
                    )
        elif self.simulation_config.Track_Parameters.allow_transition_probability:
            for i in range(self.simulation_config.Track_Parameters.num_tracks):
                # make a track with transition probability
                tracks[i] = track_generator.track_generation_with_transition(
                    diffusion_transition_matrix=self.diffusion_transition_matrix,
                    hurst_transition_matrix=self.hurst_transition_matrix,
                    diffusion_parameters=self.track_diffusion_updated,
                    hurst_parameters=self.simulation_config.Track_Parameters.hurst_exponent,
                    diffusion_state_probability=self.simulation_config.Track_Parameters.state_probability_diffusion,
                    hurst_state_probability=self.simulation_config.Track_Parameters.state_probability_hurst,
                    track_length=track_lengths[i],
                    initials=initials[i],
                    start_time=starting_frames[i],
                )
                for j in range(len(tracks[i]["frames"])):
                    points_per_time[str(int(tracks[i]["frames"][j]))].append(
                        tracks[i]["xy"][j]
                    )
        return tracks, points_per_time

    def _create_map(
        self, initial_map: np.ndarray, points_per_frame: dict, axial_function: str
    ) -> np.ndarray:
        """
        Create the simulation map from points per frame.

        Parameters:
        -----------
        initial_map : np.ndarray
            Empty space for the simulation.
        points_per_frame : dict
            Dictionary of points per frame, where keys are frame numbers and values are point coordinates.
        axial_function : str
            The function used to generate axial intensity.

        Returns:
        --------
        np.ndarray
            Updated simulation map.
        """
        for i in range(initial_map.shape[0]):
            # if empty points_per_frame for frame i then do some house keeping
            if len(points_per_frame[str(i)]) == 0:
                abs_axial_position = (
                    1.0
                    * self.simulation_config.Global_Parameters.point_intensity
                    * self.oversample_motion_time
                    / self.exposure_time
                )
                points_per_frame_xyz = np.array(points_per_frame[str(i)])
                points_per_frame_xyz = np.array(points_per_frame_xyz)
            else:
                abs_axial_position = (
                    1.0
                    * self.simulation_config.Global_Parameters.point_intensity
                    * sf.axial_intensity_factor(
                        np.abs(np.array(points_per_frame[str(i)])[:, 2]),
                        detection_range=self.axial_detection_range_pix,
                        func=self.simulation_config.Global_Parameters.axial_function,
                    )
                    * self.oversample_motion_time
                    / self.exposure_time
                )
                points_per_frame_xyz = np.array(points_per_frame[str(i)])[:, :2]
            initial_map[i], _ = sf.generate_map_from_points(
                points_per_frame_xyz,
                point_intensity=abs_axial_position,
                map=initial_map[i],
                movie=True,
                base_noise=self.simulation_config.Global_Parameters.base_noise,
                psf_sigma=self.psf_sigma_pix,
            )
        return initial_map

        pass

    def _point_per_time_selection(self, points_per_time: dict) -> dict:
        """
        Select points per frame for the simulation, considering only points during the exposure time.

        Parameters:
        -----------
        points_per_time : dict
            Dictionary of points per time, where keys are frame numbers and values are lists of points.

        Returns:
        --------
        dict
            Dictionary of points per frame, filtered by exposure time.
        """
        # The tracks and points_per_time are already created, so we just need to convert the points_per_time to points_per_frame
        # we only select the points which are in every exposure time ignoring the interval time inbetween the exposure time
        points_per_frame = dict(
            zip(
                [str(i) for i in range(self.frame_count)],
                [[] for i in range(self.frame_count)],
            )
        )
        exposure_counter = 0
        interval_counter = 0
        frame_counter = 0
        for i in range(int(self.total_time)):
            if (
                exposure_counter < int(self.exposure_time / self.oversample_motion_time)
            ) and (
                interval_counter
                <= int(self.interval_time / self.oversample_motion_time)
            ):
                # append the points to the points_per_frame
                if len(points_per_time[str(i)]) != 0:
                    for k in range(len(points_per_time[str(i)])):
                        points_per_frame[str(frame_counter)].append(
                            points_per_time[str(i)][k]
                        )
                # increment the exposure_counter
                exposure_counter += 1
            elif (
                exposure_counter
                == int(self.exposure_time / self.oversample_motion_time)
            ) and (
                interval_counter < int(self.interval_time / self.oversample_motion_time)
            ):
                # increment the interval_counter
                interval_counter += 1
            if (
                exposure_counter
                == int(self.exposure_time / self.oversample_motion_time)
            ) and (
                interval_counter
                == int(self.interval_time / self.oversample_motion_time)
            ):
                # reset the counters
                exposure_counter = 0
                interval_counter = 0
                frame_counter += 1
        return points_per_frame

    def get_cell(self) -> dict:
        """Docstring for get_cell: get the cell simulation
        Parameters:
        -----------
        None
        Returns:
        --------
        cell : dict
            dictionary of the cell simulation, keys = "map","tracks","points_per_frame"
        """
        # create the space for the simulation
        space = self._define_space(
            dims=self.simulation_config.Global_Parameters.field_of_view_dim,
            movie_frames=self.frame_count,
        )
        # create the tracks and points_per_time
        tracks, points_per_time = self._create_track_pop_dict(space)
        points_per_frame = self._point_per_time_selection(points_per_time)

        # update the space
        space_updated = self._create_map(
            initial_map=space,
            points_per_frame=points_per_frame,
            axial_function=self.simulation_config.Global_Parameters.axial_function,
        )
        return {
            "map": space_updated,
            "tracks": tracks,
            "points_per_frame": points_per_frame,
        }

    def get_and_save_sim(
        self,
        cd: str,
        img_name: str,
        subsegment_type: str,
        subsegment_num: int,
        **kwargs,
    ) -> None:
        """Docstring for make_directory_structure: make the directory structure for the simulation and save the image + the data and parameters
        Also perform the subsegmentation and save the subsegments in the appropriate directory
        Parameters:
        -----------
        cd : str
            directory to save the simulation
        img_name : str
            name of the image
        img : array-like
            image to be subsegmented
        subsegment_type : str
            type of subsegmentation to be performed, currently only "mean" is supported
        subsegment_num : int
            number of subsegments to be created
        **kwargs : dict
            dictionary of keyword arguments
        KWARGS:
        -------
        data : dict, Default = None
            dictionary of data to be saved, Keys = "map","tracks","points_per_frame" Values = array-like.
            See the return of the function simulate_cell_tracks for more details
        parameters : dict, Default = self.simulation_config
        Returns:
        --------
        none
        """
        # run the sim
        sim = self.get_cell()
        # update the kwargs with the data
        kwargs["data"] = sim
        kwargs["parameters"] = self.simulation_config
        # make the directory structure
        _ = make_directory_structure(
            cd, img_name, sim["map"], subsegment_type, subsegment_num, **kwargs
        )
        return None

    @property
    def condensates(self) -> dict:
        return self._condensates

    @condensates.setter
    def condensates(self, condensates: dict):
        self._condensates = condensates

    @decorators.deprecated(
        "This function is not useful, but is still here for a while in case I need it later"
    )
    def _format_points_per_frame(self, points_per_frame):
        """
        Docstring for _format_points_per_frame: format the points per frame dictionary so that for each key the set of tracks in it are
        converted to a numpy array of N x 2 where N is the total amount of points in that frame. You don't need this function.

        Parameters:
        -----------
        points_per_frame : dict
            keys = str(i) for i in range(self.total_time), values = list of tracks, which are collections of [x,y] coordinates

        Returns:
        --------
        points_per_frame : dict
            keys = str(i) for i in range(movie_frames), values = numpy array of N x 2 where N is the total amount of points in that frame

        """
        for i in points_per_frame.keys():
            # each value is a list of K lists that are composed of M x 2 arrays where M can be different for each list
            # we want to convert this to a numpy array of N x 2 where N is the total amount of points in that frame
            point_holder = []
            for j in points_per_frame[i]:
                point_holder.append(j)
            points_per_frame[i] = np.array(point_holder)
        return points_per_frame
