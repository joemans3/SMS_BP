"""
Simulation Configuration Module
================================

This module defines the structure and parameters required for simulating cell behavior, diffusion tracks,
and condensate dynamics within a spatial environment. The simulation configuration is described both
in terms of a JSON schema and Python dataclasses, which are used to manage and validate the input parameters
for simulations. The parameters are grouped into different categories like `Cell_Parameters`, `Track_Parameters`,
`Global_Parameters`, `Condensate_Parameters`, and `Output_Parameters`.

The simulation configuration allows for customizing aspects like track diffusion, Hurst exponents,
condensate properties, and camera settings. The module ensures that the correct units (e.g., space,
time, intensity, diffusion) are used consistently throughout the simulation, helping to provide
accurate modeling of physical phenomena.

The module follows the structure outlined in the corresponding JSON schema (`sim_config.json`), providing
an intuitive and structured way to organize simulation parameters.

Configuration Parameters
------------------------
The configuration is composed of multiple groups of parameters, each represented by a Python dataclass
and organized into the following sections:

1. **version**: `str`
    - The version of the simulation configuration file.

2. **length_unit**: `str`
    - Unit for length in the simulation (e.g., nm, um, mm).

3. **space_unit**: `str`
    - Unit for space in the simulation (e.g., pixel).

4. **time_unit**: `str`
    - Unit for time in the simulation (e.g., s, ms, us).

5. **intensity_unit**: `str`
    - Unit for intensity in the simulation (AUD only supported).

6. **diffusion_unit**: `str`
    - Unit for diffusion in the simulation (e.g., um²/s, mm²/s).

### Cell Parameters
* **cell_space**: `List[List[float]]`
    - Defines the spatial extent of the simulation space.
    - `cell_space[0]`: x coordinates of the cell space (min, max).
    - `cell_space[1]`: y coordinates of the cell space (min, max).
* **cell_axial_radius**: `float`
    - Defines the distance from `z=0` in either direction that the cell extends.
* **number_of_cells**: `int`
    - Number of cells to simulate.

### Track Parameters
* **num_tracks**: `int`
    - Number of tracks to simulate.
* **track_type**: `str`
    - Type of track to simulate (e.g., "fbm" for fractional Brownian motion).
* **track_length_mean**: `int`
    - Mean length of the track (in frames).
* **track_distribution**: `str`
    - Distribution type for track lengths (e.g., "exponential", "constant").
* **diffusion_coefficient**: `List[float]`
    - List of diffusion coefficients (in units of `diffusion_unit`).
* **diffusion_track_amount**: `List[float]`
    - Probabilities for each diffusion coefficient (must sum to 1.0).
* **hurst_exponent**: `List[float]`
    - List of Hurst exponents, defining the track behavior.
* **hurst_track_amount**: `List[float]`
    - Probabilities for each Hurst exponent (must sum to 1.0).
* **allow_transition_probability**: `bool`
    - Whether to allow transitions between different diffusion coefficients and Hurst exponents.
* **transition_matrix_time_step**: `int`
    - Time step for the transition matrices (in `time_unit`).
* **diffusion_transition_matrix**: `List[List[float]]`
    - Transition matrix for different diffusion coefficients (rows must sum to 1.0).
* **hurst_transition_matrix**: `List[List[float]]`
    - Transition matrix for different Hurst exponents (rows must sum to 1.0).
* **state_probability_diffusion**: `List[float]`
    - Initial probabilities for each diffusion coefficient state.
* **state_probability_hurst**: `List[float]`
    - Initial probabilities for each Hurst exponent state.

### Global Parameters
* **field_of_view_dim**: `List[float]`
    - Field of view dimensions in pixels.
* **frame_count**: `int`
    - Total number of frames in the simulation.
* **exposure_time**: `float`
    - Exposure time of the camera (in `time_unit`).
* **interval_time**: `float`
    - Time between frames when the camera is on (in `time_unit`).
* **oversample_motion_time**: `float`
    - Time for oversampling motion to simulate motion blur.
    - If equal to `frame_time` and `exposure_time`, there is no motion blur.
* **pixel_size**: `float`
    - Size of each pixel (in `length_unit`).
* **axial_detection_range**: `float`
    - Range for detecting molecule excitation in the z-direction (in `length_unit`).
* **base_noise**: `float`
    - Base noise of the camera (in `intensity_unit`).
* **point_intensity**: `float`
    - Intensity of a single molecule excitation (in `intensity_unit`).
* **psf_sigma**: `float`
    - Point Spread Function (PSF) sigma (in `length_unit`).
* **axial_function**: `str`
    - Function for intensity change with z (e.g., "exponential", "ones" for no effect).

### Condensate Parameters
* **initial_centers**: `List[List[float]]`
    - Initial centers of the condensates, with [x, y, z] coordinates per row.
* **initial_scale**: `List[float]`
    - Initial radius of each condensate (in `space_unit`).
* **diffusion_coefficient**: `List[float]`
    - Diffusion coefficients for each condensate (in `diffusion_unit`).
* **hurst_exponent**: `List[float]`
    - Hurst exponents for each condensate.
* **density_dif**: `float`
    - Density difference between the condensates and the rest of the cell.

### Output Parameters
* **output_path**: `str`
    - Path to the directory where the output will be saved.
* **output_name**: `str`
    - Name of the output file.
* **subsegment_type**: `str`
    - Function for projecting data across subsegments (e.g., "mean", "max", "sum").
* **subsegment_number**: `int`
    - Number of subsegments to divide the simulation frames into. The total number of frames must be divisible by the subsegment number.

Dataclasses
-----------
This module defines Python dataclasses to represent each section of the simulation configuration. The dataclasses
allow for structured data management, type-checking, and easy manipulation of simulation parameters.

1. **CellParameters**:
    - Stores the cell space, axial radius, and number of cells.

2. **TrackParameters**:
    - Stores track-related parameters such as number of tracks, track length, diffusion coefficients, and Hurst exponents.

3. **GlobalParameters**:
    - Stores global simulation settings such as field of view, frame count, and camera parameters.

4. **CondensateParameters**:
    - Stores parameters related to condensates, including their initial centers, scales, and diffusion coefficients.

5. **OutputParameters**:
    - Stores output-related settings such as file path, file name, and subsegment configurations.

6. **SimulationConfig**:
    - The main class that contains all the parameters for the simulation. Inherits from a base class to facilitate
      conversion of lists to NumPy arrays for efficient mathematical operations.
"""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, List

import numpy as np

# JSON Schema definition
schema = {
    "type": "object",
    "properties": {
        "version": {"type": "string"},
        "length_unit": {"type": "string"},
        "space_unit": {"type": "string"},
        "time_unit": {"type": "string"},
        "intensity_unit": {"type": "string"},
        "diffusion_unit": {"type": "string"},
        "Cell_Parameters": {
            "type": "object",
            "properties": {
                "cell_space": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "minItems": 2,
                    "maxItems": 2,
                },
                "cell_axial_radius": {"type": "number"},
                "number_of_cells": {"type": "integer"},
            },
            "required": ["cell_space", "cell_axial_radius", "number_of_cells"],
        },
        "Track_Parameters": {
            "type": "object",
            "properties": {
                "num_tracks": {"type": "integer"},
                "track_type": {"type": "string"},
                "track_length_mean": {"type": "integer"},
                "track_distribution": {"type": "string"},
                "diffusion_coefficient": {"type": "array", "items": {"type": "number"}},
                "diffusion_track_amount": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "hurst_exponent": {"type": "array", "items": {"type": "number"}},
                "hurst_track_amount": {"type": "array", "items": {"type": "number"}},
                "allow_transition_probability": {"type": "boolean"},
                "transition_matrix_time_step": {"type": "integer"},
                "diffusion_transition_matrix": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                "hurst_transition_matrix": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                "state_probability_diffusion": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "state_probability_hurst": {
                    "type": "array",
                    "items": {"type": "number"},
                },
            },
            "required": [
                "num_tracks",
                "track_type",
                "track_length_mean",
                "track_distribution",
                "diffusion_coefficient",
                "hurst_exponent",
                "allow_transition_probability",
            ],
        },
        "Global_Parameters": {
            "type": "object",
            "properties": {
                "field_of_view_dim": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "frame_count": {"type": "integer"},
                "exposure_time": {"type": "number"},
                "interval_time": {"type": "number"},
                "oversample_motion_time": {"type": "number"},
                "pixel_size": {"type": "number"},
                "axial_detection_range": {"type": "number"},
                "base_noise": {"type": "number"},
                "point_intensity": {"type": "number"},
                "psf_sigma": {"type": "number"},
                "axial_function": {"type": "string"},
            },
            "required": [
                "field_of_view_dim",
                "frame_count",
                "exposure_time",
                "interval_time",
                "oversample_motion_time",
                "pixel_size",
                "axial_detection_range",
                "base_noise",
                "point_intensity",
                "psf_sigma",
                "axial_function",
            ],
        },
        "Condensate_Parameters": {
            "type": "object",
            "properties": {
                "initial_centers": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                },
                "initial_scale": {"type": "array", "items": {"type": "number"}},
                "diffusion_coefficient": {"type": "array", "items": {"type": "number"}},
                "hurst_exponent": {"type": "array", "items": {"type": "number"}},
                "density_dif": {"type": "number"},
            },
            "required": [
                "initial_centers",
                "initial_scale",
                "diffusion_coefficient",
                "hurst_exponent",
                "density_dif",
            ],
        },
        "Output_Parameters": {
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "output_name": {"type": "string"},
                "subsegment_type": {"type": "string"},
                "subsegment_number": {"type": "integer"},
            },
            "required": [
                "output_path",
                "output_name",
                "subsegment_type",
                "subsegment_number",
            ],
        },
    },
    "required": [
        "version",
        "length_unit",
        "space_unit",
        "time_unit",
        "intensity_unit",
        "diffusion_unit",
        "Cell_Parameters",
        "Track_Parameters",
        "Global_Parameters",
        "Condensate_Parameters",
        "Output_Parameters",
    ],
}


# Dataclass definitions
@dataclass
class CellParameters:
    cell_space: List[List[float]]
    cell_axial_radius: float
    number_of_cells: int


@dataclass
class TrackParameters:
    num_tracks: int
    track_type: str
    track_length_mean: int
    track_distribution: str
    diffusion_coefficient: List[float]
    diffusion_track_amount: List[float]
    hurst_exponent: List[float]
    hurst_track_amount: List[float]
    allow_transition_probability: bool
    transition_matrix_time_step: int
    diffusion_transition_matrix: List[List[float]]
    hurst_transition_matrix: List[List[float]]
    state_probability_diffusion: List[float]
    state_probability_hurst: List[float]


@dataclass
class GlobalParameters:
    field_of_view_dim: List[float]
    frame_count: int
    exposure_time: float
    interval_time: float
    oversample_motion_time: float
    pixel_size: float
    axial_detection_range: float
    base_noise: float
    point_intensity: float
    psf_sigma: float
    axial_function: str


@dataclass
class CondensateParameters:
    initial_centers: List[List[float]]
    initial_scale: List[float]
    diffusion_coefficient: List[float]
    hurst_exponent: List[float]
    density_dif: float


@dataclass
class OutputParameters:
    output_path: str
    output_name: str
    subsegment_type: str
    subsegment_number: int


@dataclass
class ABCConfig:
    def make_array(self):
        """
        Recursively converts all list attributes to numpy arrays, including nested dataclasses.
        """

        def _convert(obj: Any):
            if isinstance(obj, list):
                return np.array(obj)
            elif is_dataclass(obj):
                # Traverse nested dataclass
                for field in fields(obj):
                    value = getattr(obj, field.name)
                    setattr(obj, field.name, _convert(value))
            return obj

        # Apply conversion to the main dataclass
        for field in fields(self):
            value = getattr(self, field.name)
            setattr(self, field.name, _convert(value))


@dataclass
class SimulationConfig(ABCConfig):
    version: str
    length_unit: str
    space_unit: str
    time_unit: str
    intensity_unit: str
    diffusion_unit: str
    Cell_Parameters: CellParameters
    Track_Parameters: TrackParameters
    Global_Parameters: GlobalParameters
    Condensate_Parameters: CondensateParameters
    Output_Parameters: OutputParameters
