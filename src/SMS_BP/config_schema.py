import json
from dataclasses import dataclass
from typing import List, Tuple, Union
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
        # walk along walkable attributes and make a array
        # list of att:
        for att in self.__dict__:
            if isinstance(self.__dict__[att], list):
                self.__dict__[att] = np.array(self.__dict__[att])


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
