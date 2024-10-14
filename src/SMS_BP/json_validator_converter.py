import json
from jsonschema import validate, ValidationError
from SMS_BP.config_schema import (
    schema,
    SimulationConfig,
    CellParameters,
    TrackParameters,
    GlobalParameters,
    CondensateParameters,
    OutputParameters,
)
from SMS_BP.errors import ConfigValidationError, ConfigConversionError


def validate_json(json_data: dict) -> bool:
    """
    Validates a JSON object against the defined simulation schema.

    Parameters:
    -----------
    json_data : dict
        The JSON data to be validated.

    Returns:
    --------
    bool
        Returns `True` if the JSON data is valid.

    Raises:
    -------
    ConfigValidationError
        If the JSON data does not comply with the schema.
    """
    try:
        validate(instance=json_data, schema=schema)
        return True
    except ValidationError as e:
        raise ConfigValidationError(f"JSON validation error: {e}")


def json_to_dataclass(json_data: dict) -> SimulationConfig:
    """
    Converts a validated JSON object into a `SimulationConfig` dataclass.

    Parameters:
    -----------
    json_data : dict
        The validated JSON data representing the simulation configuration.

    Returns:
    --------
    SimulationConfig
        A `SimulationConfig` object populated with data from the JSON input.

    Raises:
    -------
    ConfigConversionError
        If required keys are missing or if there is a type mismatch during conversion.
    """
    try:
        return SimulationConfig(
            version=json_data["version"],
            length_unit=json_data["length_unit"],
            space_unit=json_data["space_unit"],
            time_unit=json_data["time_unit"],
            intensity_unit=json_data["intensity_unit"],
            diffusion_unit=json_data["diffusion_unit"],
            Cell_Parameters=CellParameters(**json_data["Cell_Parameters"]),
            Track_Parameters=TrackParameters(**json_data["Track_Parameters"]),
            Global_Parameters=GlobalParameters(**json_data["Global_Parameters"]),
            Condensate_Parameters=CondensateParameters(
                **json_data["Condensate_Parameters"]
            ),
            Output_Parameters=OutputParameters(**json_data["Output_Parameters"]),
        )
    except KeyError as e:
        raise ConfigConversionError(f"Missing key in JSON data: {e}")
    except TypeError as e:
        raise ConfigConversionError(f"Type mismatch during conversion: {e}")


def load_validate_and_convert(file_path: str) -> SimulationConfig:
    """
    Loads, validates, and converts a simulation configuration file to a `SimulationConfig` dataclass.

    This function reads the JSON configuration file, validates its structure against the schema,
    and then converts it to a `SimulationConfig` dataclass.

    Parameters:
    -----------
    file_path : str
        Path to the JSON file containing the simulation configuration.

    Returns:
    --------
    SimulationConfig
        A `SimulationConfig` object populated with the data from the file.

    Raises:
    -------
    ConfigValidationError
        If the JSON file is invalid or cannot be loaded.
    ConfigConversionError
        If an error occurs during the conversion to the dataclass.
    """
    try:
        with open(file_path, "r") as file:
            json_data = json.load(file)

        validate_json(json_data)
        config = json_to_dataclass(json_data)
        return config
    except json.JSONDecodeError as e:
        raise ConfigValidationError(f"Error decoding JSON: {e}")
    except FileNotFoundError as e:
        raise ConfigValidationError(f"File not found: {e}")


def validate_and_convert(loaded_json: dict) -> SimulationConfig:
    """
    Validates and converts an in-memory JSON object to a `SimulationConfig` dataclass.

    This function is useful when the JSON data is already loaded into memory, bypassing the need to
    read from a file.

    Parameters:
    -----------
    loaded_json : dict
        The JSON data to validate and convert.

    Returns:
    --------
    SimulationConfig
        A `SimulationConfig` object populated with data from the JSON input.

    Raises:
    -------
    ConfigValidationError
        If the JSON data does not comply with the schema.
    ConfigConversionError
        If there is a type mismatch or a key is missing during conversion.
    """
    try:
        validate_json(loaded_json)
        return json_to_dataclass(loaded_json)
    except ValidationError as e:
        raise ConfigValidationError(f"JSON validation error: {e}")
    except TypeError as e:
        raise ConfigConversionError(f"Type mismatch during conversion: {e}")
    except Exception as e:
        raise ConfigValidationError(f"An unexpected error occurred: {e}")
