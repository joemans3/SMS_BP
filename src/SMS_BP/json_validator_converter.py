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


def validate_json(json_data):
    try:
        validate(instance=json_data, schema=schema)
        return True
    except ValidationError as e:
        raise ConfigValidationError(f"JSON validation error: {e}")


def json_to_dataclass(json_data):
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


def load_validate_and_convert(file_path):
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
    try:
        validate_json(loaded_json)
        return json_to_dataclass(loaded_json)
    except ValidationError as e:
        raise ConfigValidationError(f"JSON validation error: {e}")
    except TypeError as e:
        raise ConfigConversionError(f"Type mismatch during conversion: {e}")
    except Exception as e:
        raise ConfigValidationError(f"An unexpected error occurred: {e}")
