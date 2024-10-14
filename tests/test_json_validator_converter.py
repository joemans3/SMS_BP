import pytest
from SMS_BP.json_validator_converter import (
    load_validate_and_convert,
)
from SMS_BP.errors import ConfigValidationError


@pytest.fixture
def valid_file_path():
    """Fixture for a valid JSON configuration file path."""
    return (
        "tests/sim_config.json"  # Replace with the actual path to your valid JSON file
    )


@pytest.fixture
def invalid_file_path():
    """Fixture for an invalid JSON configuration file path."""
    return "invalid_sim_config.json"  # Replace with the actual path to an invalid JSON file


def test_load_valid_config(valid_file_path):
    """Test loading and validating a valid JSON config file."""
    config = load_validate_and_convert(valid_file_path)

    # Check that the config is loaded correctly
    print(config, config.version)
    assert config.version == "0.1", "Version mismatch in config"


def test_invalid_config_raises_validation_error(invalid_file_path):
    """Test that loading an invalid JSON file raises ConfigValidationError."""
    with pytest.raises(ConfigValidationError):
        load_validate_and_convert(invalid_file_path)
