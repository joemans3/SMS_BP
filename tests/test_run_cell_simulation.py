import os

from typer.testing import CliRunner

from SMS_BP.run_cell_simulation import typer_app_sms_bp

# Create a CLI test runner
runner = CliRunner()


# Test the CLI command for version output
def test_version_output():
    result = runner.invoke(typer_app_sms_bp, [])
    assert result.exit_code == 0
    assert "Using SMS_BP version" in result.output


# Test the config generation command
def test_generate_config(tmpdir):
    # Invoke the command to generate the config
    output_path = tmpdir.mkdir("test_output")
    result = runner.invoke(
        typer_app_sms_bp,
        ["config", "--output_path", str(output_path), "--recursive_o", "True"],
    )

    # Check that the command executed successfully
    assert result.exit_code == 0
    assert "Processing request to create a default config file" in result.output
    assert os.path.exists(os.path.join(output_path, "sim_config.json"))

