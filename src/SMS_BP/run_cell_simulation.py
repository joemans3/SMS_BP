import argparse
import json
import os
import sys

from SMS_BP.simulate_cell import Simulate_cells


def main_CLI():
    """
    CLI tool to run cell simulation.

    Usage:
        python run_cell_simulation.py <config_file> [--output_path <output_path>]

    Arguments:
        config_file     Path to the configuration file

    Options:
        --output_path   Path to the output directory

    """
    parser = argparse.ArgumentParser(description="CLI tool to run cell simulation")
    parser.add_argument("config_file", help="Path to the configuration file")
    parser.add_argument("--output_path", help="Path to the output directory")
    args = parser.parse_args()

    config_file = args.config_file
    output_path = args.output_path

    if not os.path.isfile(config_file):
        print("Error: Configuration file not found.")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    if "Output_Parameters" not in config:
        print("Error: 'Output_Parameters' section not found in the configuration file.")
        sys.exit(1)

    output_parameters = config["Output_Parameters"]

    if "output_path" in output_parameters:
        output_path = output_parameters["output_path"]

    if not output_path:
        print(
            "Error: Output path not provided in the configuration file or as a command-line argument."
        )
        sys.exit(1)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sim = Simulate_cells(config_file)
    sim.get_and_save_sim(
        cd=output_path,
        img_name=output_parameters.get("output_name"),
        subsegment_type=output_parameters.get("subsegment_type"),
        sub_frame_num=int(output_parameters.get("subsegment_number")),
    )


# utility CLI tool to create a copy of the sim_config.json file in the current directory from which the tool is run
def create_config():
    """
    Create a copy of the sim_config.json file in the current directory

    Usage:
        python run_cell_simulation.py create_config --output_path <output_path>
    Options:
        --output_path   Path to the output directory
    """

    parser = argparse.ArgumentParser(
        description="CLI tool to create a copy of the sim_config.json file in the current directory"
    )
    parser.add_argument("--output_path", help="Path to the output directory")
    args = parser.parse_args()

    # check if the output path is provided and is a valid directory
    if args.output_path and not os.path.isdir(args.output_path):
        # make the directory if it does not exist but tell the user
        print("Creating directory structure: ", args.output_path)
        os.makedirs(args.output_path)

    # find the project directory
    project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # find the config file
    config_file = os.path.join(project_directory, "SMS_BP", "sim_config.json")

    # if the output path is provided append it to the config file
    if args.output_path:
        TEMP_CONFIG_FILE = os.path.join(args.output_path, "sim_config.json")
    else:
        TEMP_CONFIG_FILE = os.path.join(os.getcwd(), "sim_config.json")

    # check if the config file exists in the current directory
    if os.path.exists(TEMP_CONFIG_FILE):
        # warn the user that the file already exists
        print(
            f"Warning: Configuration file already exists in the current directory: {TEMP_CONFIG_FILE}"
        )
        # stopping and do nothing
        return
    # copy the config file to the current directory
    os.system(f"cp {config_file} {TEMP_CONFIG_FILE}")


# make a new function which handles running this script without CLI arguments


def main_noCLI(file):
    """
    Run cell simulation without using CLI arguments
    """
    config_file = file
    if not os.path.isfile(config_file):
        print("Error: Configuration file not found.")
        sys.exit(1)
    with open(config_file) as f:
        config = json.load(f)
    if "Output_Parameters" not in config:
        print("Error: 'Output_Parameters' section not found in the configuration file.")
        sys.exit(1)
    output_parameters = config["Output_Parameters"]
    if "output_path" in output_parameters:
        output_path = output_parameters["output_path"]
    else:
        print(
            "Error: Output path not provided in the configuration file or as a command-line argument."
        )
        sys.exit(1)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    sim = Simulate_cells(config_file)
    sim.get_and_save_sim(
        cd=output_path,
        img_name=output_parameters.get("output_name"),
        subsegment_type=output_parameters.get("subsegment_type"),
        sub_frame_num=int(output_parameters.get("subsegment_number")),
    )


if __name__ == "__main__":
    # if the script is run from the command line
    if len(sys.argv) > 1:
        main_CLI()
    else:
        # if the script is run as a module use the project directory to find sim_config.json
        # find the project directory
        project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # find the config file
        config_file = os.path.join(project_directory, "SMS_BP", "sim_config.json")
        print(config_file)
        # run the main function
        main_noCLI(config_file)
