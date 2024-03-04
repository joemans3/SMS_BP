import sys
import os
import argparse
from SMS_BP.simulate_cell import Simulate_cells
import json


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
    parser = argparse.ArgumentParser(
        description="CLI tool to run cell simulation")
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
        print("Error: Output path not provided in the configuration file or as a command-line argument.")
        sys.exit(1)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sim = Simulate_cells(config_file)
    sim.get_and_save_sim(
        cd=output_path,
        img_name=output_parameters.get("output_name"),
        subsegment_type=output_parameters.get("subsegment_type"),
        sub_frame_num=int(output_parameters.get("subsegment_number"))
    )

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
    if not output_path:
        print("Error: Output path not provided in the configuration file or as a command-line argument.")
        sys.exit(1)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    sim = Simulate_cells(config_file)
    sim.get_and_save_sim(
        cd=output_path,
        img_name=output_parameters.get("output_name"),
        subsegment_type=output_parameters.get("subsegment_type"),
        sub_frame_num=int(output_parameters.get("subsegment_number"))
    )


if __name__ == "__main__":
    # if the script is run from the command line
    if len(sys.argv) > 1:
        main_CLI()
    else:
        # if the script is run as a module use the project directory to find sim_config.json
        # find the project directory
        project_directory = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        # find the config file
        config_file = os.path.join(
            project_directory, "SMS_BP", "sim_config.json")
        print(config_file)
        # run the main function
        main_noCLI(config_file)
