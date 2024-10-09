import json
import os
import subprocess
from pathlib import Path
from typing import Optional
import time
import rich
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer
from typing_extensions import Annotated

from SMS_BP.simulate_cell import Simulate_cells
from SMS_BP import __version__

# create a new CLI function
typer_app_sms_bp = typer.Typer(
    name="SMS_BP CLI Tool",
    short_help="CLI tool for SMS_BP.",
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


# make a callback function to run the simulation
@typer_app_sms_bp.callback()
def cell_simulation():
    """
    CLI tool to run [underline]S[/underline]ingle [underline]M[/underline]olecule [underline]S[/underline]imulation: [underline]SMS[/underline]-BP. GitHub: [green]https://github.com/joemans3/SMS_BP[/green]
    """
    # print version
    # find version using the __version__ variable in the __init__.py file
    rich.print(f"Using SMS_BP version: [bold]{__version__}[/bold]")


@typer_app_sms_bp.command(name="config")
def generate_config(
    output_path: Annotated[
        Path,
        typer.Option("--output_path", "-o", help="Path to the output file"),
    ] = Path("."),
    output_path_make_recursive: Annotated[
        Optional[bool],
        typer.Option(
            "--recursive_o",
            "-r",
            help="Make the output directory if it does not exist",
        ),
    ] = None,
) -> None:
    """
    Generate a sample configuration file for the cell simulation and save it to the specified output path.
    """

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task_1 = progress.add_task(
            description="Processing request to create a default config file ...",
            total=10,
        )

        # check if the output path is provided and is a valid directory | if not none

        try:
            output_path = Path(output_path)
        except ValueError:
            print("FileNotFoundError: Invalid output path.")
            raise typer.Abort()
        # double check if the output path is a valid directory
        if not output_path.is_dir():
            # if not, make the directory
            if output_path_make_recursive:
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                except FileExistsError:
                    print(f"FileExistsError: Directory {output_path} already exists.")
            else:
                print(f"FileNotFoundError: {output_path} is not a valid directory.")
                raise typer.Abort()
        # find the parent dir
        project_directory = Path(__file__).parent
        # find the config file
        config_file = project_directory / "sim_config.json"
        output_path = output_path / "sim_config.json"
        # copy the config file to the output path

        # complete last progress
        progress.update(task_1, completed=10)

        task_2 = progress.add_task(
            description="Copying the config file to the output path ...", total=10
        )
        try:
            subprocess.run(
                ["cp", config_file, output_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError:
            rich.print(f"Error: No config file found in {project_directory}.")
            raise typer.Abort()
        progress.update(task_2, completed=10)
        # complete
        rich.print(f"Config file saved to {output_path.resolve()}")


# second command to run the simulation using the config file path as argument
@typer_app_sms_bp.command(name="runsim")
def run_cell_simulation(
    config_file: Annotated[Path, typer.Argument(help="Path to the configuration file")],
):
    """
    Run the cell simulation using the configuration file provided.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        start_task_1 = time.time()
        task_1 = progress.add_task(
            description="Processing request to run the simulation ...", total=10
        )

        # check if the config file is a valid file
        if not os.path.isfile(config_file):
            rich.print("FileNotFoundError: Configuration file not found.")
            raise typer.Abort()
        # check if the config file is a valid json file
        try:
            with open(config_file) as f:
                config = json.load(f)
        except json.JSONDecodeError:
            rich.print("JSONDecodeError: Configuration file is not a valid JSON file.")
            raise typer.Abort()
        # check if the config file is a valid config file
        if "Output_Parameters" not in config:
            rich.print(
                "ConfigError: 'Output_Parameters' section not found in the configuration file."
            )
            raise typer.Abort()
        else:
            output_parameters = config["Output_Parameters"]
            if "output_path" in output_parameters:
                output_path = output_parameters["output_path"]
            else:
                rich.print(
                    "ConfigError: 'output_path' not found in the configuration file."
                )
                raise typer.Abort()
        # find the version flag in the config file
        if "version" in config:
            version = config["version"]
            rich.print(f"Using config version: [bold]{version}[/bold]")
        # complete last progress
        progress.update(task_1, completed=10)
        rich.print(
            "Prep work done in {:.2f} seconds.".format(time.time() - start_task_1)
        )

        time_task_2 = time.time()
        task_2 = progress.add_task(description="Running the simulation ...", total=None)
        # run the simulation
        sim = Simulate_cells(str(config_file))
        sim.get_and_save_sim(
            cd=output_path,
            img_name=output_parameters.get("output_name"),
            subsegment_type=output_parameters.get("subsegment_type"),
            sub_frame_num=int(output_parameters.get("subsegment_number")),
        )

        progress.update(task_2, completed=None)
        rich.print(
            "Simulation completed in {:.2f} seconds.".format(time.time() - time_task_2)
        )


# def main_CLI():
#     """
#     CLI tool to run cell simulation.

#     Usage:
#         python run_cell_simulation.py <config_file> [--output_path <output_path>]

#     Arguments:
#         config_file     Path to the configuration file

#     Options:
#         --output_path   Path to the output directory

#     """
#     parser = argparse.ArgumentParser(description="CLI tool to run cell simulation")
#     parser.add_argument("config_file", help="Path to the configuration file")
#     parser.add_argument("--output_path", help="Path to the output directory")
#     args = parser.parse_args()

#     config_file = args.config_file
#     output_path = args.output_path

#     if not os.path.isfile(config_file):
#         print("Error: Configuration file not found.")
#         sys.Abort()

#     with open(config_file) as f:
#         config = json.load(f)

#     if "Output_Parameters" not in config:
#         print("Error: 'Output_Parameters' section not found in the configuration file.")
#         sys.Abort()

#     output_parameters = config["Output_Parameters"]

#     if "output_path" in output_parameters:
#         output_path = output_parameters["output_path"]

#     if not output_path:
#         print(
#             "Error: Output path not provided in the configuration file or as a command-line argument."
#         )
#         sys.Abort()

#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     sim = Simulate_cells(config_file)
#     sim.get_and_save_sim(
#         cd=output_path,
#         img_name=output_parameters.get("output_name"),
#         subsegment_type=output_parameters.get("subsegment_type"),
#         sub_frame_num=int(output_parameters.get("subsegment_number")),
#     )


# # utility CLI tool to create a copy of the sim_config.json file in the current directory from which the tool is run
# def create_config():
#     """
#     Create a copy of the sim_config.json file in the current directory

#     Usage:
#         python run_cell_simulation.py create_config --output_path <output_path>
#     Options:
#         --output_path   Path to the output directory
#     """

#     parser = argparse.ArgumentParser(
#         description="CLI tool to create a copy of the sim_config.json file in the current directory"
#     )
#     parser.add_argument("--output_path", help="Path to the output directory")
#     args = parser.parse_args()

#     # check if the output path is provided and is a valid directory
#     if args.output_path and not os.path.isdir(args.output_path):
#         # make the directory if it does not exist but tell the user
#         print("Creating directory structure: ", args.output_path)
#         os.makedirs(args.output_path)

#     # find the project directory
#     project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#     # find the config file
#     config_file = os.path.join(project_directory, "SMS_BP", "sim_config.json")

#     # if the output path is provided append it to the config file
#     if args.output_path:
#         TEMP_CONFIG_FILE = os.path.join(args.output_path, "sim_config.json")
#     else:
#         TEMP_CONFIG_FILE = os.path.join(os.getcwd(), "sim_config.json")

#     # check if the config file exists in the current directory
#     if os.path.exists(TEMP_CONFIG_FILE):
#         # warn the user that the file already exists
#         print(
#             f"Warning: Configuration file already exists in the current directory: {TEMP_CONFIG_FILE}"
#         )
#         # stopping and do nothing
#         return
#     # copy the config file to the current directory
#     os.system(f"cp {config_file} {TEMP_CONFIG_FILE}")


# # make a new function which handles running this script without CLI arguments


# def main_noCLI(file):
#     """
#     Run cell simulation without using CLI arguments
#     """
#     config_file = file
#     if not os.path.isfile(config_file):
#         print("Error: Configuration file not found.")
#         sys.Abort()
#     with open(config_file) as f:
#         config = json.load(f)
#     if "Output_Parameters" not in config:
#         print("Error: 'Output_Parameters' section not found in the configuration file.")
#         sys.Abort()
#     output_parameters = config["Output_Parameters"]
#     if "output_path" in output_parameters:
#         output_path = output_parameters["output_path"]
#     else:
#         print(
#             "Error: Output path not provided in the configuration file or as a command-line argument."
#         )
#         sys.Abort()
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     sim = Simulate_cells(config_file)
#     sim.get_and_save_sim(
#         cd=output_path,
#         img_name=output_parameters.get("output_name"),
#         subsegment_type=output_parameters.get("subsegment_type"),
#         sub_frame_num=int(output_parameters.get("subsegment_number")),
#     )


# # if __name__ == "__main__":
# #     # if the script is run from the command line
# #     if len(sys.argv) > 1:
# #         main_CLI()
# #     else:
# #         # if the script is run as a module use the project directory to find sim_config.json
# #         # find the project directory
# #         project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# #         # find the config file
# #         config_file = os.path.join(project_directory, "SMS_BP", "sim_config.json")
# #         # run the main function
# #         main_noCLI(config_file)
