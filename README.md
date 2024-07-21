

# This is a short doc for the Track Simulations along with cluster simulations.
-----------------------------------------

- Author: Baljyot Singh Parmar
- Affiliation at the time of writing: McGill University, Canada. Weber Lab



## 1. Installation
-------------------
1. Make sure you have anaconda installed: <https://www.anaconda.com/download>
2. Download or clone this repository.
3. In the conda prompt, navigate to the folder where you downloaded this repository using : **cd "path_to_folder"**
4. Using the **SMS_BP.yml** file, create a new environment using: **conda env create -f SMS_BP.yml**
    - If you get an environment resolve error but you have anaconda installed just skip to step 6. The .yml file is for people who are using miniconda and might not have the packages already installed with the full anaconda install.
    - You may want to still have a conda environment so just create a generic one if you want with the name SMS_BP or whatever you want with python>=3.10. Explicitly, **conda create -n [my_env_name] python=3.10**.
5. Activate the environment using: **conda activate SMS_BP**
6. Now we will install this package in edit mode so we can use its functionalities without invoking sys.path.append() every time.
    - Run the command: **pip install -e . --config-settings editable_mode=compat**
    - This will install the package in editable mode and you can now use the package in any python environment without having to append the path every time. 

To start interacting with the program **cd SMS_BP** and follow the next steps.
Okay now we can run the simulation with the predefined variables. For your understanding I rather have you read a short User Guide before I tell you how to run or use this code. Namely because it will help you think of the features included and what is possible. Now I want you to go to USER_GUIDE/USER_GUIDE.pdf and read the document. If you don't care, go to section 4 of that document to get right to the running of this code.

7. This is a note on using the CLI tool properly. For the previous step I have forced you to go to the file location and run the script through python. But in the install (step 6) we also installed a CLI tool to interface with the program from anywhere (regardless of where you are in your terminal). The only condition is that you are in the SMS_BP conda environment. 
    - To run the CLI tool after the install we can type **run_SMS_BP [PATH_TO_CONFIG_FILE]** from anywhere assuming the path you provide is absolute.
    - In the background all this is doing is running: **from SMS_BP.run_cell_simulation import main_CLI(); main_CLI()**. This is the entry point.
    - Do note that the config checker is not robust so if you have prodived the wrong types or excluded some parameters which are required alongside other ones you will get an error. Read the SMS_BP/sim_config.md for details into the config file parameters.
8. If you run into any issues please create a Github issue on the repository as it will help me manage different issues with different people and also create a resource for people encountering a solved issue.

## 2. Viewing Detailed Source Code Documentation
------------------------------------------------
1. Source code documentation is provided in the code. If you don't want to read over it a detailed (auto-generated) version html/latex version is provided through [Doxygen](https://www.doxygen.nl/index.html).
2. The html version is located in **[path]/SMS_BP/docs/Doxygen/html**. To view the doc in your default browser use the (macOS) command `open docs/Doxygen/html/./include.html` assuming you are in the base SMS_BP directory. If not, append the relative path to the above command.
