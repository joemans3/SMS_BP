

# This is a short doc for the SMT project.
-----------------------------------------

- Author: Baljyot Singh Parmar
- Affiliation at the time of writing: McGill University, Canada. Weber Lab



## 1. Installation
-------------------
1. Make sure you have anaconda installed: <https://www.anaconda.com/download>
2. Download or clone this repository.
3. In the conda prompt, navigate to the folder where you downloaded this repository using : **cd "path_to_folder"**
4. Using the **SMS_BP.yml** file, create a new environment using: **conda env create -f SMS_BP.yml**
5. Activate the environment using: **conda activate SMS_BP**
6. Install the extra pip packages using: **pip install -r requirements.txt**
7. Since tensflow and tensorflow-probability are platform dependent we need to install inidividually.
    - Try the conda install method: **conda install tensorflow**, and **conda install tensorflow-probability**
    - If the above method fails, try the pip install method: **pip install tensorflow**, and **pip install tensorflow-probability**

8. Now we will install this package in edit mode so we can use its functionalities without invoking sys.path.append() every time.
    - Navigate to the root directory of the package using the conda prompt.
    - Run the command: **pip install -e .**
    - This will install the package in editable mode and you can now use the package in any python environment without having to append the path every time. 
        




