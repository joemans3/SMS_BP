

# This is a short doc for the Track Simualtions along with cluster simulations.
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
    - Run the command: **pip install -e . --config-settings editable_mode=compat**
    - This will install the package in editable mode and you can now use the package in any python environment without having to append the path every time. 


Okay now we can run the simulation with the predefined variables. For your understanding I rather have you read a short User Guide before I tell you how to run or use this code. Namely because it will help you think of the features included and what is possible. Now I want you to go to USER_GUIDE/USER_GUIDE.pdf and read the document. If you don't care, go to section 4 of that document to get right to the running of this code.




