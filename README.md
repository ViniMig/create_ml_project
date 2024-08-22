# Create ML Project

### Lazy dude command line project to create boilerplate folder structure and code for a Machine Learning Project.

Will be adding more documentation in the next iterations.

---

## Overview

The goal of this project is to very quickly setup most of what is needed to jump start a new ML project. Currently focusing only on Computer Vision tasks working with PyTorch.
First the folder structure will be composed as follows:

```{: .no-copy}
project-name-folder
├───data
│   ├───test
│   ├───train
│   └───validation
├───logs
├───models
└───tools
```

This structure was chosen as an approximate of what is most commonly used in computer vision problems.

A **data** folder for our image data. This folder is then divided in 3 others:

- **train**
- **test**
- **validation**

<br>

The three of them will have a similar structure in content, differing only in their use. They will all have as many subfolders as there are classes, in the case of classification problems (as is the currently use case implemented at the time).
<br>

The **logs** folder will be used to store information from training runs. All the results will be stored here in json format by default.
<br>

The **models** folder similarly to the logs will store model parameters, mening we are saving hre the `state_dict()` using `torch.save()`. The files will be in .pth format, as is preferred by PyTorch.
<br>

The **tools** folder contains most of the "engine" of the boilerplate code. In here upon creation of a new project, a few python files will be made available containing helper functions, explained more in detail bellow.
<br>

Having a look at the full tree of the default project structure we have the following:

```{: .no-copy}
│   .gitignore
│   requirements.txt
│   train_cv.py
│
├───data
│   ├───test
│   ├───train
│   └───validation
├───logs
├───models
└───tools
        create_dataloaders_cv.py
        create_model.py
        tools.py
        train_tools.py
```

The **.gitignore** is the default one generated for python projects. With time and project complexity attention should be paid in keeping this file updated, for example, we should **be very cautious when having files containg API tokens, secrets, keys, and any other relevant information we do NOT want to have available in public.**

The **requirements.txt** comes with a generic list of the modules being used in the boilerplate PyTorch computer vision code. Therefore any additional modules should be added to this file, versions specified in case necessary.

The **train_cv.py** contains a script that can be executed from the cmd linem, with the option to set some hyperparameters as needed. This file is boilerplate interface to run experiments on training certain models. It can be discarded completely, or modified as needed with the appropriate changes as the projects might require. This is just default code to try and help accelerate in the process of training models, calling to the auxiliary scripts available in the **tools** folder.

The following python files, all exist inside the tools folder. The goal is to have some minimal compartmentalization of the project, and the advantage of this all being Python code is that at any time we can either compartmentalize further or just have all files in the same directory, depending on project and on personal decisions.
Inside this folder we have currently 4 files available:

- **create_dataloaders_cv.py**:
- **create_model.py**:
- **tools.py**:
- **train_tools.py**:

---

## Usage

In order to create a new project run the following in the command line:

```cmd
python create_ml_project.py [-h] --project-name PROJECT_NAME [--project-path PROJECT_PATH]
```

Looking at the output from the help:

```{: .no-copy}
Python tool to create a new ML project.

options:
-h, --help show this help message and exit
--project-name PROJECT_NAME
Name of new project
--project-path PROJECT_PATH
Destination path where the project will live. Example: "C:\\users". Defaults to current working directory.
```
