import os
import argparse
import pathlib
import shutil

from pathlib import Path


def get_args_parser(add_help=True):
    """
    Get the arguments from the command line.
    Returns the argparser.
    """
    parser = argparse.ArgumentParser(
        description="Python tool to create a new ML project.", add_help=add_help)
    parser.add_argument('--project-name', type=str,
                        required=True, help='Name of new project')
    parser.add_argument('--project-path', type=str, default=os.getcwd(),
                        help='Destination path where the project will live. Example: "C:\\\\users". Defaults to current working directory.')
    # The current tool will work for ComputerVision only for now.
    # parser.add_argument('--project-type', type=str, default='cv', help='Type of project. cv: Computer Vision; nlp: Natural Language Processing; audio: Audio Processing')

    return parser


def create_home_directory(home_dir: pathlib.Path):
    """
    Given a path of the form 'destination_directory/project name'
    Create directory if it doesn't exist. If direcotry exists stop execution with Error.
    """
    if not home_dir.is_dir():
        print(f"[INFO]: Creating {home_dir} directory.")
        home_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise FileExistsError(
            f"[ERROR]: The directory {home_dir} already exists in the current path. Choose a different destination path or a different project name")


def create_target_directory(target_dir: pathlib.Path):
    """
    Given a target directory path
    Creates a new directory if it doesn't exist.
    Skips this step if directory exists.
    """
    if not target_dir.is_dir():
        print(f"[INFO]: Creating {target_dir} directory.")
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(
            f"[INFO]: directory with path: {target_dir} already exists! Skipping this step. Please confirm any existing data in this directory that might be residual.")


def send_default_files(home_dir: pathlib.Path, tools_dir: pathlib.Path) -> None:
    """Copies the boilerplate files (python scripts, gitignore and requirements) to their target locations.

    Args:
        home_dir (Path): The path to the home directory where the project was created.
        tools_dir (Path): The path to the tools directory.
    """
    assets_dir = Path('assets')
    scripts_dir = assets_dir / 'scripts'
    files_dir = assets_dir / 'files'

    # Scan our scripts directory to get the files to copy
    with os.scandir(scripts_dir) as it:
        for entry in it:
            if not entry.name == 'train_cv.py':
                shutil.copy(scripts_dir / entry.name, tools_dir)
            else:
                shutil.copy(scripts_dir / entry.name, home_dir)

    # Scan our files directory to get the files to copy
    with os.scandir(files_dir) as it:
        for entry in it:
            shutil.copy(files_dir / entry.name, home_dir)


def create_subfolder_structure(home_dir: pathlib.Path):
    """
    Creates the subfolders for the given project type.

    Example:
        destination_folder/home_dir
        |
         --data
           |
            --train
           |
            --test
           |
            --validation
        |
         --models
        |
         --tools
        |
         --logs
    """
    data_dir = home_dir / 'data'
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    validation_dir = data_dir / 'validation'
    models_dir = home_dir / 'models'
    tools_dir = home_dir / 'tools'
    logs_dir = home_dir / 'logs'
    target_dirs = [data_dir, train_dir, test_dir,
                   validation_dir, models_dir, tools_dir, logs_dir]

    for target_dir in target_dirs:
        create_target_directory(target_dir)

    send_default_files(home_dir, tools_dir)


def main(args: argparse.ArgumentParser):
    destination_dir = Path(args.project_path)
    home_dir = destination_dir / args.project_name
    create_home_directory(home_dir)
    create_subfolder_structure(home_dir)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
