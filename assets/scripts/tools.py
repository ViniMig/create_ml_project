import json
import torch
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import Dict, List, Tuple


def save_model(model: torch.nn.Module,
               model_name: str,
               extra_information: str = None,
               target_dir: str = 'models') -> None:
    """Saves a model to the target directory.

    Args:
        model (torch.nn.Module): A PyTorch trained model to save. Only the state_dict() is being saved.
        model_name (str): A name for the trained model to serve as base for the file name.
        extra_information (str, Optional): Extra details to add to the file name. Defaults to None.
        target_dir (str, Optional): Target directory where to save the model. Defaults to 'models'.
    """
    target_path = Path(target_dir)

    if extra_information:
        model_name += extra_information

    model_name += ".pth"
    model_save_path = target_path / model_name

    print(f"[INFO]: Saving model: {model_name} to {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def save_results(results: Dict[str, List[float]],
                 model_name: str,
                 extra_information: str = None,
                 target_dir: str = 'logs') -> None:
    """Saves the results from training a model into a json in the target directory.

    Args:
        results (dict): a dictionary containing the various results from training a model, such as train and test loss and accuracy.
        model_name (str): a name for the model used, which will be used as the base name for the file.
        extra_information (str, Optional): any extra details to be used in naming the file. Defaults to None.
        target_dir (str, Optional): the target directory to save the results. Defaults to 'logs'.
    """
    target_path = Path(target_dir)

    if extra_information:
        model_name += f"_{extra_information}"

    model_name += ".json"

    file_save_path = target_path / model_name
    with open(file_save_path, 'w') as save_file:
        json.dump(results, save_file)


def load_results(results_path: str, results_dir: str = 'logs') -> Tuple[str, str, Dict[str, List[float]]]:
    """Loads a specific set of results stored in the given json file.

    Args:
        results_path (str): The results json file to load.
        results_dir (str, Optional): The directory where the results are stored. Defaults to 'logs'.

    Returns:
        A tuple containing the model name, date and the results dictionary.
    """
    target_path = Path(results_dir)
    file_saved_path = target_path / results_path
    with open(file_saved_path) as results_file:
        results = json.load(results_file)

    # Get model name and date from file name (if it follows default behaviour)
    model_name, model_date = results_path.split('-')
    # Remove file extension '.json' from the date and reformat in a more readable shape.
    # This more readable shape consists of adding common separators for day and time every 2 characters.
    model_date = model_date[:-5]
    model_date = f"{model_date[:2]}/{model_date[2:4]}/{model_date[4:6]} {model_date[6:8]}:{model_date[8:10]}:{model_date[10:]}"

    return model_name, model_date, results


def plot_learning_curves(results: Dict[str, List[float]], model_name: str, model_date: str) -> None:
    """Plots the learning curves for the given model results.

    Args:
        results (dict): Results dictionary of the model metrics to plot.
        model_name (str): Name of the model trained.
        model_date (str): Date of the training run, usually saved in the model file name.
    """
    plot_title = f"{model_name}_{model_date}"
    epochs = np.arange(len(results['train_loss']))

    plt.figure(figsize=(10, 10))
    plt.suptitle(plot_title)

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['train_loss'], label='train loss')
    plt.plot(epochs, results['test_loss'], label='test loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['train_accuracy'], label='train accuracy')
    plt.plot(epochs, results['test_accuracy'], label='test accuracy')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show()
