import torch
import torchvision

from typing import List, Tuple
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def create_dataloader(data_dir: str,
                      transform: torchvision.transforms,
                      batch_size: int,
                      num_workers: int = 0) -> Tuple[DataLoader, List[str]]:
    """Creates a torch DataLoader on a torchvision ImageFolder on the given data dir.

    Args:
        data_dir (str): the path to the data.
        transform (torchvision.transforms): the image transforms to apply to the data.
        batch_size (int): the size of the DataLoader batch to be used.
        num_workers: the number of subprocesses for data loading. Defaults to 0, same as Pytorch.

    Returns:
        A tuple containing the DataLoader with the data in the given directory and the list of class names.
    """
    dataset = ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )

    return dataloader, class_names


def create_train_test_dataloaders(train_dir: str,
                                  test_dir: str,
                                  transform: torchvision.transforms,
                                  batch_size: int,
                                  num_workers: int = 0) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Creates torch DataLoaders using torchvision ImageFolder on the given train and test
    directories.

    Args:
        train_dir (str): the path to the train data.
        test_dir (str): the path to the test data.
        transform (torchvision.transforms): the image transforms to apply to the images in the datasets.
        batch_size (int): the size of the DataLoader batch to be used.
        num_workers: the number of subprocesses for data loading. Defaults to 0, same as Pytorch.

    Returns:
        Tuple containing:
            train_dataloader (torch.utils.data.DataLoader): dataloader for training a model with the training data.
            test_dataloader (torch.utils.data.DataLoader): dataloader for testing a model with test data.
            class_names (list): list of the class names for the current project.
    """
    train_dataloader, class_names = create_dataloader(
        train_dir,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers
    )

    test_dataloader, _ = create_dataloader(
        test_dir,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return train_dataloader, test_dataloader, class_names
