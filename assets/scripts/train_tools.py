import torch

from typing import Dict, List, Tuple
from torch import nn
from tqdm.auto import tqdm


def train_step(model: nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               loss_function: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model (nn.Module): A PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
        loss_function (nn.Module): A PyTorch loss function to minimize.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
    # Turn model into training mode
    model.train()

    # Initialize loss and accuracy accumulators
    train_loss, train_accuracy = 0, 0

    for X, y in tqdm(train_dataloader, desc="Training batch...", total=len(train_dataloader)):
        # Send data to device
        X, y = X.to(device), y.to(device)

        # Compute logits
        y_logits = model(X)

        # Compute and accumumlate the loss
        loss = loss_function(y_logits, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Make predictions and compute and accumulate accuracy
        y_probs = torch.softmax(y_logits, dim=1)
        y_preds = torch.argmax(y_probs, dim=1)

        accuracy = (y_preds == y).sum().item() / len(y_preds)
        train_accuracy += accuracy

    # Get bacth average for the metrics
    train_loss = train_loss / len(train_dataloader)
    train_accuracy = train_accuracy / len(train_dataloader)

    return train_loss, train_accuracy


def test_step(model: nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              loss_function: nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model (nn.Module): A PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
        loss_function (nn.Module): A PyTorch loss function to minimize.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of test loss and accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.1112, 0.8743)
    """
    # Initialize loss and accuracy accumulators
    test_loss, test_accuracy = 0, 0

    # Turn model into evaluation mode
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_dataloader, desc="Testing batch...", total=len(test_dataloader)):
            # Send data to device
            X, y = X.to(device), y.to(device)

            # Compute logits
            y_logits = model(X)

            # Compute and accumumlate the loss
            loss = loss_function(y_logits, y)
            test_loss += loss.item()

            # Make predictions and compute and accumulate accuracy
            y_probs = torch.softmax(y_logits, dim=1)
            y_preds = torch.argmax(y_probs, dim=1)

            accuracy = (y_preds == y).sum().item() / len(y_preds)
            test_accuracy += accuracy

    # Get bacth average for the metrics
    test_loss = test_loss / len(test_dataloader)
    test_accuracy = test_accuracy / len(test_dataloader)

    return test_loss, test_accuracy


def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_function: nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        loss_function: A PyTorch loss function to calculate loss on both datasets.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]} 
        For example if training for epochs=2: 
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]}
    """
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }

    for epoch in tqdm(range(epochs), desc="Training model..."):
        train_loss, train_accuracy = train_step(model=model,
                                                train_dataloader=train_dataloader,
                                                loss_function=loss_function,
                                                optimizer=optimizer,
                                                device=device)
        test_loss, test_accuracy = test_step(model=model,
                                             test_dataloader=test_dataloader,
                                             loss_function=loss_function,
                                             device=device)

        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_accuracy)

        # if we have more than 10 epochs only check every 5 (better to increase this interval as epochs increase in magnitude)
        if (epochs < 10) or (epochs % 5 == 0):
            print(f"Epoch: {epoch}")
            print(f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}")
            print(
                f"Train acc: {train_accuracy:.3f} | Test acc: {test_accuracy:.3f}")
            print("-"*50)

    return results
