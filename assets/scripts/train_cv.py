import argparse
import os
import torch
import torchvision

from timeit import default_timer as timer
from datetime import datetime
from torch import nn
from torchvision import transforms

from tools import tools, train_tools, create_model, create_dataloaders_cv


def get_args_parser(add_help=True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set hyperparameters to train a model.", add_help=add_help)
    parser.add_argument('--epochs', type=int, default=5,
                        help="Numbner of epochs to train the model. Defaults to 5.")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate for the optimizer. Defaults to 0.001")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for the dataloader. Defaults to 32.")
    parser.add_argument('--model_name', type=str, default='BaseModel',
                        help="Name to give to the model. Defaults to 'BaseModel'")

    return parser


def main(args: argparse.ArgumentParser):
    # Set model information variables
    model_name = args.model_name
    current_time = datetime.now().strftime("%d%m%y%H%M%S")
    model_save_name = f"{model_name}-{current_time}"

    # Set device agnostic mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seeds for reproducibility
    SEED = 14
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Set hyperparameters
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    # Set number of workers
    num_workers = os.cpu_count()

    # Create basic Data Transform. Change according to project needs. for instance in case of transfer learning use transforms appropriate for the model.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create dataloaders
    train_dataloader, test_dataloader, class_names = create_dataloaders_cv.create_train_test_dataloaders(train_dir="../data/train",
                                                                                                         test_dir="../data/test",
                                                                                                         transform=transform,
                                                                                                         batch_size=batch_size,
                                                                                                         num_workers=num_workers)

    # Create model instance and send it to device
    model = create_model.DefaultModel(
        input_size=3, hidden_units=10, num_classes=len(class_names)).to(device)

    # Create default loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # Start timer and train model
    start_time = timer()

    model_results = train_tools.train(model=model,
                                      train_dataloader=train_dataloader,
                                      test_dataloader=test_dataloader,
                                      loss_function=loss_function,
                                      optimizer=optimizer,
                                      epochs=epochs,
                                      device=device)

    # End timer and print how long the model took to train.
    end_time = timer()
    print(
        f"Model completed training {epochs} epochs in {end_time - start_time:.3f} seconds.")

    tools.save_model(model=model,
                     model_name=model_save_name)
    tools.save_results(results=model_results,
                       model_name=model_save_name)


if __name__ == "__main__":
    args = get_args_parser().parse_args
    main(args)
