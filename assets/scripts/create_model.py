import torch

from torch import nn


class DefaultModel(nn.Module):
    """ Creates a vanilla Neural Network.

    Creates a Convolutional Neural Network (CNN) with 1 Convolutional block and with given input size, hidden units and classes.

    Args:
        input_size (int): Number of input features for the model. Defaulting to 3 for this example model.
        hidden_units (int): Number of output channels to set in the convolutional layers of the model. Defaults to 10.
        num_classes (int): Number of classes to output from the model. Defaults to 3.
    Example:
        model = DefaultModel(input_size=3, hidden_units=10, num_classes=3) 
    """

    def __init__(self, input_size: int = 3, hidden_units: int = 10, num_classes: int = 3):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Assuming the default resizing transform of 64x64
        # and given the kernel size of 3, stride and padding of 1 used in the convolutions
        # and  kernel size of 2 in the pooling, using the equation:
        # floor(((n + 2p - f) / s) + 1)
        # we get the sizes after each convolution producing a 32x32 in the case of the default input size of 64x64.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*32*32,
                      out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.linear_layer_2(self.linear_layer_1(x)))
