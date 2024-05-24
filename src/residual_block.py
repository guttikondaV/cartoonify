"""
This module contains the ResidualBlock class.

The ResidualBlock class is a block used in the ResNet architecture. It consists
of two convolutional layers. The input is added to the output of the second
convolutional layer.

This block is used to create the discriminator in the GAN.
"""

import os

import torch
import torch.nn as nn
from typing import Self

try:
    if os.environ.get("ENV", "development") == "production":
        import src.utils as utils
    else:
        import utils
except ImportError:
    pass


class ResidualBlock(nn.Module):
    """
    This is the block used in the ResNet architecture. It consists of two
    convolutional layers. The input is added to the output of the second
    convolutional layer.

    This block is used to create the discriminator in the GAN.

    Parameters:

    - in_channels (int): The number of input channels.
    - out_channels (int): The number of output channels.
    """

    def __init__(self: Self, in_channels: int, out_channels: int):

        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dtype=torch.float32,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dtype=torch.float32,
            ),
        )

        self._init_weights()

    def _init_weights(self: Self):
        for layer in self.layers:
            # Initialize the weights and bias with a normal distribution
            # This initialization is recommended by the authors of the paper
            nn.init.normal_(
                layer.weight,
                mean=float(os.environ.get("INIT_MEAN", 0.0)),
                std=float(os.environ.get("INIT_STD", 0.02)),
            )
            nn.init.normal_(
                layer.bias,
                mean=float(os.environ.get("INIT_MEAN", 0.0)),
                std=float(os.environ.get("INIT_STD", 0.02)),
            )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualBlock class.

        Parameters:

        - x (torch.Tensor): The input tensor.

        Returns:

        - torch.Tensor: The output tensor.
        """
        return self.layers(x)

__all__ = ['ResidualBlock']

def main():
    random_tensor = torch.randint(0, 255, (1, 3, 256, 256), dtype=torch.float32)
    block = ResidualBlock(in_channels=3, out_channels=3)

    output = block(random_tensor)

    print(output.shape)

    residual_block_size_in_mb = utils.get_parameters_size_in_mb(block)

    print(f"Residual Block Size: {residual_block_size_in_mb:.4f} MB")


if __name__ == "__main__":
    main()
