"""
This module contains the ConvBlock class.

The ConvBlock class is the basic building block of the generator and the
discriminator. It consists of a convolutional layer, a batch normalization
layer and a leaky ReLU activation function.
"""

import os

import torch
import torch.nn as nn
from typing import Self, Tuple

try:
    if os.environ.get("ENV", "development") == "production":
        import src.utils as utils
    else:
        import utils
except ImportError:
    pass


class ConvBlock(nn.Module):
    """
    This block is the basic building block of the generator and the
    discriminator. It consists of a convolutional layer, a batch
    normalization layer and a leaky ReLU activation function.

    This layer is the basis for all learning in the GAN.
    """

    def __init__(
        self: Self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        use_transpose_convolution: bool = False,
        use_leaky_relu: bool = False,
        leaky_relu_slope: float = 0.2,
    ):
        """
        Initializes the ConvBlock class.

        Parameters:

        - in_channels (int): The number of input channels.
        - out_channels (int): The number of output channels.
        - kernel_size (int or tuple): The size of the kernel. Must be an int or (int, int)
        - stride (int): The stride of the convolutional layer.
        - padding (int): The padding of the convolutional layer
        - use_transpose_convolution (bool): Whether to use transpose convolution or not.
        - use_leaky_relu (bool): Whether to use leaky ReLU or not.
        - leaky_relu_slope (float): The slope of the leaky ReLU activation function.
        """
        super(ConvBlock, self).__init__()

        conv_block_to_use = (
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="reflect",
                dtype=torch.float32,
            )
            if not use_transpose_convolution
            else nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="zeros",
                dtype=torch.float32,
            )
        )

        self.layers = nn.Sequential(
            conv_block_to_use,
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope) if use_leaky_relu else nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self: Self):
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
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
        Forward pass of the ConvBlock class.

        Parameters:

        - x (torch.Tensor): The input tensor.

        Returns:

        - torch.Tensor: The output tensor.
        """
        return self.layers(x)


__all__ = ["ConvBlock"]


def main():
    random_tensor = torch.randint(0, 255, (1, 3, 256, 256), dtype=torch.float32)
    block = ConvBlock(3, 64, 3, 1, 1)

    output = block(random_tensor)

    print(output.shape)

    block_size_in_mb = utils.get_parameters_size_in_mb(block)

    print(f"ConvBlock Size: {block_size_in_mb:.4f} MB")


if __name__ == "__main__":
    main()
