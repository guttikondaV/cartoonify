"""
This module contains the Generator network for the GAN.

The Generator network is a deep convolutional neural network that generates
emojis from people images.

The Generator network is the first part of the GAN and is responsible for
generating new images.
"""

import os

import torch
import torch.nn as nn
from typing import Self

try:
    if os.environ.get("ENV", "development") == "production":
        import src.convolution_block as conv_block
        import src.residual_block as residual_block
        import src.utils as utils
    else:
        import convolution_block as conv_block
        import residual_block
        import utils
except ImportError:
    pass


class Generator(nn.Module):
    """
    Generator network for the GAN.
    """

    def __init__(self: Self, in_channels: int = 3, out_channels: int = 3) -> None:
        """
        Initializes the Generator network.

        Args:
            in_channels (int): The number of channels in the input image.
            out_channels (int): The number of channels in the output image.
        """
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            conv_block.ConvBlock(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
            ),  # c7s1-64
            conv_block.ConvBlock(
                in_channels=64, out_channels=128, kernel_size=3
            ),  # d128
            conv_block.ConvBlock(
                in_channels=128, out_channels=256, kernel_size=3
            ),  # d256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            conv_block.ConvBlock(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                use_transpose_convolution=True,
            ),  # u128
            conv_block.ConvBlock(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                use_transpose_convolution=True,
            ),  # u64
            conv_block.ConvBlock(
                in_channels=64,
                out_channels=out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            ),  # c7s1-3
            nn.Sigmoid(),
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Generator network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.layers(x)


__all__ = ["Generator"]


def main():
    gen = Generator(in_channels=3, out_channels=3)

    x = torch.randn(1, 3, 256, 256)

    y = gen(x)

    print(y.shape)

    generator_size_in_mb = utils.get_parameters_size_in_mb(gen)

    print(f"Generator Size: {generator_size_in_mb:.4f} MB")


if __name__ == "__main__":
    main()
