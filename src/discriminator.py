"""
Discriminator network for the GAN.

The discriminator network is a convolutional neural network that takes in an
image and outputs a probability that the image is real.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Self

try:
    if os.environ.get("ENV", "development") == "production":
        import src.convolution_block as conv_block
        import src.utils as utils
    else:
        import convolution_block as conv_block
        import utils
except ImportError:
    pass


class Discriminator(nn.Module):
    """
    Discriminator network for the GAN.

    The discriminator network is a convolutional neural network that takes in an
    image and outputs a probability that the image is real.
    """

    def __init__(self: Self, in_channels: int = 3) -> None:
        """
        Initializes the Discriminator network.

        Args:
            in_channels (int): The number of channels in the input image.
        """
        super(Discriminator, self).__init__()

        self.sigmoid = F.sigmoid
        self.softmax = F.softmax

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=64, kernel_size=4, stride=2
            ),
            conv_block.ConvBlock(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                use_leaky_relu=True,
                leaky_relu_slope=0.2,
            ),
            conv_block.ConvBlock(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                use_leaky_relu=True,
                leaky_relu_slope=0.2,
            ),
            conv_block.ConvBlock(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                use_leaky_relu=True,
                leaky_relu_slope=0.2,
            ),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1),
        )

        self._init_weights()

    def _init_weights(self: Self) -> None:
        """
        Initializes the weights of the discriminator network.
        """
        first_layer = self.layers[0]
        last_layer = self.layers[-1]

        nn.init.normal_(
            first_layer.weight,
            mean=float(os.environ.get("INIT_MEAN", 0.0)),
            std=float(os.environ.get("INIT_STD", 0.02)),
        )
        nn.init.normal_(
            first_layer.bias,
            mean=float(os.environ.get("INIT_MEAN", 0.0)),
            std=float(os.environ.get("INIT_STD", 0.02)),
        )

        nn.init.normal_(
            last_layer.weight,
            mean=float(os.environ.get("INIT_MEAN", 0.0)),
            std=float(os.environ.get("INIT_STD", 0.02)),
        )
        nn.init.normal_(
            last_layer.bias,
            mean=float(os.environ.get("INIT_MEAN", 0.0)),
            std=float(os.environ.get("INIT_STD", 0.02)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator network.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The output tensor of the discriminator network.
        """
        x = self.layers(x)
        x = torch.flatten(x)
        x = torch.sum(x)
        x = self.sigmoid(x)
        return self.softmax(x, dim=0)


__all__ = ["Discriminator"]


def main():
    discriminator = Discriminator()

    random_tensor = torch.randint(0, 255, (1, 3, 512, 512), dtype=torch.float32)

    output = discriminator(random_tensor)

    print(output.shape)
    print(output.item())

    discriminator_size_in_mb = utils.get_parameters_size_in_mb(discriminator)

    print(f"Discriminator size in MB: {discriminator_size_in_mb:.4f} MB")


if __name__ == "__main__":
    main()
