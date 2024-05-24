"""
This module contains the loss functions for the package.

The loss functions in this module are:
- AdversarialLoss: The adversarial loss for the GAN.
- CyclicLoss: The cyclic loss for the GAN.
"""

from typing import Self
import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for the GAN.

    Used for both forward and backward models separately.
    """

    def __init__(self: Self) -> None:
        super().__init__()

    def forward(self: Self, disc_pred: torch.Tensor, disc_actual: torch.Tensor):
        """
        Calculates the adversarial loss for the GAN.

        Args:
            disc_pred (torch.Tensor): The discriminator prediction.
            disc_actual (torch.Tensor): The actual discriminator prediction.

        Returns:
            torch.Tensor: The adversarial loss.
        """
        first_term = torch.square(disc_pred - 1)
        second_term = torch.square(disc_actual - 1)
        third_term = torch.square(disc_pred)

        return first_term + second_term + third_term


class CyclicLoss(nn.Module):
    """
    Cyclic loss for the GAN.

    Used for both forward and backward models together.
    """

    def __init__(self: Self, lambda_multiplier: float = 10) -> None:
        super().__init__()
        self.lambda_multiplier = lambda_multiplier
        self.l1_loss = nn.L1Loss()

    def forward(
        self: Self,
        x: torch.Tensor,
        y: torch.Tensor,
        f_g_x: torch.Tensor,
        g_f_y: torch.Tensor,
    ):
        """
        Calculates the cyclic loss for the GAN.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The target tensor.
            f_g_x (torch.Tensor): The forward generator prediction.
            g_f_y (torch.Tensor): The backward generator prediction.

        Returns:
            torch.Tensor: The cyclic loss.
        """
        first_term = self.l1_loss(f_g_x, x)
        second_term = self.l1_loss(g_f_y, y)

        return self.lambda_multiplier * (first_term + second_term)


__all__ = ["AdversarialLoss", "CyclicLoss"]


def main():
    pass


if __name__ == "__main__":
    main()
