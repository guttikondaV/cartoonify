"""
This module contains the code to time the different parts of the GAN.

Timing the different parts of the GAN helps in understanding the bottlenecks
in the code and optimizing them.
"""

import os
import time

import torch

try:
    if os.environ.get("ENV", "development") == "production":
        import src.discriminator as discriminator
        import src.generator as generator
        import src.loss as loss
    else:
        import discriminator
        import generator
        import loss
except ImportError:
    pass


def main():
    DEVICE = os.environ.get(
        "DEVICE",
        (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )

    if DEVICE == "mps":
        torch.mps.empty_cache()

    g = generator.Generator(3, 3)
    d_y = discriminator.Discriminator(3)

    f = generator.Generator(3, 3)
    d_x = discriminator.Discriminator(3)

    cyclic_loss_fn = loss.CyclicLoss()
    adversarial_loss_fn = loss.AdversarialLoss()

    g = g.to(DEVICE)
    d_y = d_y.to(DEVICE)

    f = f.to(DEVICE)
    d_x = d_x.to(DEVICE)

    cyclic_loss_fn = cyclic_loss_fn.to(DEVICE)
    adversarial_loss_fn = adversarial_loss_fn.to(DEVICE)

    x = torch.rand((1, 3, 178, 218), dtype=torch.float32).to(DEVICE)
    y = torch.rand((1, 3, 475, 475), dtype=torch.float32).to(DEVICE)

    print("=" * 40)

    # Time taken for first pair
    start = time.perf_counter_ns()
    g_x = g(x)
    end = time.perf_counter_ns()
    print(f"Time taken for G(x): {((end - start) / 1e9):.4f} s")

    print("=" * 40)

    start = time.perf_counter_ns()
    d_y_g_x = d_y(g_x)
    end = time.perf_counter_ns()
    print(f"Time taken for D_y(G(x)): {((end - start) / 1e9):.4f} s")

    print("=" * 40)

    start = time.perf_counter_ns()
    d_y_y = d_y(y)
    end = time.perf_counter_ns()
    print(f"Time taken for D_y(y): {((end - start) / 1e9):.4f} s")

    print("=" * 40)

    start = time.perf_counter_ns()
    adversarial_loss_fn(d_y_g_x, d_y_y)
    end = time.perf_counter_ns()
    print(f"Time taken for First Adversarial Loss: {((end - start) / 1e9):.4f} s")

    print("=" * 40)

    # Move to CPU
    g.to("cpu")
    d_y.to("cpu")

    f.to("cpu")
    d_x.to("cpu")

    cyclic_loss_fn.to("cpu")
    adversarial_loss_fn.to("cpu")


if __name__ == "__main__":
    main()
