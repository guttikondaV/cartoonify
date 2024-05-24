"""
Utility functions.

The functions in this module are:
- get_parameters_size_in_mb: Get the size of the parameters and buffers of a module in MB.
- load_checkpoint: Load the latest checkpoint from the checkpoints directory.
- save_checkpoint: Save the checkpoint to the checkpoints directory.
"""

import os
import pathlib
import shutil

import torch
import torch.nn as nn
import torch.optim as optim


def get_parameters_size_in_mb(module: nn.Module) -> float:
    """
    Get the size of the parameters and buffers of a module in MB.
    """
    param_size = 0

    for param in module.parameters():
        param_size += param.element_size() * param.nelement()

    buffer_size = 0

    for buffer in module.buffers():
        buffer_size += buffer.element_size() * buffer.nelement()

    size_in_mb = (param_size + buffer_size) / 1024**2

    return size_in_mb


def load_checkpoint() -> dict | None:
    """
    Load the latest checkpoint from the checkpoints directory.

    Returns:
        dict | None: The checkpoint dictionary if a checkpoint is found else None.
    """
    CKPT_DIR = os.environ.get(
        "CKPT_DIR", pathlib.Path(__file__).parent.parent / "checkpoints"
    )

    if isinstance(CKPT_DIR, str):
        CKPT_DIR = pathlib.Path(CKPT_DIR)

    if not os.path.exists(CKPT_DIR):
        return None

    files_in_ckpt_dir = sorted(os.listdir(CKPT_DIR))

    if len(files_in_ckpt_dir) == 0:
        return None

    files_in_ckpt_dir = list(
        filter(lambda file_name: file_name.endswith(".tar"), files_in_ckpt_dir)
    )

    if len(files_in_ckpt_dir) == 0:
        return None

    latest_checkpoint_filename = max(
        files_in_ckpt_dir, key=lambda file_name: file_name.split("=")[-1].split(".")[0]
    )

    checkpoint_path = os.path.join(CKPT_DIR, latest_checkpoint_filename)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    return checkpoint


def save_checkpoint(
    g: nn.Module,
    f: nn.Module,
    d_x: nn.Module,
    d_y: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
):
    """
    Save the checkpoint to the checkpoints directory.

    Args:
        g (nn.Module): The generator model.
        f (nn.Module): The generator model.
        d_x (nn.Module): The discriminator model.
        d_y (nn.Module): The discriminator model.
        optimizer (optim.Optimizer): The optimizer.
        epoch (int): The epoch number.
    """
    CKPT_DIR = os.environ.get(
        "CKPT_DIR", pathlib.Path(__file__).parent.parent / "checkpoints"
    )

    if isinstance(CKPT_DIR, str):
        CKPT_DIR = pathlib.Path(CKPT_DIR)

    INITIAL_DEVICE = os.environ.get(
        "DEVICE",
        (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )

    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    else:
        shutil.rmtree(CKPT_DIR)
        os.makedirs(CKPT_DIR)

    g = g.to("cpu")
    f = f.to("cpu")

    d_x = d_x.to("cpu")
    d_y = d_y.to("cpu")

    torch.save(
        {
            "g_state_dict": g.state_dict(),
            "f_state_dict": f.state_dict(),
            "d_x_state_dict": d_x.state_dict(),
            "d_y_state_dict": d_y.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "name": f"checkpoint_epoch={epoch}.tar",
        },
        os.path.join(CKPT_DIR, f"checkpoint_epoch={epoch}.tar"),
    )

    g = g.to(INITIAL_DEVICE)
    f = f.to(INITIAL_DEVICE)

    d_x = d_x.to(INITIAL_DEVICE)
    d_y = d_y.to(INITIAL_DEVICE)


__all__ = ["get_parameters_size_in_mb", "load_checkpoint", "save_checkpoint"]


def main():
    pass


if __name__ == "__main__":
    main()
