"""
This script trains the CycleGAN model on Adversarial losses only.

The script trains the CycleGAN model on the People and Emoji dataset. The model
consists of two generators and two discriminators. The generators are used to
transform the images from one domain to another while the discriminators are
used to distinguish between the real and fake images.

The script uses the following modules from the package:
- dataset: The dataset module for the package.
- discriminator: The discriminator module for the package.
- generator: The generator module for the package.
- loss: The loss module for the package.
- utils: The utils module for the package.

The script uses the following third-party libraries:
- albumentations: For image augmentations.
- neptune: For logging the training metrics.
- rich: For pretty printing the tables.

"""

import os
import pathlib

import rich
from rich.console import Console  # type: ignore
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision as tv
import torchvision.io as tv_io
import torchvision.transforms.v2 as tv_transforms
import tqdm

# Project Imports
try:
    if os.environ.get("ENV", "development") == "production":
        import src.dataset as dataset
        import src.discriminator as discriminator
        import src.generator as generator
        import src.loss as loss
        import src.utils as utils
    else:
        import dataset
        import discriminator
        import generator
        import loss
        import utils
except ImportError:
    pass

# SCRIPT CONSTANTS
TRAIN_DIR = os.environ.get(
    "TRAIN_DIR", pathlib.Path(__file__).parent.parent / "data/processed"
)

if isinstance(TRAIN_DIR, str):
    TRAIN_DIR = pathlib.Path(TRAIN_DIR)

EMOJI_TRAIN_DIR = TRAIN_DIR / "cartoonset/"
PEOPLE_TRAIN_DIR = TRAIN_DIR / "people_faces/"


def _print_seperator_line():
    print("=" * 80)


def _print_table(vals: list[list[str, float]]) -> None:
    """
    Prints a table with the given data.

    Args:
        table (list[list[str, float]]): The table data.
    """
    table = rich.table.Table(title="Model Sizes", show_header=True)

    table.add_column("Model", justify="left", style="cyan", no_wrap=True)
    table.add_column("Size (MB)", justify="right", style="magenta")

    for row in vals:
        model_name, model_size = row

        table.add_row(model_name, f"{model_size:.4f}")

    console = Console()
    console.print(table)

    # print(table)


def reduce_lr(current_epoch: int) -> float:
    """
    Reduces the learning rate linearly to 0 until epoch == TOTAL_EPOCHS.

    Args:
        epoch (int): The current epoch number.

    Returns:
        float: The learning rate multiplier.
    """
    TOTAL_EPOCHS = 200
    EPOCH_TO_START_DECAYING_FROM = 50

    if current_epoch < EPOCH_TO_START_DECAYING_FROM:
        return 1.0

    # Start decaying the learning rate linearly to 0 until epoch == TOTAL_EPOCHS
    return 1.0 - (current_epoch - EPOCH_TO_START_DECAYING_FROM) / (
        TOTAL_EPOCHS - EPOCH_TO_START_DECAYING_FROM
    )


def train():
    """
    Trains the CycleGAN model on Adversarial losses only.
    """

    #################### SETUP STARTS HERE ####################
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

    START_EPOCH = 1
    N_EPOCHS = 200
    MODEL_SAVE_FREQ = 1
    #################### SETUP ENDS HERE ####################

    #################### MODELS AND OPTI STARTS HERE ####################

    g = generator.Generator()
    f = generator.Generator()

    d_y = discriminator.Discriminator()
    d_x = discriminator.Discriminator()

    adversarial_loss_fn = loss.AdversarialLoss()

    optimizer = optim.Adam(
        (
            list(g.parameters())
            + list(f.parameters())
            + list(d_y.parameters())
            + list(d_x.parameters())
        ),
        lr=2e-4,
    )

    scheduler = lr_scheduler.LambdaLR(optimizer, reduce_lr)

    #################### MODELS AND OPTI ENDS HERE ####################

    #################### CHECKPOINTING STARTS HERE ####################
    checkpoint = utils.load_checkpoint()

    if checkpoint is not None:
        g.load_state_dict(checkpoint["g_state_dict"])
        f.load_state_dict(checkpoint["f_state_dict"])
        d_y.load_state_dict(checkpoint["d_y_state_dict"])
        d_x.load_state_dict(checkpoint["d_x_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        START_EPOCH = checkpoint["epoch"]

    g.to(DEVICE)
    f.to(DEVICE)

    d_y.to(DEVICE)
    d_x.to(DEVICE)
    #################### CHECKPOINTING ENDS HERE ####################

    #################### PRINTING STARTS HERE ####################
    _print_seperator_line()
    print(f"Using device: {DEVICE}")

    _print_seperator_line()

    g_size_in_mb = utils.get_parameters_size_in_mb(g)
    f_size_in_mb = utils.get_parameters_size_in_mb(f)

    d_y_size_in_mb = utils.get_parameters_size_in_mb(d_y)
    d_x_size_in_mb = utils.get_parameters_size_in_mb(d_x)

    adversarial_loss_fn_size_in_mb = utils.get_parameters_size_in_mb(
        adversarial_loss_fn
    )

    total_size_in_mb = (
        g_size_in_mb
        + f_size_in_mb
        + d_y_size_in_mb
        + d_x_size_in_mb
        + adversarial_loss_fn_size_in_mb
    )

    _print_table(
        [
            ["Generator G", g_size_in_mb],
            ["Generator F", f_size_in_mb],
            ["Discriminator D_Y", d_y_size_in_mb],
            ["Discriminator D_X", d_x_size_in_mb],
            ["Adversarial Loss", adversarial_loss_fn_size_in_mb],
            ["Total", total_size_in_mb],
        ]
    )

    _print_seperator_line()

    if checkpoint is not None:
        print(f"Found checkpoint: {checkpoint['name']}")
        _print_seperator_line()
    #################### PRINTING ENDS HERE ####################

    #################### DATA STARTS HERE ####################
    train_ds = dataset.PeopleEmojiDataset(
        PEOPLE_TRAIN_DIR,
        EMOJI_TRAIN_DIR,
        people_transform=tv_transforms.Compose(
            [
                tv_transforms.RandomHorizontalFlip(p=0.5),
                tv_transforms.ColorJitter(),
                tv_transforms.ToDtype(dtype=torch.float32),
            ]
        ),
        emoji_transform=tv_transforms.Compose(
            [
                tv_transforms.CenterCrop(475),
                tv_transforms.ToDtype(dtype=torch.float32),
            ]
        ),
    )

    train_dl = data.DataLoader(
        train_ds, batch_size=int(os.environ.get("BATCH_SIZE", 1)), shuffle=True
    )

    #################### DATA ENDS HERE ####################

    #################### TRAINING STARTS HERE ####################
    print(f"Starting training from epoch {START_EPOCH} to {N_EPOCHS}")
    _print_seperator_line()

    for epoch in range(START_EPOCH, N_EPOCHS + 1):
        g.train()
        f.train()

        d_y.train()
        d_x.train()

        for x, y in tqdm.tqdm(train_dl):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            # Adversarial Loss for first pair
            g_x = g(x)
            disc_predicted = d_y(g_x)
            disc_actual = d_y(y)

            adversarial_loss = adversarial_loss_fn(disc_predicted, disc_actual)
            adversarial_loss.backward()

            # Adversarial Loss for second pair
            f_y = f(y)
            s_disc_predicted = d_x(f_y)
            s_disc_actual = d_x(x)

            s_adversarial_loss = adversarial_loss_fn(s_disc_predicted, s_disc_actual)
            s_adversarial_loss.backward()

            optimizer.step()

        scheduler.step()

        if epoch % MODEL_SAVE_FREQ == 0:
            utils.save_checkpoint(g, f, d_x, d_y, optimizer, epoch)

    utils.save_checkpoint(g, f, d_x, d_y, optimizer, N_EPOCHS)
    #################### TRAINING ENDS HERE ####################


__all__ = ["train"]


def main():
    train()


if __name__ == "__main__":
    main()
