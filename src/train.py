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

import albumentations as A
import albumentations.pytorch.transforms as APT
import neptune
import rich
from rich.console import Console  # type: ignore
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
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


def reduce_lr(epoch: int) -> float:
    """
    Reduces the learning rate linearly to 0 until epoch == TOTAL_EPOCHS.

    Args:
        epoch (int): The current epoch number.

    Returns:
        float: The learning rate multiplier.
    """
    TOTAL_EPOCHS = 200
    START_DECAYING_FROM = 50

    if epoch < START_DECAYING_FROM:
        return 1.0

    # Start decaying the learning rate linearly to 0 until epoch == TOTAL_EPOCHS
    return 1.0 - (epoch - START_DECAYING_FROM) / (TOTAL_EPOCHS - START_DECAYING_FROM)


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
    cyclic_loss_fn = loss.CyclicLoss()

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

    cyclic_loss_fn_size_in_mb = utils.get_parameters_size_in_mb(cyclic_loss_fn)

    total_size_in_mb = (
        g_size_in_mb
        + f_size_in_mb
        + d_y_size_in_mb
        + d_x_size_in_mb
        + adversarial_loss_fn_size_in_mb
        + cyclic_loss_fn_size_in_mb
    )

    _print_table(
        [
            ["Generator G", g_size_in_mb],
            ["Generator F", f_size_in_mb],
            ["Discriminator D_Y", d_y_size_in_mb],
            ["Discriminator D_X", d_x_size_in_mb],
            ["Adversarial Loss", adversarial_loss_fn_size_in_mb],
            ["Cyclic Loss", cyclic_loss_fn_size_in_mb],
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
        people_transform=A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                APT.ToTensorV2(always_apply=True),
            ]
        ),
        emoji_transform=A.Compose(
            [
                A.CenterCrop(475, 475, always_apply=True, p=1),
                APT.ToTensorV2(always_apply=True),
            ]
        ),
    )

    train_dl = data.DataLoader(
        train_ds, batch_size=int(os.environ.get("BATCH_SIZE", 1)), shuffle=True
    )

    N_BATCHES = len(train_dl)
    #################### DATA ENDS HERE ####################

    #################### NEPTUNE STARTS HERE ####################
    run = neptune.init_run(
        project=os.environ.get("PROJECT_NAME", "emoji-generator"),
        api_token=os.environ.get("NEPTUNE_API_TOKEN"),
        name="CycleGAN Training",
    )
    #################### NEPTUNE ENDS HERE ####################

    #################### TRAINING STARTS HERE ####################
    print(f"Starting training from epoch {START_EPOCH} to {N_EPOCHS}")
    _print_seperator_line()

    for epoch in tqdm.trange(START_EPOCH, N_EPOCHS + 1):
        g.train()
        f.train()

        d_y.train()
        d_x.train()

        F_ADVERSAIRIAL_LOSS = 0.0
        S_ADVERSARIAL_LOSS = 0.0
        ADVERSARIAL_LOSS = 0.0
        CYCLIC_LOSS = 0.0
        TOTAL_LOSS = 0.0

        for x, y in train_dl:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            # Adversarial Loss for first pair
            g_x = g(x)
            disc_predicted = d_y(g_x)
            disc_actual = d_y(y)

            adversarial_loss = adversarial_loss_fn(disc_predicted, disc_actual)
            F_ADVERSAIRIAL_LOSS += adversarial_loss.item()
            adversarial_loss.backward()

            # Adversarial Loss for second pair
            f_y = f(y)
            s_disc_predicted = d_x(f_y)
            s_disc_actual = d_x(x)

            s_adversarial_loss = adversarial_loss_fn(s_disc_predicted, s_disc_actual)
            S_ADVERSARIAL_LOSS += s_adversarial_loss.item()
            s_adversarial_loss.backward()

            ADVERSARIAL_LOSS = F_ADVERSAIRIAL_LOSS + S_ADVERSARIAL_LOSS

            # Cyclic Loss
            f_g_x = f(g_x)
            g_f_y = g(f_y)

            cyclic_loss = cyclic_loss_fn(x, y, f_g_x, g_f_y)

            CYCLIC_LOSS += cyclic_loss.item()

            cyclic_loss.backward()

            optimizer.step()

            TOTAL_LOSS = ADVERSARIAL_LOSS + CYCLIC_LOSS

        scheduler.step()

        F_ADVERSAIRIAL_LOSS /= N_BATCHES
        S_ADVERSARIAL_LOSS /= N_BATCHES
        ADVERSARIAL_LOSS /= N_BATCHES
        CYCLIC_LOSS /= N_BATCHES
        TOTAL_LOSS /= N_BATCHES

        run["Losses/First Adversarial Loss"].log(F_ADVERSAIRIAL_LOSS)
        run["Losses/Second Adversarial Loss"].log(S_ADVERSARIAL_LOSS)
        run["Losses/Adversarial Loss"].log(ADVERSARIAL_LOSS)
        run["Losses/Cyclic Loss"].log(CYCLIC_LOSS)
        run["Losses/Total Loss"].log(TOTAL_LOSS)

        if epoch % MODEL_SAVE_FREQ == 0:
            utils.save_checkpoint(g, f, d_x, d_y, optimizer, epoch)

    utils.save_checkpoint(g, f, d_x, d_y, optimizer, N_EPOCHS)
    #################### TRAINING ENDS HERE ####################


__all__ = ["train"]


def main():
    train()


if __name__ == "__main__":
    main()
