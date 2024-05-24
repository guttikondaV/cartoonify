"""
This module contains the inference functions for the package.

The functions in this module are:
- infer: Infers the generated images from the generator network.
"""
import os
import pathlib

import matplotlib.pyplot as plt
import torch
import torchvision.io as tv_io

try:
    if os.environ.get("ENV", "development") == "production":
        import src.generator as generator
        import src.utils as utils
    else:
        import generator
        import utils
except ImportError:
    pass


# Used to infer raw images in a given directory


def infer(images_dir: pathlib.Path | str):
    """
    Infer the generated images from the generator network.

    Args:
        images_dir (pathlib.Path | str): The directory containing the images to
        infer.
    """
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

    PREDICTIONS_DIR = os.environ.get(
        "PREDICTIONS_DIR", pathlib.Path(__file__).parent.parent / "predictions"
    )

    if isinstance(PREDICTIONS_DIR, str):
        PREDICTIONS_DIR = pathlib.Path(PREDICTIONS_DIR)

    if not os.path.exists(PREDICTIONS_DIR):
        os.makedirs(PREDICTIONS_DIR)

    checkpoint = utils.load_checkpoint()
    g = generator.Generator()

    g.load_state_dict(checkpoint["g_state_dict"])

    g.eval()
    g.to(DEVICE)

    # Inference code here
    file_names = sorted(os.listdir(images_dir))

    if len(file_names) == 0:
        return

    file_names = list(filter(lambda x: x.endswith(".jpg"), file_names))

    if len(file_names) == 0:
        return

    for file_name in file_names:
        file_path = os.path.join(images_dir, file_name)

        image = tv_io.read_image(file_path)

        image = image.to(DEVICE)
        image = image / 255.0
        image = image.unsqueeze(0)

        with torch.no_grad():
            prediction = g(image)

            prediction = prediction.squeeze(0)

            prediction = prediction.detach()
            prediction = prediction.to("cpu")
            image = image.to("cpu")

            # Start plotting
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(image[0].permute(1, 2, 0).numpy())
            axs[0].set_title("Original Image")
            axs[0].axis("off")
            axs[1].imshow(prediction.permute(1, 2, 0).numpy())
            axs[1].set_title("Generated Image")
            axs[1].axis("off")
            plt.savefig(PREDICTIONS_DIR / f"{file_name.split('.')[0]}_prediction.jpg")
            plt.close()


__all__ = ["infer"]


def main():
    infer("/Users/varunguttikonda/kaggle/cartoonify/data/processed/people_faces/test")


if __name__ == "__main__":
    main()
