"""
This module contains the dataset class for the package.

The PeopleEmojiDataset class is a dataset class that returns a tuple of people
and emoji images. The people images are of size (3, 178, 218) while the emoji
images are of size (3,475,475).
"""

import os
import pathlib
from typing import Self, Tuple

import torch
import torch.utils.data as data
import torchvision.io as tv_io

try:
    if os.environ.get("ENV", "development") == "production":
        pass
    else:
        pass
except ImportError:
    pass


class PeopleEmojiDataset(data.Dataset):
    """
    Dataset for people and emoji images.

    This dataset returns a tuple of people and emoji images.
    The people images are of size (3, 178, 218) while the emoji images are of
    size (3,500,500)

    Expects the following directory structure for the people and emoji images:
        people_images_split_dir
        ├── train
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── test
            ├── image1.jpg
            ├── image2.jpg
            └── ...
    """

    def __init__(
        self: Self,
        people_images_split_dir: str | pathlib.Path,
        emoji_images_split_dir: str | pathlib.Path,
        people_transform=None,
        emoji_transform=None,
    ) -> None:
        """
        Initializes the PeopleEmojiDataset

         Args:
            people_images_split_dir (str | pathlib.Path): The directory containing
            the people images.
            emoji_images_split_dir (str | pathlib.Path): The directory containing
            the emoji images.
            people_transform (Callable): A callable that takes in an image and
            returns a transformed image. Use only Albumentations transforms.
            emoji_transform (Callable): A callable that takes in an image and
            returns a transformed image. Use only Albumentations transforms.
        """
        super(PeopleEmojiDataset, self).__init__()

        self.people_images_split_dir = people_images_split_dir
        self.emoji_images_split_dir = emoji_images_split_dir

        self.people_transform = people_transform
        self.emoji_transform = emoji_transform

        self.people_images_filenames = sorted(
            os.listdir(os.path.join(self.people_images_split_dir, "train"))
        )
        self.emoji_images_filenames = sorted(
            os.listdir(os.path.join(self.emoji_images_split_dir, "train"))
        )

        self.min_length = min(
            len(self.people_images_filenames), len(self.emoji_images_filenames)
        )

    def __len__(self: Self) -> int:
        """
        Returns the number of elements in the dataset

        Returns:
            int: the length of the dataset (i.e. the number of rows)
        """
        return max(len(self.people_images_filenames), len(self.emoji_images_filenames))

    def __getitem__(self: Self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the people and emoji images at the given index

        Args:
            index (int): The index of the images to return

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the people and
            emoji images
        """
        people_idx = index % len(self.people_images_filenames)
        emoji_idx = index % len(self.emoji_images_filenames)

        people_image_file_name = os.path.join(
            self.people_images_split_dir,
            "train",
            self.people_images_filenames[people_idx],
        )
        emoji_image_file_name = os.path.join(
            self.emoji_images_split_dir, "train", self.emoji_images_filenames[emoji_idx]
        )

        people_image = tv_io.read_image(people_image_file_name)
        emoji_image = tv_io.read_image(emoji_image_file_name)

        emoji_image = emoji_image[:3, :, :]

        if self.people_transform is not None:
            people_image = self.people_transform(people_image)

        if self.emoji_transform is not None:
            emoji_image = self.emoji_transform(emoji_image)

        return people_image, emoji_image


__all__ = ["PeopleEmojiDataset"]


def main():
    pass


if __name__ == "__main__":
    main()
