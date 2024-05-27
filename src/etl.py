"""
This module contains the ETL functions for the package.

The functions in this module are:
- extract_data: Extracts the downloaded datasets.
- load_data: Loads the datasets.
"""

import functools
import os
import pathlib
import random
import shutil
import tarfile
import zipfile
from typing import Callable

# Downloading the datasets isn't possible as both the datasets are behind an interactive button.
# There is no way to download the datasets programmatically.

DATA_DIR = os.environ.get("DATA_DIR", pathlib.Path(__file__).parent.parent / "data")

if isinstance(DATA_DIR, str):
    DATA_DIR = pathlib.Path(DATA_DIR)

DOWNLOADED_DATA_DIR = DATA_DIR / "downloaded"
EXTRACTED_DATA_DIR = DATA_DIR / "extracted"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def skip_if_directory_exists(directory: str | pathlib.Path) -> Callable:
    """
    A decorator that skips the function if the directory exists.

    Args:
        directory (str | pathlib.Path): The directory to check for existence.
    """

    def wrapper(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            if os.path.exists(directory):
                return
            return func(*args, **kwargs)

        return inner

    return wrapper


def throw_if_path_does_not_exist(path: str | pathlib.Path) -> Callable:
    """
    A decorator that throws an error if the directory does not exist.

    Args:
        path (str | pathlib.Path): The path to check for existence.
    """

    def wrapper(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} does not exist.")
            return func(*args, **kwargs)

        return inner

    return wrapper


@skip_if_directory_exists(EXTRACTED_DATA_DIR / "cartoonset100k")
@throw_if_path_does_not_exist(DOWNLOADED_DATA_DIR / "cartoonset100k.tgz")
def _extract_cartoonset():
    """
    Extracts the cartoonset100k dataset.
    """
    # Use tarfile
    with tarfile.open(DOWNLOADED_DATA_DIR / "cartoonset100k.tgz") as file:
        file.extractall(EXTRACTED_DATA_DIR)


@skip_if_directory_exists(EXTRACTED_DATA_DIR / "img_align_celeba")
@throw_if_path_does_not_exist(DOWNLOADED_DATA_DIR / "img_align_celeba.zip")
def _extract_people_faces():
    """
    Extracts the img_align_celeba dataset.
    """
    # Use zipfile
    with zipfile.ZipFile(DOWNLOADED_DATA_DIR / "img_align_celeba.zip") as file:
        file.extractall(EXTRACTED_DATA_DIR)


def extract_data():
    """
    Extracts the datasets.
    """
    _extract_cartoonset()
    _extract_people_faces()


def _load_cartoonset(dataset_size: int, test_size: int):
    """
    Loads the cartoonset100k dataset

    Process the dataset and save it in the processed data directory.
    """
    if dataset_size >= 1_000_000:
        dataset_size = 1_000_000

    SRC_DIR = EXTRACTED_DATA_DIR / "cartoonset100k"
    folders = sorted(os.listdir(SRC_DIR))

    total_image_file_names = []

    for folder in folders:
        folder_files = os.listdir(SRC_DIR / folder)
        folder_image_files = filter(lambda x: x.endswith(".png"), folder_files)
        folder_image_files = list(
            map(lambda x: f"{SRC_DIR}/{folder}/{x}", folder_image_files)
        )

        total_image_file_names.extend(folder_image_files)

    train_file_names = random.sample(total_image_file_names, dataset_size)

    test_file_names = random.sample(
        list(set(total_image_file_names) - set(train_file_names)), test_size
    )

    DST_DIR = PROCESSED_DATA_DIR / "cartoonset/train"

    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)

    for idx, file_name in enumerate(train_file_names):
        dst_file_path = DST_DIR / f"{idx}.png"

        shutil.copy(file_name, dst_file_path)

    DST_DIR = PROCESSED_DATA_DIR / "cartoonset/test"

    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)

    for idx, file_name in enumerate(test_file_names):
        dst_file_path = DST_DIR / f"{idx}.png"

        shutil.copy(file_name, dst_file_path)


def _load_people_faces(dataset_size: int, test_size: int):
    """
    Loads the img_align_celeba dataset

    Process the dataset and save it in the processed data directory.
    """
    if dataset_size >= 2_000_000:
        dataset_size = 2_000_000

    SRC_DIR = EXTRACTED_DATA_DIR / "img_align_celeba"
    file_names = sorted(os.listdir(SRC_DIR))

    train_file_names = random.sample(file_names, dataset_size)
    test_file_names = random.sample(
        list(set(file_names) - set(train_file_names)), test_size
    )

    DST_DIR = PROCESSED_DATA_DIR / "people_faces/train"

    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)

    for idx, file_name in enumerate(train_file_names):
        src_file_path = SRC_DIR / file_name
        dst_file_path = DST_DIR / f"{idx}.jpg"

        shutil.copy(src_file_path, dst_file_path)

    DST_DIR = PROCESSED_DATA_DIR / "people_faces/test"

    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)

    for idx, file_name in enumerate(test_file_names):
        src_file_path = SRC_DIR / file_name
        dst_file_path = DST_DIR / f"{idx}.jpg"

        shutil.copy(src_file_path, dst_file_path)


def load_data(dataset_size: int, test_size: int = 20):
    """
    Loads the datasets.
    """
    _load_cartoonset(dataset_size, test_size)
    _load_people_faces(dataset_size, test_size)


def main():
    DATASET_SIZE = int(os.environ.get("DATASET_SIZE", 10_000))
    TEST_SIZE = 20
    extract_data()
    load_data(DATASET_SIZE, TEST_SIZE)


if __name__ == "__main__":
    main()

__all__ = ["skip_if_directory_exists", "extract_data", "load_data"]
