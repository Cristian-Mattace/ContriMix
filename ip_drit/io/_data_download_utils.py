"""Helpers functions to download files from URLs."""
import logging
import shutil
from pathlib import Path
from typing import Callable

import requests


def dowload_image_from_url(url: str, target_folder: Path, file_name_extraction_func: Callable) -> Path:
    """Downloads an image from the URL into the target folder.

    Args:
        url: The URL of the image to download.
        target_folder: The target folder that the image will be downloaded to.

    Returns:
        A Path object that points to the location of the download file on the computer.
    """
    file_name = file_name_extraction_func(url)
    target_file_name = target_folder / file_name
    _download_file_from_url_into_a_target_location(url=url, target_file_name=target_file_name)
    return target_file_name


def _download_file_from_url_into_a_target_location(url: str, target_file_name: Path) -> None:
    if not target_file_name.exists():
        with requests.get(url, stream=True) as r:
            logging.info(f"Download file from url {url}")
            # If image download was successful
            if r.status_code == 200:
                # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
                r.raw.decode_content = True
                with open(target_file_name, "wb") as file:
                    shutil.copyfileobj(r.raw, file)
                logging.info("Successful.")
            else:
                logging.info("Failed.")


def download_metadata(url: str, target_folder: Path) -> Path:
    """Downloads an metadata file from the URL into the target folder.

    Args:
        url: The URL of the image to download.
        target_folder: The target folder that the image will be downloaded to.

    Returns:
        A Path object that points to the location of the download file on the computer.
    """
    file_name = "metadata.json"
    target_folder.mkdir(parents=True, exist_ok=True)
    target_file_name = target_folder / file_name
    _download_file_from_url_into_a_target_location(url=url, target_file_name=target_file_name)
    return target_file_name
