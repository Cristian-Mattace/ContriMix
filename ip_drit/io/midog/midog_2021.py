"""A module that provides functions relating to the MIDOG dataset."""
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

from .._data_download_utils import dowload_image_from_url
from .._data_download_utils import download_metadata


def dowload_midog_2021_dataset_into_a_folder_return_data_paths(
    target_folder: Path,
) -> Dict[str, Union[List[Path], Path, List[int]]]:
    """Downloads the MIDOG 2021 dataset into a folder.

    Args:
        target_folder: The path to the target folder to download the data to.

    Returns:
        A dictionary with:
            A field name 'image_paths' contains a list of paths to the downloaded images.
            A field name 'domain_indices' contains the domain index of each image.
            A field name 'metadata_path' contains a path to the metadata of the dataset.
    """
    num_images = 200
    first_image_idx = 1
    image_paths: List[Path] = []
    domain_indices: List[int] = []
    for im_idx in range(first_image_idx, first_image_idx + num_images):
        url = f"https://zenodo.org/record/4643381/files/{im_idx:03d}.tiff?download=1"
        image_paths.append(
            dowload_image_from_url(url=url, target_folder=target_folder, file_name_extraction_func=_url_to_file_name)
        )
        domain_indices.append(_domain_index_from_image_index(image_idx=im_idx))

    metadata_path = download_metadata(
        url="https://zenodo.org/record/4643381/files/MIDOG.json?download=1", target_folder=target_folder
    )
    return {"image_paths": image_paths, "domain_indices": domain_indices, "metadata_path": metadata_path}


def _domain_index_from_image_index(image_idx: int) -> int:
    """Returns the domain index of an image givens the image_idx."""
    if 1 <= image_idx < 51:
        return 0
    elif 51 <= image_idx < 101:
        return 1
    elif 101 <= image_idx < 151:
        return 2
    else:
        return 3


def _url_to_file_name(url: str) -> str:
    """Returns the file name from a URL for an image."""
    return url.split("?")[0].split("/")[-1]
