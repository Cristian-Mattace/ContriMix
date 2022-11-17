"""A module that defines the sample information for each patch."""
import logging
import pickle as pkl
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from itertools import product
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import cv2
import numpy as np

from ip_drit.slide import Slide

INPUT_PATCH_SIZE_PIXELS = 512


@dataclass
class Sample:
    """A class that defines information about each patch."""

    row_idx: int
    col_idx: int
    image_path: Path


def generate_sample_lists_by_domain_indices(
    dataset_info: Dict[str, Union[str, List[str], List[int]]],
    sample_folder: Path,
    patch_size_pixels: int,
    max_num_samples_per_domain: int = 100000,
) -> Dict[int, List[Sample]]:
    """Generates a dictionary of sample lists for each domain.

    Each slide will have an equal number of representative in the same list.
    The samples within each slide are shuffled. All samples from all slides in 1 domain are shuffled.

    Args:
        dataset_info: A dictionary that contains information about the dataset.
        sample_folder: A path to a folder that stores the sample information of each slide.
        patch_size_pixels: The size of each sampling patch in pixels.
        max_num_samples_per_domain (optional): The maximum number of samples per domains. Defaults to 100000.
    """
    slide_idx_list_by_domain_idx: Dict[int, List[int]] = {
        domain_idx: [
            slide_idx
            for slide_idx, slide_domain_idx in enumerate(dataset_info["domain_indices"])
            if slide_domain_idx == domain_idx
        ]
        for domain_idx in np.unique(dataset_info["domain_indices"])
    }

    num_sample_per_slide_by_domain_idx = {
        domain_idx: int(np.ceil((max_num_samples_per_domain / len(slide_idx_list))))
        for domain_idx, slide_idx_list in slide_idx_list_by_domain_idx.items()
    }

    num_sample_per_slide_by_slide_idx: Dict[int, int] = {
        slide_idx: num_sample_per_slide_by_domain_idx[domain_idx]
        for domain_idx, slide_idx_list in slide_idx_list_by_domain_idx.items()
        for slide_idx in slide_idx_list
    }

    sample_list_by_slide_idx: Dict[int, List[Sample]] = _generate_sample_list_by_slide_idx(
        image_paths=dataset_info["image_paths"],
        num_sample_per_slide_by_slide_idx=num_sample_per_slide_by_slide_idx,
        sample_folder=sample_folder,
        patch_size_pixels=patch_size_pixels,
    )

    return {
        domain_idx: _permute_samples_from_all_slides_of_one_domain(
            [sample_list_by_slide_idx[slide_index] for slide_index in slide_indices]
        )
        for domain_idx, slide_indices in slide_idx_list_by_domain_idx.items()
    }


def _generate_sample_list_by_slide_idx(
    image_paths: List[Path],
    num_sample_per_slide_by_slide_idx: Dict[int, int],
    sample_folder: Path,
    patch_size_pixels: int,
) -> Dict[int, List[Sample]]:
    """Generates the sample list keyed by the slide index.

    Args:
        image_paths: A list that conatains the paths to all images.
        num_sample_per_slide_by_slide_idx: A dictionary of number of sample per slide by slide index.
        sample_folder: A path to the folder used to save all samples.
        patch_size_pixels: The size of each sampling patch in pixels.

    Returns:
        A dictionary of lists of samples, keyed by the index of the slide.
    """
    sample_list_by_slide_idx: Dict[int, List[Sample]] = {}
    for slide_idx, image_path in enumerate(image_paths):
        sample_file_name = sample_folder / f"samples_{slide_idx}.pkl"
        num_samples_to_obtain = num_sample_per_slide_by_slide_idx[slide_idx]
        if not sample_file_name.exists():
            logging.info(f"Did not find the samples for slide with index {slide_idx}, will generate it.")
            slide_samples = _generate_wsi_sampling_points(
                image_path=image_path, num_samples_to_obtain=num_samples_to_obtain, patch_size_pixels=patch_size_pixels
            )
            with open(sample_file_name, "wb") as file:
                pkl.dump(slide_samples, file)
        else:
            with open(sample_file_name, "rb") as file:
                slide_samples = pkl.load(file)
        logging.info(f"Found {len(slide_samples)} for slide with idx {slide_idx}.")
        sample_list_by_slide_idx[slide_idx] = slide_samples
    return sample_list_by_slide_idx


def _generate_wsi_sampling_points(
    image_path: Path,
    num_samples_to_obtain: int,
    patch_size_pixels: int,
    minimum_tissue_fraction_to_be_selected: float = 0.8,
    num_candidate_over_num_samples_factor: float = 2.0,
) -> List[Sample]:
    """Generates a list of Sample for each slide.

    Args:
        image_path: The Path to the image file.
        domain_idx: The index of the domain from which the patch is.
        num_samples_to_obtain: The number of candidates per core to consider for sampling coordinates per slide.
        patch_size_pixels: The size of each sampling patch.
        minimum_tissue_fraction_to_be_selected (optional): The number that specifies the minimum fraction of tissue
            for a candidate to be selected as a sampling point. Defaults to be 0.1.
        num_candidate_over_num_samples_factor (optional): A factor that is used to count the number of candidates from
            the number of samples to obtain. Defaults to 2.0
    Returns:
        A list of Samples that contains the coordinates of the center point of each sampling patch.
    """
    downsampling_factor = 4
    tissue_mask = _he_tissue_mask(slide=Slide(file_name=image_path), downsampling_factor=downsampling_factor)
    ds_nrows, ds_ncols = tissue_mask.shape
    ds_patch_size_pixels = patch_size_pixels // downsampling_factor
    ds_half_patch_size_pixels = ds_patch_size_pixels // 2
    num_candidates = int(num_samples_to_obtain * num_candidate_over_num_samples_factor)
    grid_spacing_pixels = int(
        np.sqrt((ds_nrows - ds_patch_size_pixels) * (ds_ncols - ds_patch_size_pixels) / num_candidates)
    )

    sampling_candidates = product(
        range(ds_half_patch_size_pixels, ds_nrows - ds_half_patch_size_pixels, grid_spacing_pixels),
        range(ds_half_patch_size_pixels, ds_ncols - ds_half_patch_size_pixels, grid_spacing_pixels),
    )
    useful_tissue_fractions = map(
        lambda x: _compute_usable_tissue_fraction(x, patch_size_pixels=ds_patch_size_pixels, tissue_mask=tissue_mask),
        sampling_candidates,
    )

    return np.random.permutation(
        [
            Sample(
                row_idx=int(x[0]) * downsampling_factor, col_idx=int(x[1]) * downsampling_factor, image_path=image_path
            )
            for x, y in zip(sampling_candidates, useful_tissue_fractions)
            if y > minimum_tissue_fraction_to_be_selected
        ]
    )[:num_samples_to_obtain]


def _he_tissue_mask(slide: Slide, downsampling_factor: int) -> np.ndarray:
    """Computes the tissue mask from the H & E slide image.

    Args:
        slide: The Slide object that contains the slide image.
    """
    # Compute the tissue mask from the input slide id.
    CLOSING_KEREL_SIZE_PIXELS = 6
    im = slide[::downsampling_factor, ::downsampling_factor]
    # Due to histogram adjustment, the background value may not be the same, need to use Otsu for this.
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=5)
    sobely = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=5)
    grad_sqr = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    threshold_val, tissue_mask = cv2.threshold(grad_sqr, 0, 255, cv2.THRESH_OTSU)
    kernel = np.ones((CLOSING_KEREL_SIZE_PIXELS, CLOSING_KEREL_SIZE_PIXELS), np.uint8)
    return cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)


def _compute_usable_tissue_fraction(
    center_coords: Tuple[int, int], patch_size_pixels: int, tissue_mask: np.ndarray
) -> float:
    row_idx, col_idx = center_coords
    nrows, ncols = tissue_mask.shape
    half_patch_size_pixels = patch_size_pixels // 2
    if not half_patch_size_pixels <= row_idx <= nrows - half_patch_size_pixels:
        raise ValueError(
            f"row_idx {row_idx} must be between [{half_patch_size_pixels}, {nrows - half_patch_size_pixels}]"
        )
    if not half_patch_size_pixels <= col_idx <= ncols - half_patch_size_pixels:
        raise ValueError(
            f"row_idx {col_idx} must be between [{half_patch_size_pixels}, {ncols - half_patch_size_pixels}]"
        )

    return (
        float(
            np.sum(
                tissue_mask[
                    row_idx - half_patch_size_pixels : row_idx + half_patch_size_pixels,
                    col_idx - half_patch_size_pixels : col_idx + half_patch_size_pixels,
                ]
            )
        )
        / float(patch_size_pixels**2)
        / 255
    )


def _permute_samples_from_all_slides_of_one_domain(sample_lists_from_one_domain: List[List[Sample]]) -> List[Sample]:
    """Combines the lists of samples from all the slides, permutes, and keeps only some of them."""
    return np.random.permutation(list(chain.from_iterable(sample_lists_from_one_domain)))
