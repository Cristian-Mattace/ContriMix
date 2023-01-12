"""A module that defines the public datasets from different domains.

This module is adapted from the WILDSDataset
"""
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from abc import ABC
from abc import abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from ip_drit.sampling import Sample
from ip_drit.slide import Slide


class AbstractPublicDataset(ABC):
    """A class that defines the interfaces for different public dataset."""
    def __init__(self) -> None:
        self._dataset_name: Optional[str] = None
        self._dataset_dir: Optional[str] = None

    @property
    def dataset_name(self) -> str:
        """Returns the name of the dataset."""
        return self._dataset_name

    @property
    def dataset_dir(self) -> str:
        """Returns the location of the dataset."""
        return self._dataset_dir


class MultiDomainDataset(Dataset):
    """A custom dataset that contains images from different domains.

    The domain index is round-robin. Meaning, the dataset will return [sample from domain 1][sample from domain 2] ...
    and so on.

    Args:
        sample_list_by_domain_index: A dictionary of sample list keyed by the index of the domain.
        transforms: A list of transforms to be applied on the sample.
        input_patch_size_pixels: The size of the input patch in pixels. This is the patch size.
        domain_indices (optional): A list that specifies which domains to sample from. Default to None,
            in which case, samples will be obtained from all domains.
    """

    TMA_MPP = 0.2522
    _HDF5_CHUNK_SIZE = 256
    _TMA_DOMAIN_INDEX = 0

    def __init__(
        self,
        sample_list_by_domain_index: Dict[int, List[Sample]],
        transforms: List[object],
        input_patch_size_pixels: int,
        domain_indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self._transforms = Compose(transforms)
        self._check_all_the_domains_have_an_equal_number_of_samples(sample_list_by_domain_index)
        self._sample_list_by_domain_index: Dict[int, List[Sample]] = sample_list_by_domain_index
        self._num_samples_per_domain = len(next(iter(self._sample_list_by_domain_index.values())))
        self._num_domains = len(self._sample_list_by_domain_index)
        self._input_patch_size_pixels = input_patch_size_pixels
        self._num_domains = len(self._sample_list_by_domain_index)
        self._domain_indices = domain_indices
        if self._domain_indices is None:
            self._domain_indices = list(self._sample_list_by_domain_index.keys())

    @staticmethod
    def _check_all_the_domains_have_an_equal_number_of_samples(
        sample_list_by_domain_index: Dict[int, List[Sample]]
    ) -> None:
        if len(set(len(x) for x in sample_list_by_domain_index.values())) > 1:
            raise ValueError("Not every domain have the same number of samples.")

    def __len__(self):
        return self._num_samples_per_domain * self._num_domains

    def __getitem__(self, idx: int) -> np.ndarray:
        """Returns an image with the slide index `idx`."""
        domain_idx = idx % self._num_domains
        sample_idx_in_domain = idx // self._num_domains
        return self.get_item_with_domain_idx(sample_idx=sample_idx_in_domain, domain_idx=domain_idx)

    def get_item_with_domain_idx(self, sample_idx: int, domain_idx: int) -> torch.Tensor:
        """Returns a slide platform image patch."""
        all_domain_indices = list(self._sample_list_by_domain_index.keys())
        sample = self._sample_list_by_domain_index[all_domain_indices[domain_idx]][sample_idx]
        im = self._get_slide_platform_patch(sample, domain_idx)
        return self._transforms(im)

    def _get_slide_platform_patch(self, sample: Sample, domain_index: int) -> np.ndarray:
        """Gets a patch from the slide platform."""
        slide = Slide(file_name=sample.image_path, domain_index=domain_index)
        return slide[
            sample.row_idx - self._input_patch_size_pixels // 2 : sample.row_idx + self._input_patch_size_pixels // 2,
            sample.col_idx - self._input_patch_size_pixels // 2 : sample.col_idx + self._input_patch_size_pixels // 2,
        ]
