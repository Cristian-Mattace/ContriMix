"""A module that defines the public datasets from different domains.

This module is adapted from the WILDSDataset
"""
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from pathlib import Path
from abc import ABC
from absl import logging
from abc import abstractmethod
import numpy as np
import torch
import time
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from ip_drit.sampling import Sample
from ip_drit.slide import Slide
from ._utils import download_and_extract_archive
logging.set_verbosity(logging.INFO)

class AbstractPublicDataset(ABC):
    """A class that defines the interfaces for different public dataset.

    Args:
        dataset_dir: The target folder for storing the dataset.
    """
    _dataset_name: Optional[str] = None
    _version: Optional[str] = None
    _DOWNLOAD_URL_BY_VERSION: Dict[str, str] = {}
    _metadata_fields: List[str] = []

    def __init__(self, dataset_dir: Path) -> None:
        self._data_dir: Path = dataset_dir
        self._patch_size_pixels: Optional[Tuple[int, int]] = None
        self._make_sure_folder_exists()
        if not self._dataset_exists_locally():
            logging.info(f"{self._dataset_name} does not exist locally. Downloading it now!")
            self._download_dataset()

    def _make_sure_folder_exists(self) -> None:
        self._data_dir.mkdir(exist_ok=True)

    def _dataset_exists_locally(self) -> bool:
        download_url = self._DOWNLOAD_URL_BY_VERSION[self._version]
        # There are two ways to download a dataset:
        # 1. Automatically through the WILDS package
        # 2. From a third party (e.g. OGB-MolPCBA is downloaded through the OGB package)
        # Datasets downloaded from a third party need not have a download_url and RELEASE text file.
        version_file = os.path.join(self._data_dir, f'RELEASE_v{self.version}.txt')
        return os.path.exists(self._data_dir) and os.path.exists(version_file)

    def _download_dataset(self) -> None:
        download_url = self._DOWNLOAD_URL_BY_VERSION[self._version]

        #from wilds.datasets.download_utils import download_and_extract_archive
        logging.info(f'Downloading dataset to {self._data_dir}...')
        logging.info(f'You can also download the dataset manually at https://wilds.stanford.edu/downloads.')

        try:
            start_time = time.time()
            download_and_extract_archive(
                url=download_url,
                download_root=self._data_dir,
                extract_root=self._data_dir,
                filename='archive.tar.gz',
                remove_finished=True)

            logging.info(f"\nIt took {round((time.time() - start_time) / 60, 2)} minutes to download.\n")
        except Exception as e:
            logging.info(f"\n{os.path.join(self._data_dir, 'archive.tar.gz')} may be corrupted. Please try deleting it and rerunning this command.\n")
            raise e

    @property
    def dataset_name(self) -> str:
        """Returns the name of the dataset."""
        return self._dataset_name

    @property
    def dataset_dir(self) -> Path:
        """Returns the location of the dataset."""
        return self._data_dir

    @property
    def version(self) -> str:
        return self._version

    @property
    def patch_size_pixel(self) -> Tuple[int, int]:
        return self._patch_size_pixels
    
    def get_input(self, idx: int) -> np.ndarray:
        """
        Gets the input image with index 'idx'.

        Args:
            idx: The index of the input image to get.

        Returns:
            A numpy array of the loaded image.
        """
        raise NotImplementedError

    @property
    def split_dict(self) -> Optional[Dict[str, int]]:
        return getattr(self, "_SPLIT_INDEX_BY_SPLIT_STRING", None)

    @property
    def metadata_map(self) -> Optional[Dict[str, Any]]:
        """
        Returns a dictionary that, for each metadata field, contains a list that maps from
        integers (in metadata_array) to a string representing what that integer means.
        This is only used for logging, so that we print out more intelligible metadata values.
        Each key must be in metadata_fields.
        For example, if we have
            metadata_fields = ['hospital', 'y']
            metadata_map = {'hospital': ['East', 'West']}
        then if metadata_array[i, 0] == 0, the i-th data point belongs to the 'East' hospital
        while if metadata_array[i, 0] == 1, it belongs to the 'West' hospital.
        """
        return getattr(self, '_metadata_map', None)

    @property
    def metadata_fields(self) -> List[str]:
        """
        A list of strings naming each column of the metadata table, e.g., ['hospital', 'y'].
        Must include 'y'.
        """
        return self._metadata_fields

    @property
    def split_array(self) -> np.ndarray:
        """
        An array of integers, with split_array[i] representing what split the i-th data point
        belongs to.
        """
        return self._split_array

    @property
    def collate(self) -> Optional[Callable]:
        """
        Torch function to collate items in a batch.
        By default, returns None -> uses default torch collate.
        """
        return getattr(self, '_collate', None)

    @property
    def metadata_array(self):
        """
        A Tensor of metadata, with the i-th row representing the metadata associated with
        the i-th data point. The columns correspond to the metadata_fields defined above.
        """
        return self._metadata_array
    def get_subset(self, split: str, frac: float=1.0, transform: Optional[Callable] =None):
        """
        Return a subset of a dataset based on the split definition.

        Args:
            split: A Split identifier, e.g., 'train', 'ood_val', 'test', 'id_val'. Must be a key in in self.split_dict.
            frac (optional): What fraction of the split to randomly sample. Used for fast development on a small dataset.
                Defaults to 1.0.
            transform (optional): Any data transformations to be applied to the input x. Defaults to None, in which case,
                no transformation will be performed.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split name {split} is not found in dataset's split_dict.")

        split_idx = np.where(self.split_array == self.split_dict[split])[0]

        if frac < 1.0:
            # Randomly sample a fraction of the split
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])

        return SubsetPublicDataset(self, split_idx, transform)

class SubsetPublicDataset(AbstractPublicDataset):
    """
    A class that acts like `torch.utils.data.Subset`.
    We pass in `transform` (which is used for data augmentation) explicitly
    because it can potentially vary on the training vs. test subsets.

    `do_transform_y` (bool):

    Args:
        full_dataset: The full dataset that includes the datapoints from all splits

        do_transform_y: When this is false (the default), `self.transform ` acts only on  `x`. Set this to true if
            self.transform` should operate on `(x,y)` instead of just `x`.
    """
    def __init__(self,
                 full_dataset: AbstractPublicDataset,
                 indices: List[int],
                 transform,
                 do_transform_y: bool=False):
        self._dataset = full_dataset
        self._indices = indices
        inherited_attrs = ['_dataset_name', '_data_dir', '_collate',
                           '_split_scheme', '_split_dict', '_split_names',
                           '_y_size', '_n_classes',
                           '_metadata_fields', '_metadata_map']
        for attr_name in inherited_attrs:
            if hasattr(self._dataset, attr_name):
                setattr(self, attr_name, getattr(self._dataset, attr_name))
        self._transform = transform
        self._do_transform_y = do_transform_y

    def __getitem__(self, idx):
        x, y, metadata = self._dataset[self._indices[idx]]
        if self._transform is not None:
            if self._do_transform_y:
                x, y = self._transform(x, y)
            else:
                x = self._transform(x)
        return x, y, metadata

    def __len__(self):
        return len(self._indices)

    @property
    def split_array(self):
        return self._dataset._split_array[self._indices]

    @property
    def y_array(self):
        return self._dataset._y_array[self._indices]

    @property
    def metadata_array(self):
        return self._dataset.metadata_array[self._indices]

    def eval(self, y_pred, y_true, metadata):
        return self._dataset.eval(y_pred, y_true, metadata)


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
