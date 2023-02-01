"""A module that defines the public datasets from different domains.

This module is adapted from the WILDSDataset
"""
import os
import time
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from absl import logging

from ._utils import download_and_extract_archive
from ip_drit.common.metrics import Metric

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
        # There are two ways to download a dataset:
        # 1. Automatically through the WILDS package
        # 2. From a third party (e.g. OGB-MolPCBA is downloaded through the OGB package)
        # Datasets downloaded from a third party need not have a download_url and RELEASE text file.
        version_file = os.path.join(self._data_dir, f"RELEASE_v{self.version}.txt")
        return os.path.exists(self._data_dir) and os.path.exists(version_file)

    def _download_dataset(self) -> None:
        download_url = self._DOWNLOAD_URL_BY_VERSION[self._version]

        # from wilds.datasets.download_utils import download_and_extract_archive
        logging.info(f"Downloading dataset to {self._data_dir}...")
        logging.info(f"You can also download the dataset manually at https://wilds.stanford.edu/downloads.")

        try:
            start_time = time.time()
            download_and_extract_archive(
                url=download_url,
                download_root=self._data_dir,
                extract_root=self._data_dir,
                filename="archive.tar.gz",
                remove_finished=True,
            )

            logging.info(f"\nIt took {round((time.time() - start_time) / 60, 2)} minutes to download.\n")
        except Exception as e:
            logging.info(
                f"\n{os.path.join(self._data_dir, 'archive.tar.gz')} may be corrupted. Please try deleting it and "
                + "rerunning this command.\n"
            )
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
        """Gets the input image with index 'idx'.

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
        """Returns a dictionary for the metadata mapping.

        For each metadata field, contains a list that maps from
        integers (in metadata_array) to a string representing what that integer means.
        This is only used for logging, so that we print out more intelligible metadata values.
        Each key must be in metadata_fields.
        For example, if we have
            metadata_fields = ['hospital', 'y']
            metadata_map = {'hospital': ['East', 'West']}
        then if metadata_array[i, 0] == 0, the i-th data point belongs to the 'East' hospital
        while if metadata_array[i, 0] == 1, it belongs to the 'West' hospital.
        """
        return getattr(self, "_metadata_map", None)

    @property
    def metadata_fields(self) -> List[str]:
        """A list of strings naming each column of the metadata table, e.g., ['hospital', 'y'], must include 'y'."""
        return self._metadata_fields

    @property
    def split_array(self) -> np.ndarray:
        """An array of integers, with split_array[i] representing what split the i-th data point."""
        return self._split_array

    @property
    def collate(self) -> Optional[Callable]:
        """A torch function to collate items in a batch.

        By default, returns None -> uses default torch collate.
        """
        return getattr(self, "_collate", None)

    @property
    def metadata_array(self) -> torch.Tensor:
        """A Tensor of metadata.

        The i-th row representing the metadata associated with the i-th data point. The columns correspond to the
        metadata_fields defined above.
        """
        return self._metadata_array

    def get_subset(
        self, split: str, frac: float = 1.0, transform: Optional[Callable] = None
    ) -> Union["AbstractPublicDataset", "SubsetPublicDataset"]:
        """Return a subset of a dataset based on the split definition.

        Args:
            split: A Split identifier, e.g., 'train', 'ood_val', 'test', 'id_val'. Must be a key in in self.split_dict.
            frac (optional): What fraction of the split to randomly sample. Used for fast development on a small
                dataset. Defaults to 1.0.
            transform (optional): Any data transformations to be applied to the input x. Defaults to None, in which
                case, no transformation will be performed.

        Returns
            A (potentially subsampled) subset of the WILDSDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split name {split} is not found in dataset's split_dict.")

        split_idx = np.where(self.split_array == self.split_dict[split])[0]

        if frac < 1.0:
            # Randomly sample a fraction of the split
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])

        return SubsetPublicDataset(self, split_idx, transform)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self.y_array[idx]
        metadata = self.metadata_array[idx]
        return x, y, metadata

    @property
    def y_array(self) -> torch.Tensor:
        """A Tensor of targets (e.g., labels for classification tasks).

        Here, y_array[i] represents the target of the i-th data point. y_array[i] can contain multiple elements.
        """
        return self._y_array

    @property
    def original_resolution(self) -> Tuple[int, int]:
        """Original image resolution for image datasets."""
        return getattr(self, "_original_resolution", None)

    @abstractmethod
    def eval(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        metadata: torch.Tensor,
        prediction_fn: Optional[Callable] = None,
    ) -> Tuple[Dict, str]:
        """Evaluates a results based on the network prediction and network groundtruth.

        Args:
            y_pred: Predicted targets or predicted logit.
            y_true: True targets
            metadata: Metadata
            prediction_fn (optional): A function that converts the logits to the predicted label.

        Returns:
            A dictionary of results
            A pretty print version of the results
        """
        raise NotImplementedError

    @staticmethod
    def _standard_group_eval(
        metric: Metric,
        grouper,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        metadata: torch.Tensor,
        aggregate: bool = True,
    ) -> Tuple[Dict, str]:
        """Standard group evaluation.

        Args:
            metric: Metric to use for eval
            grouper: Grouper object that converts metadata into groups
            y_pred: Predicted targets
            y_true: True targets
            metadata: Metadata
            aggregate (optional): If True, aggregate the results. Defaults to True.

        Returns:
            A dictionary of results.
            A pretty print version of the results
        """
        results, results_str = {}, ""
        if aggregate:
            results.update(metric.compute(y_pred, y_true))
            results_str += f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"

        g = grouper.metadata_to_group_indices(metadata)
        group_results = metric.compute_group_wise(y_pred, y_true, g, grouper.n_groups)
        for group_idx in range(grouper.n_groups):
            group_str = grouper.group_field_str(group_idx)
            group_metric = group_results[metric.group_metric_field(group_idx)]
            group_counts = group_results[metric.group_count_field(group_idx)]
            results[f"{metric.name}_{group_str}"] = group_metric
            results[f"count_{group_str}"] = group_counts
            if group_results[metric.group_count_field(group_idx)] == 0:
                continue
            results_str += (
                f"  {grouper.group_str(group_idx)}  "
                f"[n = {group_results[metric.group_count_field(group_idx)]:6.0f}]:\t"
                f"{metric.name} = {group_results[metric.group_metric_field(group_idx)]:5.3f}\n"
            )
        results[f"{metric.worst_group_metric_field}"] = group_results[f"{metric.worst_group_metric_field}"]
        results_str += f"Worst-group {metric.name}: {group_results[metric.worst_group_metric_field]:.3f}\n"
        return results, results_str


class SubsetPublicDataset(AbstractPublicDataset):
    """A class that acts like `torch.utils.data.Subset` for subset of another dataset.

    We pass in `transform` (which is used for data augmentation) explicitly
    because it can potentially vary on the training vs. test subsets.

    Args:
        full_dataset: The full dataset that includes the datapoints from all splits
        indices: A list of int that specifies the indices in the big dataset to use for the subdataset.
        do_transform_y: When this is false (the default), `self.transform ` acts only on  `x`. Set this to true if
            self.transform` should operate on `(x,y)` instead of just `x`.
    """

    def __init__(
        self, full_dataset: AbstractPublicDataset, indices: List[int], transform, do_transform_y: bool = False
    ) -> None:
        self._dataset = full_dataset
        self._indices = indices
        inherited_attrs = [
            "_dataset_name",
            "_data_dir",
            "_collate",
            "_split_scheme",
            "_split_dict",
            "_split_names",
            "_y_size",
            "_n_classes",
            "_metadata_fields",
            "_metadata_map",
        ]
        for attr_name in inherited_attrs:
            if hasattr(self._dataset, attr_name):
                setattr(self, attr_name, getattr(self._dataset, attr_name))
        self._transform = transform
        self._do_transform_y = do_transform_y

    def __getitem__(self, idx: int) -> Tuple[int, int, np.ndarray]:
        x, y, metadata = self._dataset[self._indices[idx]]
        if self._transform is not None:
            if self._do_transform_y:
                x, y = self._transform(x, y)
            else:
                x = self._transform(x)
        return x, y, metadata

    def __len__(self) -> int:
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

    def eval(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        metadata: torch.Tensor,
        prediction_fn: Optional[Callable] = None,
    ) -> Tuple[Dict, str]:
        return self._dataset.eval(y_pred, y_true, metadata, prediction_fn=prediction_fn)
