"""A module tat defines a unlabel dataset."""
from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch

from ._dataset import AbstractPublicDataset
from ._dataset import SubsetLabeledPublicDataset


class AbstractUnlabelPublicDataset(AbstractPublicDataset):
    """A class that defines the interfaces for unlabeled public dataset.

    Args:
        dataset_dir: The target folder for storing the dataset.
        transform (optional): The transform to apply on the whole dataset. Defaults to None.
    """

    def __init__(self, dataset_dir: Path, transform: Optional[Callable] = None) -> None:
        super().__init__(dataset_dir, transform)
        self._pseudolabels: Optional[torch.Tensor] = None

    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, torch.Tensor], Tuple[np.ndarray, int, torch.Tensor]]:
        x = self._get_input(idx)
        if self._transform is not None:
            x = self._transform(x)
        if self._pseudolabels is None:
            return x, self.metadata_array[idx]
        else:
            return x, self._pseudolabels[idx], self.metadata_array[idx]

    def _get_subset_from_split_indices(
        self, split_idx: np.ndarray, transform: Optional[Callable] = None
    ) -> "SubsetUnlabeledPublicDataset":
        return SubsetUnlabeledPublicDataset(self, split_idx, transform)

    @property
    def pseudolabels(self) -> torch.Tensor:
        if self._pseudolabels is None:
            raise ValueError(f"Pseudolabels are not yet calculated for the {self.__class__.__name__}")
        else:
            return self._pseudolabels

    @pseudolabels.setter
    def pseudolabels(self, labels: torch.Tensor) -> None:
        self._pseudolabels = labels


class SubsetUnlabeledPublicDataset(SubsetLabeledPublicDataset):
    """A class that acts like `torch.utils.data.Subset` for subset of another dataset.

    We pass in `transform` (which is used for data augmentation) explicitly
    because it can potentially vary on the training vs. test subsets.

    Args:
        full_dataset: The full unlabeled dataset that includes the datapoints from all splits
        indices: A list of int that specifies the indices in the big dataset to use for the subdataset.
        do_transform_y: When this is false (the default), `self.transform ` acts only on  `x`. Set this to true if
            self.transform` should operate on `(x,y)` instead of just `x`.
    """

    def __getitem__(self, idx: int) -> Union[Tuple[int, int, np.ndarray], Tuple[int, np.ndarray]]:
        return self._dataset[self._indices[idx]]
