"""A module that defines the dataloader."""
import logging
from enum import auto
from enum import Enum
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from ip_drit.common.grouper import AbstractGrouper
from ip_drit.datasets import AbstractPublicDataset
from ip_drit.datasets import SubsetPublicDataset


class LoaderType(Enum):
    """An enum class that defines the type of the loader."""

    STANDARD = auto()
    GROUP = auto()


def get_train_loader(
    loader_type: LoaderType,
    dataset: SubsetPublicDataset,
    batch_size: int,
    uniform_over_groups: Optional[bool] = None,
    grouper: Optional[AbstractGrouper] = None,
    distinct_groups: bool = True,
    train_n_groups_per_batch: int = None,
    **loader_kwargs,
) -> DataLoader:
    """Constructs and returns the data loader for training.

    Args:
        loader_type: Loader type. This can be LoaderType.STANDARD for standard loaders and LoaderType.GROUP for group
            loaders. The later first samples groups and then samples a fixed number of examples belonging to each group.
        dataset: The dataset that we want to load the data from. This is normally a subset of the full dataset.
        batch_size: Batch size.
        uniform_over_groups (optional): Whether to sample the groups uniformly or according to the natural data
            distribution. Setting to None applies the defaults for each type of loaders. For standard loaders, the
            default is False. For group loaders, the default is True.
        grouper (optional): Grouper used for group loaders or for uniform_over_groups=True
        distinct_groups (optional): Whether to sample distinct_groups within each minibatch for group loaders.
        train_n_groups_per_batch (optional): Number of groups to sample in each minibatch for group loaders.
        loader_kwargs: kwargs passed into torch DataLoader initialization.

    Output:
        The generated Data loader object.
    """
    if loader_type == LoaderType.STANDARD:
        if uniform_over_groups is None or not uniform_over_groups:
            return DataLoader(
                dataset,
                shuffle=True,  # Shuffle training dataset
                sampler=None,
                collate_fn=dataset.collate,
                batch_size=batch_size,
                **loader_kwargs,
            )
        else:
            _validate_grouper_availability(uniform_over_groups=uniform_over_groups, grouper=grouper)
            groups, group_counts = grouper.metadata_to_group_indices(dataset.metadata_array, return_counts=True)
            group_weights = 1 / group_counts
            weights = group_weights[groups]
            return DataLoader(
                dataset,
                shuffle=False,  # The WeightedRandomSampler already shuffles
                # Replacement needs to be set to True, otherwise we'll run out of minority samples
                sampler=WeightedRandomSampler(weights, len(dataset), replacement=True),
                collate_fn=dataset.collate,
                batch_size=batch_size,
                **loader_kwargs,
            )

    elif loader_type == LoaderType.GROUP:
        if uniform_over_groups is None:
            uniform_over_groups = True

        _validate_grouper_availability(uniform_over_groups=uniform_over_groups, grouper=grouper)

        group_indices_all_datapoints = grouper.metadata_to_group_indices(dataset.metadata_array)
        num_groups_available = len(np.unique(group_indices_all_datapoints.numpy()))

        logging.info(f"Train data loader has {num_groups_available} groups.")
        if train_n_groups_per_batch > num_groups_available:
            raise ValueError(
                f"'train_n_groups_per_batch' was set to {train_n_groups_per_batch} for the training dataloader "
                + f"but there are at most {num_groups_available} groups available."
            )

        return DataLoader(
            dataset,
            shuffle=None,
            sampler=None,
            collate_fn=dataset.collate,
            batch_sampler=GroupSampler(
                group_idxs=group_indices_all_datapoints,
                batch_size=batch_size,
                n_groups_per_batch=train_n_groups_per_batch,
                uniform_over_groups=uniform_over_groups,
                distinct_groups=distinct_groups,
            ),
            drop_last=False,
            **loader_kwargs,
        )


def _validate_grouper_availability(uniform_over_groups: bool, grouper: Optional[AbstractGrouper]) -> None:
    if uniform_over_groups and grouper is None:
        raise ValueError("Grouper can't be None when uniform_over_groups is True")


def get_eval_loader(
    loader_type: LoaderType, dataset: SubsetPublicDataset, batch_size: int, **loader_kwargs
) -> DataLoader:
    """Constructs and returns the data loader for evaluation.

    Args:
        loader_type: The dataloader type 'standard' for standard loaders.
        dataset: A subddataset
        batch_size: Batch size
        loader_kwargs: kwargs passed into torch DataLoader initialization.

    Returns:
        A data loader for evaluation
    """
    if loader_type == LoaderType.STANDARD:
        return DataLoader(
            dataset,
            shuffle=False,  # Do not shuffle eval datasets
            sampler=None,
            collate_fn=dataset.collate,
            batch_size=batch_size,
            **loader_kwargs,
        )
    else:
        raise ValueError(f"The type of evaluation loader {loader_type.name} is not supported!")


class GroupSampler:
    """Constructs batches by first sampling groups, then sampling data from those groups.

    It drops the last batch if it's incomplete.

    Args:
        group_idxs: A tensor that specifies the indices of different data points.
        batch_size: The size of the loading batch.
        n_groups_per_batch: The number of group per batch.
        uniform_over_groups: Whether to sample the groups uniformly or according to the natural data distribution.
            Setting to None applies the defaults for each type of loaders. For standard loaders, the
            default is False. For group loaders, the default is True.
        distinct_groups (optional): Whether to sample distinct_groups within each minibatch for group loaders.
    """

    def __init__(
        self,
        group_idxs: torch.Tensor,
        batch_size: int,
        n_groups_per_batch: int,
        uniform_over_groups: bool = True,
        distinct_groups: bool = True,
    ):

        if batch_size % n_groups_per_batch != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be evenly divisible by n_groups_per_batch ({n_groups_per_batch})."
            )
        if len(group_idxs) < batch_size:
            raise ValueError(
                f"The dataset has only {len(group_idxs)} examples but the batch size is {batch_size}. "
                + "There must be enough examples to form at least one complete batch."
            )

        self._unique_groups, self._unique_group_element_indices, unique_group_counts = _split_into_groups(group_idxs)

        self._distinct_groups: bool = distinct_groups
        self._n_groups_per_batch: int = n_groups_per_batch
        self._n_points_per_group: int = batch_size // n_groups_per_batch

        if n_groups_per_batch > len(group_idxs):
            raise ValueError(
                f"Can't generate a group sampler with n_groups_per_batch ({n_groups_per_batch}) larger "
                + f" than the number of available group {len(group_idxs)}"
            )

        self._num_batches = len(group_idxs) // batch_size

        logging.info(
            f"GroupSampler initialized with uniform_over_group = {uniform_over_groups}, "
            + f"batch size = {batch_size}, num batches = {self._num_batches}, number of points per groups = "
            + f"{self._n_points_per_group}, number of groups per batch = {n_groups_per_batch}."
        )

        if uniform_over_groups:
            self._group_prob = None
        else:
            self._group_prob = self._group_propability_from_group_counts(unique_group_counts.numpy())

    def __iter__(self) -> Generator:
        for _ in range(self._num_batches):
            # Note that we are selecting group indices rather than groups
            unique_group_idxes_for_batch = np.random.choice(
                len(self._unique_groups),
                size=self._n_groups_per_batch,
                replace=(not self._distinct_groups),
                p=self._group_prob,
            )

            sampled_ids = [
                np.random.choice(
                    self._unique_group_element_indices[group_idx],
                    size=self._n_points_per_group,
                    # If the size of the group is not larger than the number points per group reguired, do replacement.
                    replace=len(self._unique_group_element_indices[group_idx]) <= self._n_points_per_group,
                    p=None,
                )
                for group_idx in unique_group_idxes_for_batch
            ]

            yield np.concatenate(sampled_ids)

    def __len__(self):
        return self._num_batches

    @staticmethod
    def _group_propability_from_group_counts(group_count: np.ndarray) -> np.ndarray:
        return group_count / group_count.sum()


def _split_into_groups(group_idxs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Splits a tensor into multiple groups.

    Args:
        group_idxs: A tensor of length n of group indices where n is the number of data points.

    Returns:
        A tensor of unique indices present in group_idxs.
        A list of Tensors, where the i-th tensor is the indices of the elements of group_idxs that equal the unique
            group index i-th. It has the same length as len(groups).
        A tensor of element counts in each group.
    """
    unique_group_idxs, unique_group_counts = torch.unique(group_idxs, sorted=False, return_counts=True)
    unique_group_element_indices = []
    for group_idx in unique_group_idxs:
        unique_group_element_indices.append(torch.nonzero(group_idxs == group_idx, as_tuple=True)[0])
    return unique_group_idxs, unique_group_element_indices, unique_group_counts
