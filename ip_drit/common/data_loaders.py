"""A module that defines the dataloader."""
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from ip_drit.common.grouper import AbstractGrouper
from ip_drit.datasets import AbstractPublicDataset
from ip_drit.datasets import SubsetPublicDataset


def get_train_loader(
    loader_type: str,
    dataset: Union[AbstractPublicDataset, SubsetPublicDataset],
    batch_size: int,
    uniform_over_groups: Optional[bool] = None,
    grouper: Optional[AbstractGrouper] = None,
    distinct_groups: bool = True,
    n_groups_per_batch: int = None,
    **loader_kwargs,
) -> DataLoader:
    """Constructs and returns the data loader for training.

    Args:
        loader_type: Loader type. This can be 'standard' for standard loaders and 'group' for group loaders, which
            first samples groups and then samples a fixed number of examples belonging to each group.
        dataset: The dataset that we want to load the data from.
        batch_size: Batch size.
        uniform_over_groups (optional): Whether to sample the groups uniformly or according to the natural data
            distribution. Setting to None applies the defaults for each type of loaders. For standard loaders, the
            default is False. For group loaders, the default is True.
        grouper (optional): Grouper used for group loaders or for uniform_over_groups=True
        distinct_groups (optional): Whether to sample distinct_groups within each minibatch for group loaders.
        n_groups_per_batch (optional): Number of groups to sample in each minibatch for group loaders.
        loader_kwargs: kwargs passed into torch DataLoader initialization.

    Output:
        The generated Data loader object.
    """
    if loader_type == "standard":
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
            if grouper is None:
                raise ("Grouper can't be None when uniform_over_groups is True")

            groups, group_counts = grouper.metadata_to_group(dataset.metadata_array, return_counts=True)
            group_weights = 1 / group_counts
            weights = group_weights[groups]

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
            return DataLoader(
                dataset,
                shuffle=False,  # The WeightedRandomSampler already shuffles
                sampler=sampler,
                collate_fn=dataset.collate,
                batch_size=batch_size,
                **loader_kwargs,
            )

    elif loader_type == "group":
        if uniform_over_groups is None:
            uniform_over_groups = True
        assert grouper is not None
        assert n_groups_per_batch is not None
        if n_groups_per_batch > grouper.n_groups:
            raise ValueError(
                f"n_groups_per_batch was set to {n_groups_per_batch} "
                + f"but there are only {grouper.n_groups} groups specified."
            )

        group_ids = grouper.metadata_to_group(dataset.metadata_array)
        batch_sampler = GroupSampler(
            group_ids=group_ids,
            batch_size=batch_size,
            n_groups_per_batch=n_groups_per_batch,
            uniform_over_groups=uniform_over_groups,
            distinct_groups=distinct_groups,
        )

        return DataLoader(
            dataset,
            shuffle=None,
            sampler=None,
            collate_fn=dataset.collate,
            batch_sampler=batch_sampler,
            drop_last=False,
            **loader_kwargs,
        )


def get_eval_loader(
    loader_type: str, dataset: Union[AbstractPublicDataset, SubsetPublicDataset], batch_size: int, **loader_kwargs
) -> DataLoader:
    """Constructs and returns the data loader for evaluation.

    Args:
        loader_type: The dataloader type 'standard' for standard loaders.
        dataset: Data
        batch_size: Batch size
        loader_kwargs: kwargs passed into torch DataLoader initialization.

    Returns:
        A data loader for evaluation
    """
    if loader_type == "standard":
        return DataLoader(
            dataset,
            shuffle=False,  # Do not shuffle eval datasets
            sampler=None,
            collate_fn=dataset.collate,
            batch_size=batch_size,
            **loader_kwargs,
        )


class GroupSampler:
    """Constructs batches by first sampling groups, then sampling data from those groups.

    It drops the last batch if it's incomplete.

    Args:
        group_ids: A tensor that specifies the id of different data points
        batch_size: The size of the loading batch.
        n_groups_per_batch: The number of group per batch.
        uniform_over_groups: Whether to sample the groups uniformly or according to the natural data
            distribution. Setting to None applies the defaults for each type of loaders. For standard loaders, the
            default is False. For group loaders, the default is True.
        distinct_groups (optional): Whether to sample distinct_groups within each minibatch for group loaders.
    """

    def __init__(
        self,
        group_ids: torch.Tensor,
        batch_size: int,
        n_groups_per_batch: int,
        uniform_over_groups: bool = True,
        distinct_groups: bool = True,
    ):

        if batch_size % n_groups_per_batch != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be evenly divisible by n_groups_per_batch ({n_groups_per_batch})."
            )
        if len(group_ids) < batch_size:
            raise ValueError(
                f"The dataset has only {len(group_ids)} examples but the batch size is {batch_size}. "
                + "There must be enough examples to form at least one complete batch."
            )

        self._group_ids = group_ids
        self._unique_groups, self._group_indices, unique_counts = split_into_groups(group_ids)

        self._distinct_groups = distinct_groups
        self._n_groups_per_batch = n_groups_per_batch
        self._n_points_per_group = batch_size // n_groups_per_batch

        self._dataset_size = len(group_ids)
        self._num_batches = self._dataset_size // batch_size

        if uniform_over_groups:  # Sample uniformly over groups
            self._group_prob = None
        else:  # Sample a group proportionately to its size
            self._group_prob = unique_counts.numpy() / unique_counts.numpy().sum()

    def __iter__(self):
        for batch_id in range(self._num_batches):
            # Note that we are selecting group indices rather than groups
            groups_for_batch = np.random.choice(
                len(self._unique_groups),
                size=self._n_groups_per_batch,
                replace=(not self._distinct_groups),
                p=self._group_prob,
            )
            sampled_ids = [
                np.random.choice(
                    self._group_indices[group],
                    size=self._n_points_per_group,
                    replace=len(self._group_indices[group])
                    <= self._n_points_per_group,  # False if the group is larger than the sample size
                    p=None,
                )
                for group in groups_for_batch
            ]

            # Flatten
            sampled_ids = np.concatenate(sampled_ids)
            yield sampled_ids

    def __len__(self):
        return self._num_batches


def split_into_groups(g: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Splits a tensor into multiple groups.

    Args:
        g: A tensor of groups

    Returns:
        A tensor of unique groups present in g/
        A list of Tensors, where the i-th tensor is the indices of the elements of g that equal groups[i].
            Has the same length as len(groups).
       A tensor of element counts in each group.
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts
