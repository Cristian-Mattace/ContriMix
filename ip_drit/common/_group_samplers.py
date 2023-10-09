"""A modules that implements the group samplers."""
from typing import Generator
from typing import List
from typing import Tuple

import numpy as np
import torch


def _split_into_groups(group_idxs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Splits a tensor of group indices into multiple groups that have the same index within each group.

    Args:
        group_idxs: A tensor of length-n of group indices where n is the number of data points.

    Returns:
        A tensor of unique group indices for all the indices found in group_idxs.
        A list of Tensors, where the i-th tensor is the element indices with index that is equal the unique
            group index i-th. It has the same length as len(groups).
        A tensor of element counts in each group.
    """
    unique_group_idxs, unique_group_counts = torch.unique(group_idxs, sorted=False, return_counts=True)
    unique_group_element_indices = []
    for group_idx in unique_group_idxs:
        unique_group_element_indices.append(torch.nonzero(group_idxs == group_idx, as_tuple=True)[0])
    return unique_group_idxs, unique_group_element_indices, unique_group_counts


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

        self._unique_group_indices, self._unique_group_element_indices, unique_group_counts = _split_into_groups(
            group_idxs
        )

        self._distinct_groups: bool = distinct_groups
        self._n_groups_per_batch: int = n_groups_per_batch
        self._n_points_per_group: int = batch_size // n_groups_per_batch

        if n_groups_per_batch > len(group_idxs):
            raise ValueError(
                f"Can't generate a group sampler with n_groups_per_batch ({n_groups_per_batch}) larger "
                + f" than the number of available group {len(group_idxs)}"
            )

        self._num_batches = len(group_idxs) // batch_size

        print(
            f"{self.__class__} initialized with uniform_over_group = {uniform_over_groups}, "
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
                len(self._unique_group_indices),
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

            yield np.random.permutation(np.concatenate(sampled_ids))

    def __len__(self):
        return self._num_batches

    @staticmethod
    def _group_propability_from_group_counts(group_count: np.ndarray) -> np.ndarray:
        return group_count / group_count.sum()


class GroupSamplerWithFixedSizedGroupChunk(GroupSampler):
    """A group samplers that generates multiple chunks of samples that have the same group.

    It will returns the batch as
    |I11 I12 ... I1k| I21 I22 .... I2k| ....
    <-- chunk 0 ----><--- chunk 1 ----><-- ...
    It drops the last batch if it's incomplete.

    Args:
        chunk_size: The size of the chunk (k). Every chunk_size samples from the left will have the same group. Two
            different chunks may have samples of the same group.
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
        chunk_size: int,
        group_idxs: torch.Tensor,
        batch_size: int,
        n_groups_per_batch: int,
        uniform_over_groups: bool = True,
        distinct_groups: bool = True,
    ):
        if chunk_size < 2:
            raise ValueError("The size of the chunk can't be less than 2!")
        self._chunk_size = chunk_size
        if not batch_size % chunk_size == 0:
            raise ValueError(f"Batch_size ({batch_size}) is not a multiple of the chunk size {chunk_size}.")

        self._num_chunks_per_batch = batch_size // chunk_size

        super().__init__(
            group_idxs=group_idxs,
            batch_size=batch_size,
            n_groups_per_batch=n_groups_per_batch,
            uniform_over_groups=uniform_over_groups,
            distinct_groups=distinct_groups,
        )

        self._validate_chunk_size()
        print(f"GroupSamplerWithFixedSizedGroupChunk is used group sampler with chunk size = {chunk_size}!")

    def _validate_chunk_size(self) -> None:
        min_group_size = min(len(x) for x in self._unique_group_element_indices)
        if min_group_size < self._chunk_size:
            raise ValueError(
                f"Minimum group size = {min_group_size} cannot be smaller than the chunk size of "
                + f"{self._chunk_size}"
            )

    def __iter__(self) -> Generator:
        for _ in range(self._num_batches):
            # Note that we are selecting group indices rather than groups
            unique_group_idxes_for_batch = np.random.choice(
                len(self._unique_group_indices),
                size=self._num_chunks_per_batch,
                replace=(not self._distinct_groups),
                p=self._group_prob,
            )

            sampled_ids = [
                np.random.choice(
                    self._unique_group_element_indices[group_idx],
                    size=self._chunk_size,
                    # If the size of the group is not larger than the number points per group reguired, do replacement.
                    replace=len(self._unique_group_element_indices[group_idx]) <= self._chunk_size,
                    p=None,
                )
                for group_idx in unique_group_idxes_for_batch
            ]

            yield np.concatenate(sampled_ids)
