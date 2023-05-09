"""A module that defines the dataloader."""
import logging
from enum import auto
from enum import Enum
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from ip_drit.common.grouper import AbstractGrouper
from ip_drit.datasets import SubsetLabeledPublicDataset


class LoaderType(Enum):
    """An enum class that defines the type of the loader."""

    STANDARD = auto()
    GROUP = auto()
    TARGERTED = auto()


def get_train_loader(
    loader_type: LoaderType,
    dataset: SubsetLabeledPublicDataset,
    batch_size: int,
    group_sampler_parameters: Optional[Dict[str, Any]] = None,
    uniform_over_groups: Optional[bool] = None,
    grouper: Optional[AbstractGrouper] = None,
    distinct_groups: bool = True,
    train_n_groups_per_batch: int = None,
    reset_random_generator_after_every_epoch: bool = False,
    seed: Optional[int] = None,
    run_on_cluster: bool = True,
    **loader_kwargs,
) -> DataLoader:
    """Constructs and returns the data loader for training.

    Args:
        loader_type: Loader type. This can be LoaderType.STANDARD for standard loaders and LoaderType.GROUP for group
            loaders. The later first samples groups and then samples a fixed number of examples belonging to each group.
        dataset: The dataset that we want to load the data from. This is normally a subset of the full dataset.
        batch_size: Batch size.
        group_sampler_parameters (optional): The parameters of the group sampler.
        uniform_over_groups (optional): Whether to sample the groups uniformly or according to the natural data
            distribution. Setting to None applies the defaults for each type of loaders. For standard loaders, the
            default is False. For group loaders, the default is True.
        grouper (optional): Grouper used for group loaders or for uniform_over_groups=True
        distinct_groups (optional): Whether to sample distinct_groups within each minibatch for group loaders.
        train_n_groups_per_batch (optional): Number of groups to sample in each minibatch for group loaders.
        reset_random_generator_after_every_epoch (optional): If True, each worker is initialized with a specified random
            seed that is similar for every epoch. Defaults to False.
        seed (optional): input seed from flags. Defautls to None.
        run_on_cluster (optional): if True, the code will be execuated on he cluster.
        loader_kwargs: kwargs passed into torch DataLoader initialization.

    Output:
        The generated Data loader object.
    """
    if loader_type == LoaderType.STANDARD:
        return _generate_standard_data_loader(
            loader_kwargs=loader_kwargs,
            dataset=dataset,
            batch_size=batch_size,
            uniform_over_groups=uniform_over_groups,
            grouper=grouper,
            reset_random_generator_after_every_epoch=reset_random_generator_after_every_epoch,
            run_on_cluster=run_on_cluster,
            seed=seed,
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

        if reset_random_generator_after_every_epoch:
            g = torch.Generator()
            g.manual_seed(seed)
        else:
            g = None

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
            num_workers=_num_of_workers(
                run_on_cluster
            ),  # Setting this different than 1 is important for worker_init_fn to work.
            worker_init_fn=_worker_init_fn if reset_random_generator_after_every_epoch else None,
            persistent_workers=True,
            generator=g,
            **loader_kwargs,
        )
    elif loader_type == LoaderType.TARGERTED:
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

        if reset_random_generator_after_every_epoch:
            g = torch.Generator()
            g.manual_seed(seed)
        else:
            g = None

        return DataLoader(
            dataset,
            shuffle=None,
            sampler=None,
            collate_fn=dataset.collate,
            batch_sampler=GroupSamplerWithFixedSizedGroupChunk(
                chunk_size=group_sampler_parameters["chunk_size"],
                group_idxs=group_indices_all_datapoints,
                batch_size=batch_size,
                n_groups_per_batch=train_n_groups_per_batch,
                uniform_over_groups=uniform_over_groups,
                distinct_groups=distinct_groups,
            ),
            drop_last=False,
            num_workers=_num_of_workers(
                run_on_cluster
            ),  # Setting this different than 1 is important for worker_init_fn to work.
            worker_init_fn=_worker_init_fn if reset_random_generator_after_every_epoch else None,
            persistent_workers=True,
            generator=g,
            **loader_kwargs,
        )
    else:
        raise ValueError(f"Loader type {loader_type.name} is not supported")


def _generate_standard_data_loader(
    loader_kwargs,
    dataset: SubsetLabeledPublicDataset,
    batch_size: int,
    uniform_over_groups: Optional[bool] = None,
    grouper: Optional[AbstractGrouper] = None,
    reset_random_generator_after_every_epoch: Optional[bool] = False,
    run_on_cluster: bool = True,
    seed: int = 0,
) -> DataLoader:
    if reset_random_generator_after_every_epoch:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    if uniform_over_groups is None or not uniform_over_groups:
        return DataLoader(
            dataset,
            shuffle=True,  # Shuffle training dataset
            sampler=None,
            collate_fn=dataset.collate,
            batch_size=batch_size,
            worker_init_fn=_worker_init_fn if reset_random_generator_after_every_epoch else None,
            num_workers=_num_of_workers(
                run_on_cluster
            ),  # Setting this different than 1 is important for worker_init_fn not
            generator=g,
            persistent_workers=True,  # Workers are created after every dataset consumption
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
            num_workers=_num_of_workers(
                run_on_cluster
            ),  # Setting this different than 1 is important for worker_init_fn to work.
            worker_init_fn=_worker_init_fn if reset_random_generator_after_every_epoch else None,
            generator=g,
            persistent_workers=True,  # Workers are created after every dataset consumption
            **loader_kwargs,
        )


def _num_of_workers(run_on_cluster: bool) -> int:
    return 8 if run_on_cluster else 1


def _worker_init_fn(worker_id: int) -> None:
    """A helper function for worker initialization.

    This function makes sure that every workers are initialized with a reperated random seed in the beginning of each
    epoch. See https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/.
    """
    seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.debug(f"WorkerID: {worker_id}, Random seed set as {seed}")


def _validate_grouper_availability(uniform_over_groups: bool, grouper: Optional[AbstractGrouper]) -> None:
    if uniform_over_groups and grouper is None:
        raise ValueError("Grouper can't be None when uniform_over_groups is True")


def get_eval_loader(
    loader_type: LoaderType,
    dataset: SubsetLabeledPublicDataset,
    batch_size: int,
    reset_random_generator_after_every_epoch: bool = False,
    seed: int = 0,
    run_on_cluster: bool = True,
    num_workers: Optional[int] = None,
    **loader_kwargs,
) -> DataLoader:
    """Constructs and returns the data loader for evaluation.

    Args:
        loader_type: The dataloader type 'standard' for standard loaders.
        dataset: A subddataset
        batch_size: Batch size
        loader_kwargs: kwargs passed into torch DataLoader initialization.
        reset_random_generator_after_every_epoch (optional): If True, each worker is initialized with a specified random
            seed that is similar for every epoch. Defaults to False.
        seed (optional): input seed. Defaults to 0.
        run_on_cluster (optional: if True, the code will be executed on the cluster. Defaults to True.

    Returns:
        A data loader for evaluation
    """
    if loader_type == LoaderType.STANDARD:
        if reset_random_generator_after_every_epoch:
            g = torch.Generator()
            g.manual_seed(seed)
        else:
            g = None

        return DataLoader(
            dataset,
            shuffle=False,  # Do not shuffle eval datasets
            sampler=None,
            collate_fn=dataset.collate,
            batch_size=batch_size,
            persistent_workers=True,
            num_workers=_num_of_workers(run_on_cluster)
            if num_workers is None
            else num_workers,  # Setting this different than 1 is important for worker_init_fn to work.
            worker_init_fn=_worker_init_fn if reset_random_generator_after_every_epoch else None,
            generator=g,
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

        logging.info(
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

            yield np.concatenate(sampled_ids)

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
            different chunks may have samples from the same group.
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
        logging.info(f"GroupSamplerWithFixedSizedGroupChunk is used group sampler with chunk size = {chunk_size}!")

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


class InfiniteDataIterator:
    """A data iterator that continuously produces the data.

    Normally used for unlabeled data.
    Adapted from https://github.com/thuml/Transfer-Learning-Library
    Args:
        dataloader: The original data loader.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self._dataloader = dataloader
        self._iter = iter(self._dataloader)

    def __next__(self):
        """Returns the next sample in the batch."""
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self._dataloader)
            data = next(self._iter)
        return data

    def __len__(self) -> int:
        return len(self._dataloader)
