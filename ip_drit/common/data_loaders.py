"""A module that defines the dataloader."""
import logging
from enum import auto
from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import torch
from catalyst.data.sampler import DistributedSamplerWrapper
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler

from ip_drit.common._group_samplers import GroupSampler
from ip_drit.common._group_samplers import GroupSamplerWithFixedSizedGroupChunk
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
    ddp_params: Dict[str, int],
    group_sampler_parameters: Optional[Dict[str, Any]] = None,
    uniform_over_groups: Optional[bool] = None,
    grouper: Optional[AbstractGrouper] = None,
    distinct_groups: bool = True,
    train_n_groups_per_batch: int = None,
    reset_random_generator_after_every_epoch: bool = False,
    seed: Optional[int] = None,
    run_on_cluster: bool = True,
    loader_kwargs: Optional[Dict[str, Any]] = None,
    use_ddp_over_dp: bool = False,
) -> DataLoader:
    """Constructs and returns the data loader for training.

    Args:
        loader_type: Loader type. This can be LoaderType.STANDARD for standard loaders and LoaderType.GROUP for group
            loaders. The later first samples groups and then samples a fixed number of examples belonging to each group.
        dataset: The dataset that we want to load the data from. This is normally a subset of the full dataset.
        batch_size: Batch size.
        ddp_params: A dictionary of DDP parameters.
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
        use_ddp_over_dp (optional): If True, the DDP will be used. Defaults to False.
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
            use_ddp_over_dp=use_ddp_over_dp,
            ddp_params=ddp_params,
        )
    elif loader_type == LoaderType.GROUP:
        if uniform_over_groups is None:
            uniform_over_groups = True

        _validate_grouper_availability(uniform_over_groups=uniform_over_groups, grouper=grouper)

        group_indices_all_datapoints = grouper.metadata_to_group(dataset.metadata_array)
        num_groups_available = len(np.unique(group_indices_all_datapoints.numpy()))

        print(f"Train data loader has {num_groups_available} groups.")
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

        sampler = GroupSampler(
            group_idxs=group_indices_all_datapoints,
            batch_size=batch_size,
            n_groups_per_batch=train_n_groups_per_batch,
            uniform_over_groups=uniform_over_groups,
            distinct_groups=distinct_groups,
        )
        if use_ddp_over_dp:
            sampler = _wrap_sampler_for_dpp(sampler=sampler, ddp_params=ddp_params)

        return DataLoader(
            dataset,
            shuffle=None,
            sampler=None,
            collate_fn=dataset.collate,
            batch_sampler=sampler,
            drop_last=False,
            num_workers=_num_of_workers(
                run_on_cluster
            ),  # Setting this different than 1 is important for worker_init_fn to work.
            worker_init_fn=_worker_init_fn if reset_random_generator_after_every_epoch else None,
            persistent_workers=True,
            generator=g,
        )
    elif loader_type == LoaderType.TARGERTED:
        if uniform_over_groups is None:
            uniform_over_groups = True

        _validate_grouper_availability(uniform_over_groups=uniform_over_groups, grouper=grouper)

        group_indices_all_datapoints = grouper.metadata_to_group_indices(dataset.metadata_array)
        num_groups_available = len(np.unique(group_indices_all_datapoints.numpy()))

        print(f"Train data loader has {num_groups_available} groups.")
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

        sampler = GroupSamplerWithFixedSizedGroupChunk(
            chunk_size=group_sampler_parameters["chunk_size"],
            group_idxs=group_indices_all_datapoints,
            batch_size=batch_size,
            n_groups_per_batch=train_n_groups_per_batch,
            uniform_over_groups=uniform_over_groups,
            distinct_groups=distinct_groups,
        )
        if use_ddp_over_dp:
            sampler = _wrap_sampler_for_dpp(sampler=sampler, ddp_params=ddp_params)

        return DataLoader(
            dataset,
            shuffle=None,
            sampler=None,
            collate_fn=dataset.collate,
            batch_sampler=sampler,
            drop_last=False,
            num_workers=_num_of_workers(
                run_on_cluster
            ),  # Setting this different than 1 is important for worker_init_fn to work.
            worker_init_fn=_worker_init_fn if reset_random_generator_after_every_epoch else None,
            persistent_workers=True,
            generator=g,
        )
    else:
        raise ValueError(f"Loader type {loader_type.name} is not supported")


def _generate_standard_data_loader(
    loader_kwargs,
    dataset: SubsetLabeledPublicDataset,
    batch_size: int,
    use_ddp_over_dp: bool,
    ddp_params: Dict[str, int],
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
        sampler = None
        if use_ddp_over_dp:
            sampler = DistributedSampler(
                dataset=dataset,
                shuffle=False,
                num_replicas=ddp_params.get("num_replicas"),
                rank=ddp_params.get("local_rank"),
                drop_last=True,
                seed=0,
            )

        return DataLoader(
            dataset,
            shuffle=True,  # Shuffle training dataset
            sampler=sampler,
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
        groups, group_counts = grouper.metadata_to_group(dataset.metadata_array, return_counts=True)
        group_weights = 1 / group_counts
        weights = group_weights[groups]
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
        if use_ddp_over_dp:
            sampler = _wrap_sampler_for_dpp(sampler, ddp_params=ddp_params)

        return DataLoader(
            dataset,
            shuffle=False,  # The WeightedRandomSampler already shuffles
            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler=sampler,
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
    ddp_params: Dict[str, int],
    reset_random_generator_after_every_epoch: bool = False,
    seed: int = 0,
    run_on_cluster: bool = True,
    num_workers: Optional[int] = None,
    use_ddp_over_dp: bool = False,
    **loader_kwargs,
) -> DataLoader:
    """Constructs and returns the data loader for evaluation.

    Note that in DDP, only master do the evaluation. Other processes are blocked.

    Args:
        loader_type: The dataloader type 'standard' for standard loaders.
        dataset: A subddataset
        batch_size: Batch size
        ddp_params: A dictionary of DDP parameters.
        loader_kwargs: kwargs passed into torch DataLoader initialization.
        reset_random_generator_after_every_epoch (optional): If True, each worker is initialized with a specified random
            seed that is similar for every epoch. Defaults to False.
        seed (optional): input seed. Defaults to 0.
        run_on_cluster (optional): if True, the code will be executed on the cluster. Defaults to True.
        num_workers (optional): The number of data loading workers.
        use_ddp_over_dp (optional): If True, the DDP will be used. Defaults to False.

    Returns:
        A data loader for evaluation
    """
    if loader_type == LoaderType.STANDARD:
        if reset_random_generator_after_every_epoch:
            g = torch.Generator()
            g.manual_seed(seed)
        else:
            g = None

        sampler = None
        if use_ddp_over_dp:
            sampler = DistributedSampler(
                dataset=dataset,
                shuffle=False,
                num_replicas=ddp_params.get("num_replicas"),
                rank=ddp_params.get("local_rank"),
                drop_last=True,
                seed=0,
            )

        return DataLoader(
            dataset,
            shuffle=False,  # Do not shuffle eval datasets
            sampler=sampler,
            collate_fn=dataset.collate,
            batch_size=batch_size,
            persistent_workers=True,
            num_workers=_num_of_workers(run_on_cluster)
            if num_workers is None
            else num_workers,  # Setting this different than 1 is important for worker_init_fn to work.
            worker_init_fn=_worker_init_fn if reset_random_generator_after_every_epoch else None,
            generator=g,
        )
    else:
        raise ValueError(f"The type of evaluation loader {loader_type.name} is not supported!")


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


def _wrap_sampler_for_dpp(sampler, ddp_params: Dict[str, int]) -> DistributedSamplerWrapper:
    local_rank = ddp_params.get("local_rank")
    num_replicas = ddp_params.get("num_replicas")
    return DistributedSamplerWrapper(sampler, num_replicas=num_replicas, rank=local_rank)
