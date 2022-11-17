"""A module that provides utitilies for scripts."""
from collections import OrderedDict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from ip_drit.io.midog.midog_2021 import dowload_midog_2021_dataset_into_a_folder_return_data_paths
from ip_drit.sampling import Sample


def nearest_even_minibatch_size(org_minibatch_size: int) -> int:
    """Returns a minibatch size that is even and closest to the given minibatch size."""
    return (org_minibatch_size // 2) * 2


def get_number_of_available_gpus() -> int:
    """Returns the number of available GPUs."""
    return torch.cuda.device_count()


def adjust_num_samples_to_be_a_multiple_of_number_of_gpus_and_batch_size_per_gpu_product(
    org_num_samples: int, num_gpu: int, batch_size_per_gpu: int, num_domains: int
) -> int:
    """Returns a modified number of samples that is a multiple of (num_gpu * batch_size_per_gpu * num_domains).

    This ensure that:
    1/ All the domains have the same number of samples in the dataset.
    2/ All mini-batch has the same size on each GPU.
    """
    return (org_num_samples // (num_gpu * batch_size_per_gpu * num_domains)) * (
        num_gpu * batch_size_per_gpu * num_domains
    )


def clean_up_all_memory_used_by_cuda() -> None:
    """Releases all releasable memory used by cuda."""
    torch.cuda.empty_cache()


def make_sure_folder_exists(folders: List[Path]) -> None:
    """Creates folders if not exists."""
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def prepare_saving_folders_for_checkpoint(save_dir: Path) -> Dict[str, Path]:
    """Return a dictionary of folders for saving.

    Args:
        save_dir: A path to saving the checkpoints.
    """
    if save_dir.exists():
        raise ValueError(
            f"The saving folder {save_dir.name} already exists. Consider deleting it or changing "
            + "the experiment name!"
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    return {"every_n_epochs_dir": save_dir / "every_n", "top_n_dir": save_dir / "top_n"}


def loss_string_from_weight_dict(loss_weights_by_name: Dict[str, float]) -> int:
    """Produces a string that describe the loss weights from a dictionary of loss weights by name."""
    return "_".join(f"{k}_{v}" for k, v in loss_weights_by_name.items())


def load_trained_model_from_checkpoint(checkpoint_path: Path, network: nn.Module, starts_str: str) -> nn.Module:
    """Loads the model from the checkpoint on s3.

    Args:
        checkpoint_url: The path to the trained checkpoint.
        network: A network to load the checkpoint parameters to.
        starts_str: A first few letters of the network variable to load the checkpoint from.

    Returns:
        The loaded model.
    """
    state_dict = torch.load(str(checkpoint_path))["state_dict"]
    state_dict = OrderedDict([k[len(starts_str) :], v] for k, v in state_dict.items() if k.startswith(starts_str))
    network.load_state_dict(state_dict)
    return network


def prepare_training_dataset(data_folder: Path) -> Dict[Any, Union[str, List[str], List[int]]]:
    """Downloads public dataset into local folder, samples and returns a dictionary of Sample list by domain index."""
    # MIDOG_FOLDER = Path("/home/huutan86/Documents/MIDOG_2021/images")
    return dowload_midog_2021_dataset_into_a_folder_return_data_paths(target_folder=data_folder)


def adjust_samples_by_domain_idx_dict_to_the_target_number_of_sample(
    samples_by_domain_idx: Dict[int, List[Sample]], target_number_of_samples: int
) -> Dict[int, List[Sample]]:
    """Reduces the number of samples equally in each domain so that the total number of samples reach the target.

    Args:
        samples_by_domain_idx: A dictionary in which each item is a list of samples, keyed by the index of the domain.
        target_number_of_samples: The total target number of samples across all domains.

    Returns:
        The truncated samples by domain idx dictionary.
    """
    total_num_samples = np.sum([len(x) for x in samples_by_domain_idx.values()])
    num_domains = len(samples_by_domain_idx)
    if total_num_samples < target_number_of_samples:
        raise ValueError(
            f"The total number of samples {total_num_samples} is less than the target number of samples "
            + f"required {target_number_of_samples}"
        )
    num_samples_per_domain = int(np.ceil(target_number_of_samples / num_domains))
    return {k: v[:num_samples_per_domain] for k, v in samples_by_domain_idx.items()}
