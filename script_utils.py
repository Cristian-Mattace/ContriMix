"""A utility module for training."""
import argparse
import logging
import os
from collections import defaultdict
from typing import Any
from typing import Dict

import torch.cuda
from torch.utils.data import DataLoader

from ip_drit.algorithms.single_model_algorithm import SingleModelAlgorithm
from ip_drit.common.data_loaders import get_eval_loader
from ip_drit.common.data_loaders import get_train_loader
from ip_drit.common.grouper import CombinatorialGrouper
from ip_drit.datasets import AbstractPublicDataset
from ip_drit.datasets import SubsetPublicDataset
from ip_drit.logger import BatchLogger
from ip_drit.logger import Logger
from ip_drit.patch_transform import initialize_transform


def parse_bool(v: str) -> bool:
    """Converts a string to true boolean value."""
    return bool(v.lower())


def configure_split_dict_by_names(
    full_dataset: AbstractPublicDataset, grouper: CombinatorialGrouper, config_dict: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Configures the split dict for different splits (train, id_val, test, ood_val etc).

    Args:
        full_dataset: A full dataset with different split names defined.
        grouper: A grouper that is used define how the sample images should be grouped together.
        config_dict: A configuration dictionary.

    Returns:
        A dictionary of dictionaries, keyed by the name of the split.
    """
    split_dict = defaultdict(dict)
    for split_name in full_dataset.split_dict:
        logging.info(f"Generating split dict for split {split_name}")
        split_dict[split_name]["dataset"] = full_dataset.get_subset(
            split=split_name,
            frac=1.0,
            transform=initialize_transform(
                transform_name=config_dict["transform"], config_dict=config_dict, full_dataset=full_dataset
            ),
        )

        split_dict[split_name]["loader"] = _get_data_loader_by_split_name(
            sub_dataset=split_dict[split_name]["dataset"],
            grouper=grouper,
            split_name=split_name,
            config_dict=config_dict,
        )

        split_dict[split_name]["eval_logger"] = BatchLogger(
            os.path.join(config_dict["log_dir"], f"{split_name}_eval.csv"), mode="w", use_wandb=False
        )

        split_dict[split_name]["algo_logger"] = BatchLogger(
            os.path.join(config_dict["log_dir"], f"{split_name}_train.csv"), mode="w", use_wandb=False
        )

        split_dict[split_name]["verbose"] = config_dict["verbose"]
        split_dict[split_name]["split"] = split_name

    return split_dict


def _get_data_loader_by_split_name(
    sub_dataset: SubsetPublicDataset, grouper: CombinatorialGrouper, split_name: str, config_dict: Dict[str, Any]
) -> DataLoader:
    if split_name == "train":
        return get_train_loader(
            loader_type=config_dict["train_loader"],
            dataset=sub_dataset,
            batch_size=config_dict["batch_size"],
            uniform_over_groups=config_dict["uniform_over_groups"],
            grouper=grouper,
            distinct_groups=config_dict["distinct_groups"],
            n_groups_per_batch=config_dict["n_groups_per_batch"],
        )
    elif split_name == "ood_val":
        return get_eval_loader(loader_type="standard", dataset=sub_dataset, batch_size=config_dict["batch_size"])


def use_data_parallel() -> bool:
    """Returns True of GPU training is available, otherwise, False."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logging.info(f"Only {device_count} GPU(s) detected")
        return device_count > 1
    return False


def detach_and_clone(obj: torch.Tensor) -> torch.Tensor:
    """Detaches and clone a tensor."""
    if torch.is_tensor(obj):
        return obj.detach().clone()
    else:
        raise TypeError("Invalid type for detach_and_clone")


def log_results(
    algorithm: SingleModelAlgorithm,
    split_dict: Dict[str, Any],
    general_logger: Logger,
    epoch: int,
    effective_batch_idx: int,
) -> None:
    """Logs the results of the algorithm.

    Args:
        algorithm: the
    """
    if algorithm.has_log:
        log = algorithm.get_log()
        log["epoch"] = epoch
        log["batch"] = effective_batch_idx
        split_dict["algo_logger"].log(log)
        if split_dict["verbose"]:
            general_logger.write(algorithm.get_pretty_log_str())
        algorithm.reset_log()
