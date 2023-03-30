"""A utility module for training."""
import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch.cuda
from torch.utils.data import DataLoader

from ip_drit.algorithms.single_model_algorithm import ModelAlgorithm
from ip_drit.algorithms.single_model_algorithm import SingleModelAlgorithm
from ip_drit.common.data_loaders import get_eval_loader
from ip_drit.common.data_loaders import get_train_loader
from ip_drit.common.data_loaders import LoaderType
from ip_drit.common.grouper import CombinatorialGrouper
from ip_drit.datasets import AbstractPublicDataset
from ip_drit.datasets import SubsetLabeledPublicDataset
from ip_drit.logger import BatchLogger
from ip_drit.logger import Logger
from ip_drit.patch_transform import initialize_transform
from ip_drit.patch_transform import TransformationType


def parse_bool(v: str) -> bool:
    """Converts a string to true boolean value."""
    if v.lower() == "false":
        return False
    elif v.lower() == "true":
        return True
    else:
        raise ValueError("Unknown string to convert to boolean!")


def configure_split_dict_by_names(
    full_dataset: AbstractPublicDataset, grouper: CombinatorialGrouper, config_dict: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Configures the split dict for different splits (`train`, `id_val`, `val`, `test`).

    Args:
        full_dataset: A full dataset with different split names defined.
        grouper: A grouper that is used define how the sample images should be grouped together.
        config_dict: A configuration dictionary.
        transform: If True, the type of transformation added.

    Returns:
        A dictionary of dictionaries, keyed by the name of the split.
    """
    split_dict = defaultdict(dict)
    for split_name in full_dataset.split_dict:
        logging.info(f"Generating split dict for split {split_name}")
        subdataset = full_dataset.get_subset(
            split=split_name,
            frac=1.0,
            transform=initialize_transform(
                transform_name=config_dict["transform"], config_dict=config_dict, full_dataset=full_dataset
            ),
        )

        split_dict[split_name]["dataset"] = subdataset
        logging.info(f"Dataset size = {len(subdataset)}")

        split_dict[split_name]["loader"] = _get_data_loader_by_split_name(
            sub_dataset=subdataset, grouper=grouper, split_name=split_name, config_dict=config_dict
        )

        split_dict[split_name]["eval_logger"] = BatchLogger(
            os.path.join(config_dict["log_dir"], f"{split_name}_eval.csv"), mode="w", use_wandb=False
        )

        split_dict[split_name]["algo_logger"] = BatchLogger(
            os.path.join(config_dict["log_dir"], f"{split_name}_algo.csv"), mode="w", use_wandb=False
        )

        split_dict[split_name]["verbose"] = config_dict["verbose"]
        split_dict[split_name]["report_batch_metric"] = config_dict["report_batch_metric"]
        split_dict[split_name]["split"] = split_name

    return split_dict


def _get_data_loader_by_split_name(
    sub_dataset: SubsetLabeledPublicDataset, grouper: CombinatorialGrouper, split_name: str, config_dict: Dict[str, Any]
) -> DataLoader:
    if split_name in ["train", "train_unlabeled"]:
        return get_train_loader(
            loader_type=config_dict["train_loader"],
            dataset=sub_dataset,
            batch_size=config_dict["batch_size"],
            uniform_over_groups=config_dict["uniform_over_groups"],
            grouper=grouper,
            distinct_groups=config_dict["distinct_groups"],
            train_n_groups_per_batch=config_dict["n_groups_per_batch"],
            reset_random_generator_after_every_epoch=config_dict["reset_random_generator_after_every_epoch"],
            seed=config_dict["seed"],
            run_on_cluster=config_dict["run_on_cluster"],
        )

    elif split_name in ("id_val", "test", "val", "val_unlabeled", "test_unlabeled"):
        return get_eval_loader(
            loader_type=LoaderType.STANDARD,
            dataset=sub_dataset,
            batch_size=config_dict["batch_size"],
            reset_random_generator_after_every_epoch=config_dict["reset_random_generator_after_every_epoch"],
            seed=config_dict["seed"],
            run_on_cluster=config_dict["run_on_cluster"],
        )
    else:
        raise ValueError(f"Unknown split name {split_name}")


def use_data_parallel() -> bool:
    """Returns True if more than 1 training device is available, otherwise, False."""
    return num_of_available_devices() > 1


def num_of_available_devices() -> int:
    """Gets the number of available training devices."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


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
        algorithm: the algorithm that was run on.
        split_dict: The dictionary for the current split.
        general_logger: The general logger that is used to write the log output.
        epoch: The current epoch index.
        effective_batch_idx: The effective batch index.
    """
    if algorithm.has_log:
        log = algorithm.get_log()
        log["epoch"] = epoch
        log["batch"] = effective_batch_idx
        split_dict["algo_logger"].log(log)
        if split_dict["verbose"] and split_dict["report_batch_metric"]:
            general_logger.write(algorithm.get_pretty_log_str())
        algorithm.reset_log()


def configure_parser() -> argparse.ArgumentParser:
    """Configures the parser's parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_on_cluster",
        type=parse_bool,
        default=True,
        const=True,
        nargs="?",
        help="A string to specify where to run the code. "
        + "Defaults to True, in which case the code will be run on the cluster.",
    )
    parser.add_argument(
        "--verbose",
        type=parse_bool,
        default=False,
        const=True,
        nargs="?",
        help="A string to specify if we want to display several images or not. Defaults to False.",
    )

    parser.add_argument(
        "--sample_uniform_over_groups",
        type=parse_bool,
        default=True,
        const=True,
        nargs="?",
        help="A string to specify how the sampling should be done. If True, sample examples such that batches are "
        + "uniform over groups. If False, sample using the group probabilities that is proportional to group size.",
    )
    parser.add_argument(
        "--num_groups_per_training_batch", type=int, default=3, help="The number of groups per training batch."
    )

    parser.add_argument(
        "--log_every_n_batches", type=int, default=3, help="The number of batches to log once. Defaults to 3."
    )

    parser.add_argument(
        "--use_full_dataset",
        type=parse_bool,
        default=True,
        help="If True, full dataset will be used. Defaults to False.",
    )

    parser.add_argument(
        "--log_dir_cluster",
        type=str,
        default="/jupyter-users-home/tan-2enguyen/all_log_dir",
        help="Directory for logging in cluster",
    )

    parser.add_argument(
        "--log_dir_local", type=str, default="/Users/tan.nguyen/", help="Directory for logging in local"
    )

    parser.add_argument(
        "--dataset_dir_cluster",
        type=str,
        default="/jupyter-users-home/tan-2enguyen/datasets",
        help="Directory for datasets in cluster",
    )

    parser.add_argument(
        "--dataset_dir_local", type=str, default="/Users/tan.nguyen/datasets", help="Directory for datasets in local"
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed, use values from 0 to 9.")

    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer")

    parser.add_argument("--run_eval_after_train", type=parse_bool, default=False, help="Optional: run eval after train")

    parser.add_argument(
        "--eval_only",
        type=parse_bool,
        default=False,
        help="If true, only evaluation is done. Defaults to False, in which case, training will be performed.",
    )

    parser.add_argument(
        "--eval_epoch",
        type=int,
        default=None,
        help="If specified, this epoch is used for evaluation, else the epoch with the best metric is used.",
    )

    parser.add_argument(
        "--model_prefix",
        type=str,
        default="/jupyter-users-home/dinkar-2ejuyal/all_log_dir/erm_camelyon",
        help="The prefix to the model path for evaluation mode. "
        "It will be appended by either best_model or a specific epoch number to generate evaluation model path.",
    )

    parser.add_argument("--n_epochs", type=int, default=30, help="Number of epochs to train for")
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="The path to a pretrained model.")

    parser.add_argument(
        "--reset_random_generator_after_every_epoch",
        type=parse_bool,
        default=False,
        help="If True, the random number generator will be restarted using the same seed after every epoches.",
    )

    parser.add_argument("--soft_pseudolabels", type=parse_bool, default=False, help="If True, use soft pseudo labels")
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="*",
        default=None,
        help="The list of GPUs to use. Defaults to " + "None, in which case, all GPUs will be used.",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=None,
        help="The default batch size. Defaults to None, in" + "which case, it will be automatically calculated.",
    )
    parser.add_argument(
        "--drop_centers", nargs="+", default=[], help="Drop centers from train set, has to be a subset of [0,3,4]"
    )
    parser.add_argument(
        "--contrimix_num_attr_vectors",
        type=int,
        default=4,
        help="number of attribute vectors to use for contrimix",
    )
    return parser


def dataset_and_log_location(
    run_on_cluster: bool, log_dir_cluster: str, log_dir_local: str, dataset_dir_cluster: str, dataset_dir_local: str
) -> Tuple[Path, Path]:
    """Returns the location for the dataset and the log directory."""
    if run_on_cluster:
        return Path(dataset_dir_cluster), Path(log_dir_cluster)
    else:
        return Path(dataset_dir_local), Path(log_dir_local)


def generate_eval_model_path(eval_epoch: int, model_prefix: str, seed: int) -> str:
    """Returns the path to the evaluation model."""
    if eval_epoch is None:
        eval_model_path: str = os.path.join(model_prefix, f"camelyon17_seed:{seed}_epoch:best_model.pth")
    else:
        eval_model_path: str = os.path.join(model_prefix, f"camelyon17_seed:{seed}_epoch:{eval_epoch}_model.pth")
    return eval_model_path


def calculate_batch_size(
    algorithm: ModelAlgorithm, run_on_cluster: bool, batch_size_per_gpu: Optional[int] = None
) -> int:
    """Calculates the batch size for a given 'algorithm' and wether the code 'run_on_cluster' or not."""
    num_devices = num_of_available_devices()
    logging.info(f"Number of training devices = {num_devices}.")
    PER_GPU_BATCH_SIZE_BY_ALGORITHM_ON_CLUSTER: Dict[ModelAlgorithm, int] = {
        ModelAlgorithm.CONTRIMIX: 210,
        ModelAlgorithm.ERM: 1500,
        ModelAlgorithm.NOISY_STUDENT: 900,
    }

    PER_GPU_BATCH_SIZE_BY_ALGORITHM_LOCAL: Dict[ModelAlgorithm, int] = {
        ModelAlgorithm.CONTRIMIX: 153,
        ModelAlgorithm.ERM: 90,
        ModelAlgorithm.NOISY_STUDENT: 45,
    }

    if batch_size_per_gpu is None:
        if run_on_cluster:
            batch_size_per_gpu = PER_GPU_BATCH_SIZE_BY_ALGORITHM_ON_CLUSTER[algorithm]
        else:
            batch_size_per_gpu = PER_GPU_BATCH_SIZE_BY_ALGORITHM_LOCAL[algorithm]
    else:
        batch_size_per_gpu = batch_size_per_gpu

    batch_size = batch_size_per_gpu * num_devices
    logging.info(f"Using a batch size of {batch_size} for {batch_size_per_gpu}/device * {num_devices} device(s).")
    return batch_size


def set_visible_gpus(gpu_ids: Optional[List[int]] = None) -> None:
    """Sets specific GPUs to be available.

    Args:
        gpu_ids (optional): A list of GPU ids to set. Defaults to None, in which case, all GPUs should be available.
    """
    if gpu_ids is not None:
        gpu_ids = ",".join(str(i) for i in gpu_ids)
    else:
        gpu_ids = ",".join(str(i) for i in range(torch.cuda.device_count()))
    logging.info(f"Setting GPU ids {gpu_ids} to be visible!")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
