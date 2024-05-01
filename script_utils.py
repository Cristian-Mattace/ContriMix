"""A utility module for training."""
import argparse
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch.cuda
from torch.utils.data import DataLoader
from wilds.common.grouper import CombinatorialGrouper

from ip_drit.algorithms.single_model_algorithm import ModelAlgorithm
from ip_drit.algorithms.single_model_algorithm import SingleModelAlgorithm
from ip_drit.common.data_loaders import get_eval_loader
from ip_drit.common.data_loaders import get_train_loader
from ip_drit.common.data_loaders import LoaderType
from ip_drit.datasets import AbstractPublicDataset
from ip_drit.datasets import SubsetLabeledPublicDataset
from ip_drit.logger import BatchLogger
from ip_drit.patch_transform import initialize_transform


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
        print(f"Generating split dict for split {split_name}")
        subdataset = full_dataset.get_subset(
            split=split_name,
            frac=1.0,
            transform=initialize_transform(
                transform_name=config_dict["transform"],
                config_dict=config_dict,
                full_dataset=full_dataset,
                is_training=(split_name == "train"),
            ),
        )

        split_dict[split_name]["dataset"] = subdataset
        print(f"Dataset size = {len(subdataset)}")

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
        split_dict[split_name]["name"] = full_dataset.split_names[split_name]

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
            group_sampler_parameters=config_dict.get("group_sampler_parameters", None),
            distinct_groups=config_dict["distinct_groups"],
            train_n_groups_per_batch=config_dict["n_groups_per_batch"],
            reset_random_generator_after_every_epoch=config_dict["reset_random_generator_after_every_epoch"],
            seed=config_dict["seed"],
            run_on_cluster=config_dict["run_on_cluster"],
            loader_kwargs=config_dict.get("loader_kwargs", {}),
            ddp_params=config_dict.get("ddp_params", {}),
            use_ddp_over_dp=config_dict["use_ddp_over_dp"],
        )

    elif split_name in ("id_val", "test", "val", "val_unlabeled", "test_unlabeled", "id_test"):
        return get_eval_loader(
            loader_type=LoaderType.STANDARD,
            dataset=sub_dataset,
            batch_size=config_dict["batch_size"],
            reset_random_generator_after_every_epoch=config_dict["reset_random_generator_after_every_epoch"],
            seed=config_dict["seed"],
            run_on_cluster=config_dict["run_on_cluster"],
            loader_kwargs=config_dict.get("loader_kwargs", {}),
            ddp_params=config_dict.get("ddp_params", {}),
            use_ddp_over_dp=config_dict["use_ddp_over_dp"],
        )
    else:
        raise ValueError(f"Unknown split name {split_name}")


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
    algorithm: SingleModelAlgorithm, split_dict: Dict[str, Any], epoch: int, effective_batch_idx: int
) -> None:
    """Logs the results of the algorithm.

    Args:
        algorithm: the algorithm that was run on.
        split_dict: The dictionary for the current split.
        epoch: The current epoch index.
        effective_batch_idx: The effective batch index.
    """
    if algorithm.has_log:
        log = algorithm.get_log()
        log["epoch"] = epoch
        log["batch"] = effective_batch_idx
        split_dict["algo_logger"].log(log)
        if split_dict["verbose"] and split_dict["report_batch_metric"]:
            print(algorithm.get_pretty_log_str())
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
        default=False,
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
        "--epoch_offset",
        type=int,
        default=0,
        help="If specified, this epoch is used to restart the train.",
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
        "--val_center",
        type=int,
        default=1,
        help="The default valuation center is center number 1.",
    )

    parser.add_argument(
        "--test_center",
        type=int,
        default=2,
        help="The default test center is center number 1.",
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
        "--contrimix_num_attr_vectors", type=int, default=8, help="number of attribute vectors to use for contrimix"
    )
    parser.add_argument("--noise_std", type=float, default=0.0, help="gaussian noise std")
    parser.add_argument(
        "--num_mixing_per_image", type=int, default=2, help="number of mixing per image for contrimix. Defaults to 2"
    )
    parser.add_argument(
        "--use_cut_mix",
        type=parse_bool,
        default=True,
        help="If true, only evaluation is done. Defaults to True, in which case, cutmix will be used.",
    )
    parser.add_argument(
        "--distinct_groups",
        type=parse_bool,
        default=True,
        help="If true, distinct groups will be used. Defaults to True.",
    )
    parser.add_argument(
        "--normalize_signals_into_to_backbone",
        type=parse_bool,
        default=True,
        help="If true, the inputs will be normalized to the backbone. Defaults to True.",
    )
    parser.add_argument(
        "--cut_mix_alpha", type=float, default=1.0, help="If true, the alpha parameters for CutMix. Defaults to 1.0."
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
    dataset_name: str,
    algorithm: ModelAlgorithm,
    run_on_cluster: bool,
    batch_size_per_gpu: Optional[int] = None,
    ddp_mode: bool = False,
) -> int:
    """Calculates the batch size for a given 'algorithm' and wether the code 'run_on_cluster' or not.

    Args:
        dataset_name: name of the dataset
        algorithm: name of the algorithm
        run_on_cluster: If True, the code will be run on the cluster. Otherwise, it will be run locally on a Mac.
        batch_size_per_gpu (optional): The specified number of samples per GPUs. If specified, returns that values.
            Otherwise, returns the default values.
        ddp_mode: If True, the ddp mode is used. In this case, the batch size is for 1 GPU.
    """
    if ddp_mode:
        num_devices = 1
        DEFAULT_BATCHSIZE_DICT_BY_DATASET_NAME_ON_CLUSTER: Dict[str, Dict[ModelAlgorithm, int]] = {
            "camelyon17": {ModelAlgorithm.CONTRIMIX: 360}
        }

    else:
        num_devices = num_of_available_devices()
        DEFAULT_BATCHSIZE_DICT_BY_DATASET_NAME_ON_CLUSTER: Dict[str, Dict[ModelAlgorithm, int]] = {
            "camelyon17": {ModelAlgorithm.CONTRIMIX: 210, ModelAlgorithm.ERM: 1500},
            "tcga_unlabeled": {ModelAlgorithm.CONTRIMIX: 10},
        }
        DEFAULT_BATCHSIZE_DICT_BY_DATASET_NAME_ON_LOCAL: Dict[str, Dict[ModelAlgorithm, int]] = {
            "camelyon17": {ModelAlgorithm.CONTRIMIX: 72, ModelAlgorithm.ERM: 90}
        }

    print(f"Number of training devices = {num_devices}.")

    if batch_size_per_gpu is None:
        if run_on_cluster:
            batch_size_per_gpu = DEFAULT_BATCHSIZE_DICT_BY_DATASET_NAME_ON_CLUSTER[dataset_name][algorithm]
        else:
            batch_size_per_gpu = DEFAULT_BATCHSIZE_DICT_BY_DATASET_NAME_ON_LOCAL[dataset_name][algorithm]
    else:
        batch_size_per_gpu = batch_size_per_gpu

    batch_size = batch_size_per_gpu * num_devices
    print(f"Using a batch size of {batch_size} for {batch_size_per_gpu} samples per device.")
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
    print(f"Setting GPU ids {gpu_ids} to be visible!")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


def is_master_process(config_dict: Dict[str, Any]) -> bool:
    """Returns True if this is a master process."""
    if config_dict["use_ddp_over_dp"]:
        return config_dict["ddp_params"]["local_rank"] == 0
    return True


def set_seed(seed: int):
    """Sets seed."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_group_data(datasets, grouper, logger):
    """Logs the data from different groups."""
    for k, dataset in datasets.items():
        name = dataset["name"]
        dataset = dataset["dataset"]
        logger.write(f"{name} data...\n")
        if grouper is None:
            logger.write(f"    n = {len(dataset)}\n")
        else:
            _, group_counts = grouper.metadata_to_group(dataset.metadata_array, return_counts=True)
            group_counts = group_counts.tolist()
            for group_idx in range(grouper.n_groups):
                logger.write(f"    {grouper.group_str(group_idx)}: n = {group_counts[group_idx]:.0f}\n")
    logger.flush()
