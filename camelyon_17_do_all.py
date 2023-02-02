"""A scripts to run the benchmark for the Camelyon dataset."""
import argparse
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Tuple

import torch.cuda
from absl import app

from ip_drit.algorithms.initializer import initialize_algorithm
from ip_drit.algorithms.single_model_algorithm import ModelAlgorithm
from ip_drit.common.grouper import CombinatorialGrouper
from ip_drit.datasets.camelyon17 import CamelyonDataset
from ip_drit.logger import Logger
from ip_drit.models.wild_model_initializer import WildModel
from ip_drit.patch_transform import TransformationType
from script_utils import calculate_batch_size
from script_utils import configure_split_dict_by_names
from script_utils import parse_bool
from script_utils import use_data_parallel
from train_utils import train

logging.getLogger().setLevel(logging.INFO)


def main():
    """Demo scripts for training, evaluation with Camelyon."""
    logging.info("Running the Camelyon 17 dataset benchmark.")
    parser = _configure_parser()
    FLAGS = parser.parse_args()
    all_dataset_dir, all_log_dir = _dataset_and_log_location(FLAGS.run_on_cluster)

    all_dataset_dir.mkdir(exist_ok=True)
    all_log_dir.mkdir(exist_ok=True)

    camelyon_dataset = CamelyonDataset(dataset_dir=all_dataset_dir / "camelyon17/")

    log_dir = all_log_dir / "erm_camelyon"
    log_dir.mkdir(exist_ok=True)

    config_dict: Dict[str, Any] = {
        "algorithm": ModelAlgorithm.ERM,
        "model": WildModel.DENSENET121,
        "transform": TransformationType.WEAK,
        "target_resolution": None,  # Keep the original dataset resolution
        "scheduler_metric_split": "val",
        "train_group_by_fields": ["hospital"],
        "loss_function": "multitask_bce",
        "algo_log_metric": "accuracy",
        "log_dir": str(log_dir),
        "gradient_accumulation_steps": 1,
        "n_epochs": 20,
        "log_every_n_batches": FLAGS.log_every_n_batches,
        "train_loader": "group",
        "batch_size": calculate_batch_size(FLAGS.run_on_cluster),
        "uniform_over_groups": FLAGS.sample_uniform_over_groups,  #
        "distinct_groups": False,  # If True, enforce groups sampled per batch are distinct.
        "n_groups_per_batch": FLAGS.num_groups_per_training_batch,  # 4
        "scheduler": "linear_schedule_with_warmup",
        "scheduler_kwargs": {"num_warmup_steps": 3},
        "scheduler_metric_name": "scheduler_metric_name",
        "optimizer": "SGD",
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "optimizer_kwargs": {"momentum": 0.9},
        "max_grad_norm": 0.5,
        "use_data_parallel": use_data_parallel(),
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "use_unlabeled_y": False,  # If true, unlabeled loaders will also the true labels for the unlabeled data.
        "verbose": True,
        "report_batch_metric": True,
        "metric": "acc_avg",
        "val_metric_decreasing": False,
        # Saving parameters
        "save_step": 5,
        "seed": 0,
        "save_last": True,
        "save_best": True,
        "save_pred": True,
        "eval_only": False,  # If True, only evaluation will be performed, no training.
    }

    logger = Logger(fpath=str(log_dir / "log.txt"))

    train_grouper = CombinatorialGrouper(dataset=camelyon_dataset, groupby_fields=config_dict["train_group_by_fields"])

    split_dict_by_names = configure_split_dict_by_names(
        full_dataset=camelyon_dataset, grouper=train_grouper, config_dict=config_dict
    )
    algorithm = initialize_algorithm(
        config=config_dict, split_dict_by_name=split_dict_by_names, train_grouper=train_grouper
    )

    if not config_dict["eval_only"]:
        logging.info("Training mode!")
        train(
            algorithm=algorithm,
            split_dict_by_name=split_dict_by_names,
            general_logger=logger,
            config_dict=config_dict,
            epoch_offset=0,
        )
    else:
        logging.info("Evaluation mode!")


def _configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_on_cluster",
        type=parse_bool,
        default=True,
        const=True,
        nargs="?",
        help="A string to specify where to run the the code. "
        + "Defaults to True, in which case the code will be run on the cluster.",
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
    return parser


def _dataset_and_log_location(run_on_cluster: bool) -> Tuple[Path, Path]:
    if run_on_cluster:
        return Path("/jupyter-users-home/tan-2enguyen/datasets"), Path("/jupyter-users-home/tan-2enguyen/all_log_dir")
    else:
        return Path("/Users/tan.nguyen/datasets"), Path("/Users/tan.nguyen/")


if __name__ == "__main__":
    main()
