"""A scripts to run the benchmark for the Camelyon dataset."""
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import torch.cuda
from absl import app
from train import train
from utils import configure_split_dict_by_names
from utils import use_data_parallel

from ip_drit.algorithms.initializer import initialize_algorithm
from ip_drit.algorithms.single_model_algorithm import ModelAlgorithm
from ip_drit.common.grouper import CombinatorialGrouper
from ip_drit.datasets import AbstractPublicDataset
from ip_drit.datasets.camelyon17 import CamelyonDataset
from ip_drit.logger import Logger
from ip_drit.models.wild_model_initializer import WildModel
from ip_drit.patch_transform import TransformationType


def main(argv):
    """Demo scripts for training, evaluation with Camelyon."""
    del argv
    logging.info("Running the Camelyon 17 dataset benchmark.")
    all_dataset_dir = Path("/Users/tan.nguyen/datasets")
    all_dataset_dir.mkdir(exist_ok=True)
    camelyon_dataset = CamelyonDataset(dataset_dir=all_dataset_dir / "camelyon17/")

    log_dir = Path("/Users/tan.nguyen/erm_camelyon")
    log_dir.mkdir(exist_ok=True)

    config_dict: Dict[str, Any] = {
        "algorithm": ModelAlgorithm.ERM,
        "model": WildModel.DENSENET121,
        "transform": TransformationType.WEAK,
        "target_resolution": None,  # Keep the original dataset resolution
        "scheduler_metric_split": "val",
        "group_by_fields": ["hospital"],
        "loss_function": "multitask_bce",
        "algo_log_metric": "accuracy",
        "log_dir": str(log_dir),
        "gradient_accumulation_steps": 2,
        "n_epochs": 20,
        "log_every_n_batches": 2,
        "train_loader": "group",
        "batch_size": 64,
        "uniform_over_groups": True,  # If True, sample examples such that batches are uniform over groups.
        "distinct_groups": False,  # If True, enforce groups sampled per batch are distinct.
        "n_groups_per_batch": 1,  # 4
        "scheduler": "linear_schedule_with_warmup",
        "scheduler_kwargs": {"num_warmup_steps": 3},
        "scheduler_metric_name": "scheduler_metric_name",
        "no_group_logging": True,
        "optimizer": "SGD",
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "optimizer_kwargs": {"momentum": 0.9},
        "max_grad_norm": 0.5,
        "use_data_parallel": use_data_parallel(),
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "use_unlabeled_y": False,  # If true, unlabeled loaders will also the true labels for the unlabeled data.
        "verbose": True,
        "val_metric": "acc_avg",
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

    train_grouper = CombinatorialGrouper(dataset=camelyon_dataset, groupby_fields=config_dict["group_by_fields"])

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


if __name__ == "__main__":
    app.run(main)
