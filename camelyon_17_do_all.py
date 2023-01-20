"""A script to run the benchmark for the Camelyon dataset."""
import argparse
import logging

import torch.cuda
from absl import app
from ip_drit.datasets.camelyon17 import CamelyonDataset
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from collections import defaultdict
from ip_drit.datasets import AbstractPublicDataset
from ip_drit.models.wild_model_initializer import WildModel
from ip_drit.algorithms.single_model_algorithm import ModelAlgorithm
from ip_drit.algorithms.initializer import initialize_algorithm
from ip_drit.common.grouper import CombinatorialGrouper
from ip_drit.common.data_loaders import get_train_loader
from ip_drit.datasets import SubsetPublicDataset
from ip_drit.logger import Logger
from torch.utils.data import DataLoader
from train import train

def main(argv):
    del argv
    logging.info("Running the Camelyon 17 dataset benchmark.")
    all_dataset_dir = Path("/Users/tan.nguyen/datasets")
    all_dataset_dir.mkdir(exist_ok=True)
    camelyon_dataset = CamelyonDataset(dataset_dir=all_dataset_dir / "camelyon17/")

    log_dir = Path("/Users/tan.nguyen/erm_camelyon")
    log_dir.mkdir(exist_ok=True)

    config_dict: Dict[str, Any] = {
        'algorithm': ModelAlgorithm.ERM,
        'model': WildModel.DENSENET121,
        'group_by_fields': ['hospital'],
        'loss_function': 'cross_entropy',
        'algo_log_metric': 'accuracy',
        'gradient_accumulation_steps': 1,
        'n_epochs': 5,
        'train_loader': 'group',
        'batch_size': 32,
        'uniform_over_groups': True,  # If True, sample examples such that batches are uniform over groups.
        'distinct_groups': True, # If True, enforce groups sampled per batch are distinct.
        'n_groups_per_batch': 4,

        'scheduler': 'linear_schedule_with_warmup',
        'scheduler_kwargs': {
            'num_warmup_steps': 3,
        },
        'scheduler_metric_name': 'scheduler_metric_name',

        'no_group_logging': True,

        'optimizer': 'SGD',
        'lr': 1e-3,
        'weight_decay': 1e-2,
        'optimizer_kwargs': {
            'momentum': 0.9,
        },

        'max_grad_norm': 0.5,
        'use_data_parallel': _use_data_parallel(),
        'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),

        'use_unlabeled_y': False,  # If true, unlabeled loaders will also the true labels for the unlabeled data.
    }

    logger = Logger(fpath=str(log_dir / 'log.txt'))

    train_grouper = _initialize_grouper(full_dataset=camelyon_dataset, group_by_fields=config_dict['group_by_fields'])
    algorithm = initialize_algorithm(
        config=config_dict,
        split_dict_by_name=_configure_split_dict_by_names(
            full_dataset=camelyon_dataset,
            grouper=train_grouper,
            config_dict=config_dict,
        ),
        train_grouper=train_grouper
    )

    train(
        algorithm=algorithm,
        general_logger=logger,
        config_dict=config_dict,
        epoch_offset=0,
    )


def _initialize_grouper(full_dataset: AbstractPublicDataset, group_by_fields: List[str]) -> CombinatorialGrouper:
    return CombinatorialGrouper(
        dataset=full_dataset,
        groupby_fields=group_by_fields,
    )

def _configure_split_dict_by_names(
        full_dataset: AbstractPublicDataset,
        grouper: CombinatorialGrouper,
        config_dict: Dict[str, Any],
) -> Dict[str, Dict]:
    """Configures the split dict for different splits."""
    split_dict = defaultdict(dict)
    for split_name in full_dataset.split_dict:
        logging.info(f"Generating split dict for split {split_name}")
        split_dict[split_name]['dataset'] = full_dataset.get_subset(
            split=split_name,
            frac=1.0,
            transform=None
        )

        split_dict[split_name]['loader'] = _get_data_loader_by_split_name(
            sub_dataset=split_dict[split_name]['dataset'],
            grouper=grouper,
            split_name=split_name,
            config_dict=config_dict,
        )
    return split_dict

def _get_data_loader_by_split_name(
        sub_dataset: SubsetPublicDataset,
        grouper: CombinatorialGrouper,
        split_name: str,
        config_dict: Dict[str, Any]
) -> DataLoader:
    if split_name == 'train':
        return get_train_loader(
            loader_type = config_dict['train_loader'],
            dataset = sub_dataset,
            batch_size = config_dict['batch_size'],
            uniform_over_groups = config_dict['uniform_over_groups'],
            grouper = grouper,
            distinct_groups = config_dict['distinct_groups'],
            n_groups_per_batch = config_dict['n_groups_per_batch'],
        )

def _use_data_parallel() -> bool:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logging.info(f"Only {device_count} GPU(s) detected")
        return device_count > 1
    return False


if __name__ == "__main__":
    app.run(main)