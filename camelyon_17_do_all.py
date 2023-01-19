"""A script to run the benchmark for the Camelyon dataset."""
import argparse
import logging
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
def main(argv):
    del argv
    logging.info("Running the Camelyon 17 dataset benchmark.")
    all_dataset_dir = Path("/Users/tan.nguyen/datasets")
    all_dataset_dir.mkdir(exist_ok=True)
    camelyon_dataset = CamelyonDataset(dataset_dir=all_dataset_dir / "camelyon17/")

    config_dict: Dict[str, Any] = {
        'algorithm': ModelAlgorithm.ERM,
        'model': WildModel.DENSENET121,
        'group_by_fields': ['hospital']
    }

    initialize_algorithm(
        config=config_dict,
        split_dict_by_name=_configure_split_dict_by_names(full_dataset=camelyon_dataset),
        train_grouper=_initialize_grouper(full_dataset=camelyon_dataset, group_by_fields=config_dict['group_by_fields'])
    )

def _configure_split_dict_by_names(full_dataset: AbstractPublicDataset) -> Dict[str, Dict]:
    """Configure the split dict for different splits"""
    config_dict = defaultdict(dict)
    for split_name in full_dataset.split_dict:
        logging.info(f"Generating split dict for split {split_name}")

def _initialize_grouper(full_dataset: AbstractPublicDataset, group_by_fields: List[str]) -> CombinatorialGrouper:
    return CombinatorialGrouper(
        dataset=full_dataset,
        groupby_fields=group_by_fields,
    )

if __name__ == "__main__":
    app.run(main)