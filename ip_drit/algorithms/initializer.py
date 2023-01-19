"""A module that initialize different algorithms."""
from typing import Any
from typing import Dict
import logging
from .single_model_algorithm import SingleModelAlgorithm
from .single_model_algorithm import ModelAlgorithm
from ip_drit.common.grouper import AbstractGrouper

def initialize_algorithm(
        config: Dict[str, Any],
        split_dict_by_name: Dict[str, Dict],
        train_grouper: AbstractGrouper) -> SingleModelAlgorithm:
    """
    Initialize an algorithm based on the provided config dictionary.

    Args:
        config: A dictionary that is used to configure hwo the model should be initialized.
        split_dict_by_name: A dictionary whose key are 'train', 'ood_val', 'id_val' corresponding to different splits.
            For each key, the value is a dictionary with further (key, values) that defines the attribute of the split.
            They key can be 'loader' (for the data loader), 'dataset' (for the dataset), 'name' (for the name of the
            split), 'eval_logger', and 'algo_logger'.
        train_grouper: A grouper object that defines the groups for which we compute/log statistics for.


    Returns:
        The initialized algorithm.
    """
    if config['algorithm'] == ModelAlgorithm.ERM:
        pass