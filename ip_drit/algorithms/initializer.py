"""A module that initialize different algorithms."""
import math
from typing import Any
from typing import Dict
import logging
from .single_model_algorithm import SingleModelAlgorithm
from .single_model_algorithm import ModelAlgorithm
from ._erm import ERM
from ip_drit.datasets import SubsetPublicDataset
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.loss.initializer import initialize_loss
from ip_drit.common.metrics import MSE
from ip_drit.common.metrics import Accuracy
from ip_drit.common.metrics import MultiTaskAccuracy
from ip_drit.common.metrics import MultiTaskAveragePrecision
from ip_drit.common.metrics import multiclass_logits_to_pred
from ip_drit.common.metrics import binary_logits_to_pred

algo_log_metrics = {
    'accuracy': Accuracy(prediction_fn=binary_logits_to_pred),
    'mse': MSE(),
    'multitask_accuracy': MultiTaskAccuracy(prediction_fn=multiclass_logits_to_pred),
    'multitask_binary_accuracy': MultiTaskAccuracy(prediction_fn=binary_logits_to_pred),
    'multitask_avgprec': MultiTaskAveragePrecision(prediction_fn=None),
    None: None,
}

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
        loss: The loss module to use.
        metrics: The metrics to use.

    Returns:
        The initialized algorithm.
    """

    train_loader = split_dict_by_name['train']['loader']

    if config['algorithm'] == ModelAlgorithm.ERM:
        return ERM(
            config=config,
            d_out=1,   # Classification problem for now
            grouper=train_grouper,
            loss=initialize_loss(loss_type=config['loss_function']),
            metric=algo_log_metrics[config['algo_log_metric']],
            n_train_steps= math.ceil(len(train_loader)/config['gradient_accumulation_steps']) * config['n_epochs'],
        )
