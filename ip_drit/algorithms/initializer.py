"""A module that initialize different algorithms."""
import logging
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import torch.nn as nn

from ._contrimix import ContriMix
from ._erm import ERM
from ._noisy_student import NoisyStudent
from .single_model_algorithm import ModelAlgorithm
from .single_model_algorithm import SingleModelAlgorithm
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.metrics import Accuracy
from ip_drit.common.metrics import binary_logits_to_pred
from ip_drit.common.metrics import MSE
from ip_drit.common.metrics import multiclass_logits_to_pred
from ip_drit.common.metrics import MultiTaskAccuracy
from ip_drit.common.metrics import MultiTaskAveragePrecision
from ip_drit.loss import ContriMixLoss
from ip_drit.loss.initializer import initialize_loss
from saving_utils import load

algo_log_metrics = {
    "accuracy": Accuracy(prediction_fn=binary_logits_to_pred),
    "mse": MSE(),
    "multitask_accuracy": MultiTaskAccuracy(prediction_fn=multiclass_logits_to_pred),
    "multitask_binary_accuracy": MultiTaskAccuracy(prediction_fn=binary_logits_to_pred),
    "multitask_avgprec": MultiTaskAveragePrecision(prediction_fn=None),
    None: None,
}


def initialize_algorithm(
    config: Dict[str, Any],
    labeled_split_dict_by_name: Dict[str, Dict],
    train_grouper: AbstractGrouper,
    unlabeled_split_dict_by_name: Optional[Dict[str, Dict]] = None,
) -> SingleModelAlgorithm:
    """Initializes an algorithm based on the provided config dictionary.

    Args:
        config: A dictionary that is used to configure hwo the model should be initialized.
        labeled_split_dict_by_name: A dictionary whose key are 'train', 'val', 'id_val', 'test' of different splits.
            For each key, the value is a dictionary with further (key, values) that defines the attribute of the
            split. They key can be 'loader' (for the data loader), 'dataset' (for the dataset), 'name' (for the name of
            the split), 'eval_logger', and 'algo_logger'.
        train_grouper: A grouper object that defines the groups for which we compute/log statistics for.
        unlabeled_split_dict_by_name (optional):  A dictionary whose key are 'train_unlabeled', 'val_unlabeled',
            'test_unlabeled' corresponding to different splits. For each key, the value is a dictionary with further
            (key, values) that defines the attribute of the split. They key can be 'loader' (for the data loader),
            'dataset' (for the dataset), 'name' (for the name of the split), 'eval_logger', and 'algo_logger'. Defaults
            to None.

    Returns:
        The initialized algorithm.
    """
    train_loader = labeled_split_dict_by_name["train"]["loader"]
    num_train_steps = math.ceil(len(train_loader) / config["gradient_accumulation_steps"]) * config["n_epochs"]
    logging.info(f"Initializing the {config['algorithm'].name} algorithm!")

    if config["algorithm"] == ModelAlgorithm.ERM:
        algorithm = ERM(
            config=config,
            d_out=1,  # Classification problem for now
            grouper=train_grouper,
            loss=initialize_loss(loss_type=config["loss_function"]),
            metric=algo_log_metrics[config["algo_log_metric"]],
            n_train_steps=num_train_steps,
        )
    elif config["algorithm"] == ModelAlgorithm.CONTRIMIX:
        logging.warning(
            f"Initlializing the ContriMix algorithm, using ContriMixLoss ignoring the specified loss type of"
            + f"{config['loss_function']}."
        )

        algorithm = ContriMix(
            config=config,
            d_out=1,
            grouper=train_grouper,
            loss=ContriMixLoss(
                loss_fn=nn.BCEWithLogitsLoss(reduction="none"),
                loss_weights_by_name={
                    "entropy_weight": 0.1,
                    "self_recon_weight": 0.3,
                    "attr_cons_weight": 0.1,
                    "cont_cons_weight": 0.5,
                },
            ),
            metric=algo_log_metrics[config["algo_log_metric"]],
            n_train_steps=num_train_steps,
        )
    elif config["algorithm"] == ModelAlgorithm.NOISY_STUDENT:
        algorithm = NoisyStudent(
            config=config,
            d_out=1,
            grouper=train_grouper,
            loss=initialize_loss(loss_type=config["loss_function"]),
            unlabeled_loss=_compute_unlabeled_loss(
                use_soft_pseudo_label=config["soft_pseudolabels"], loss_type=config["loss_function"]
            ),
            metric=algo_log_metrics[config["algo_log_metric"]],
            n_train_steps=num_train_steps,
        )
    else:
        raise ValueError(f"The algorithm {config['algorithm']} is not supported!")

    if config["pretrained_model_path"] is not None:
        pretrain_model_path = config["pretrained_model_path"]
        logging.info(f"Loading pretrain model from {pretrain_model_path}.")
        load(module=algorithm, path=pretrain_model_path, device=config["device"])
        logging.info(f"Loaded model state from {pretrain_model_path}!")

    return algorithm


def _compute_unlabeled_loss(use_soft_pseudo_label: bool, loss_type: str) -> Callable:
    """Returns a loss function for the unlabeled samples."""
    if use_soft_pseudo_label:
        raise ValueError("NoisyStudent currently does not support soft pseudo labels.")
    else:
        return initialize_loss(loss_type=loss_type)
