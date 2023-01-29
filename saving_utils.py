"""A utility module for saving."""
import os
from typing import Any
from typing import Dict

import pandas as pd
import torch

from ip_drit.algorithms.single_model_algorithm import SingleModelAlgorithm


def save_pred_if_needed(
    y_pred: torch.Tensor,
    split_dict: Dict[str, Any],
    epoch: int,
    config_dict: Dict[str, Any],
    is_best: bool,
    force_save=False,
) -> None:
    """Saves a prediction if needed.

    Args:
        y_pred: A tensor that contains the prediction.
        split_dict: A dictionary that defines different things for each split.
        epoch: The epoch index.
        config_dict: A configuration dictionary.
        is_best: If True, the current metrics is the current best metrics so far.
        force_save: If True, the save is obligatory.
    """
    if config_dict["save_pred"]:
        prefix: str = _get_pred_prefix(split_dict, config_dict)
        if force_save or (config_dict["save_step"] is not None and (epoch + 1) % config_dict["save_step"] == 0):
            _save_pred(y_pred, prefix + f"epoch:{epoch}_pred")
        if (not force_save) and config_dict["save_last"]:
            _save_pred(y_pred, prefix + f"epoch:last_pred")
        if config_dict["save_best"] and is_best:
            _save_pred(y_pred, prefix + f"epoch:best_pred")


def _get_pred_prefix(split_dict: Dict[str, Any], config_dict: Dict[str, Any]) -> str:
    dataset_name = split_dict["dataset"].dataset_name
    split = split_dict["split"]
    replicate_str = f"seed:{config_dict['seed']}"
    prefix = os.path.join(config_dict["log_dir"], f"{dataset_name}_split:{split}_{replicate_str}_")
    return prefix


def _save_pred(y_pred: torch.Tensor, path_prefix: str):
    # Single tensor
    if torch.is_tensor(y_pred):
        df = pd.DataFrame(y_pred.numpy())
        df.to_csv(path_prefix + ".csv", index=False, header=False)
    # Dictionary
    elif isinstance(y_pred, dict) or isinstance(y_pred, list):
        torch.save(y_pred, path_prefix + ".pth")
    else:
        raise TypeError("Invalid type for save_pred")


def save_model_if_needed(
    algorithm: SingleModelAlgorithm,
    split_dict: Dict[str, Any],
    epoch: int,
    config_dict: Dict[str, Any],
    is_best: bool,
    best_val_metric: float,
) -> None:
    """Saves the model checkpoint."""
    prefix = _get_model_prefix(split_dict["dataset"].dataset_name, config_dict)
    if config_dict["save_step"] is not None and (epoch + 1) % config_dict["save_step"] == 0:
        _save_model(algorithm, epoch, best_val_metric, prefix + f"epoch:{epoch}_model.pth")
    if config_dict["save_last"]:
        _save_model(algorithm, epoch, best_val_metric, prefix + "epoch:last_model.pth")
    if config_dict["save_best"] and is_best:
        _save_model(algorithm, epoch, best_val_metric, prefix + "epoch:best_model.pth")


def _get_model_prefix(dataset_name: str, config_dict: Dict[str, Any]) -> str:
    return os.path.join(config_dict["log_dir"], f"{dataset_name}_seed:{config_dict['seed']}_")


def _save_model(algorithm: SingleModelAlgorithm, epoch: int, best_val_metric: float, path: str) -> None:
    """Save the model checkpoint.

    Args:
        algorithm: The algorithm used to train the model.
        epoch: The epoch number.
        best_val_metric: The best validation metric.
        path: The path to the saving folder.
    """
    state = {}
    state["algorithm"] = algorithm.state_dict()
    state["epoch"] = epoch
    state["best_val_metric"] = best_val_metric
    torch.save(state, path)
