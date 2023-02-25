"""A utility module for saving."""
import logging
import os
import re
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import torch
import torch.nn as nn

from ip_drit.algorithms.multi_model_algorithm import MultimodelAlgorithm
from ip_drit.algorithms.single_model_algorithm import SingleModelAlgorithm
from ip_drit.common.metrics import Metric


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


def _save_model(
    algorithm: Union[SingleModelAlgorithm, MultimodelAlgorithm], epoch: int, best_val_metric: float, path: str
) -> None:
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
    logging.info(f"Saving the model to {path}.")


def load(module: nn.Module, path: str, device: Optional[str] = None, tries: int = 2) -> Tuple[int, Metric]:
    """Handles loading weights saved from this repo/model into an algorithm/model.

    Attempts to handle key mismatches between this module's state_dict and the loaded state_dict.

    Args:
        module: module to load parameters for.
        path: path to .pth file
        device: device to load tensors on
        tries: number of times to run the match_keys() function

    Returns:
        prev_epoch: The index of the previous epoch.
        best_val_metric: best_val_metric in state.
    """
    if device is not None:
        state = torch.load(path, map_location=device)
    else:
        state = torch.load(path)

    # Loading from a saved WILDS Algorithm object
    if "algorithm" in state:
        prev_epoch = state["epoch"]
        best_val_metric = state["best_val_metric"]
        state = state["algorithm"]
    # Loading from a pretrained SwAV model
    elif "state_dict" in state:
        state = state["state_dict"]
        prev_epoch, best_val_metric = None, None
    else:
        prev_epoch, best_val_metric = None, None

    # If keys match perfectly, load_state_dict() will work
    try:
        module.load_state_dict(state)
    except Exception:
        # Otherwise, attempt to reconcile mismatched keys and load with strict=False
        module_keys = module.state_dict().keys()
        for _ in range(tries):
            state = _match_keys(state, list(module_keys))
            module.load_state_dict(state, strict=False)
            leftover_state = {k: v for k, v in state.items() if k in list(state.keys() - module_keys)}
            leftover_module_keys = module_keys - state.keys()
            if len(leftover_state) == 0 or len(leftover_module_keys) == 0:
                break
            state, module_keys = leftover_state, leftover_module_keys
        if len(module_keys - state.keys()) > 0:
            logging.warning(
                f"Some module parameters could not be found in the loaded state:" f" {module_keys-state.keys()}"
            )
    return prev_epoch, best_val_metric


def load_model_state_dict_from_checkpoint(model_path: str, start_str: str = "_model.module.") -> OrderedDict:
    """Loads the module from checkpoint.

    Args:
        network: A module to load the parameter from the checkpoint.
        path: path to .pth file that contains the trained model.
        start_str: A first few letters of the network variable to load the checkpoint from. Defaults to  "._model.".

    Returns:
        A state dict to load the paramater from
    """
    state_dict = torch.load(model_path)["algorithm"]
    return OrderedDict([k[len(start_str) :], v] for k, v in state_dict.items() if k.startswith(start_str))


def _match_keys(d, ref):
    """Matches the format of keys between d (a dict) and ref (a list of keys).

    Helper function for situations where two algorithms share the same model, and we'd like to warm-start one
    algorithm with the model of another. Some algorithms (e.g. FixMatch) save the featurizer, classifier within a
    sequential, and thus the featurizer keys may look like 'model.module.0._' 'model.0._' or 'model.module.model.0._',
    and the classifier keys may look like 'model.module.1._' 'model.1._' or 'model.module.model.1._'
    while simple algorithms (e.g. ERM) use no sequential 'model._'
    """
    # hard-coded exceptions
    d = {re.sub("model.1.", "model.classifier.", k): v for k, v in d.items()}
    d = {k: v for k, v in d.items() if "pre_classifier" not in k}  # this causes errors

    # probe the proper transformation from d.keys() -> reference
    # do this by splitting d's first key on '.' until we get a string that is a strict substring of something in ref
    success = False
    probe = list(d.keys())[0].split(".")
    for i in range(len(probe)):
        probe_str = ".".join(probe[i:])
        matches = list(
            filter(lambda ref_k: len(ref_k) >= len(probe_str) and probe_str == ref_k[-len(probe_str) :], ref)
        )
        matches = list(
            filter(lambda ref_k: "layer" not in ref_k, matches)
        )  # handle resnet probe being too simple, e.g. 'weight'
        if len(matches) == 0:
            continue
        else:
            success = True
            append = [m[: -len(probe_str)] for m in matches]
            remove = ".".join(probe[:i]) + "."
            break
    if not success:
        raise Exception("These dictionaries have irreconcilable keys")

    return_d = {}
    for a in append:
        for k, v in d.items():
            return_d[re.sub(remove, a, k)] = v

    # hard-coded exceptions
    if "model.classifier.weight" in return_d:
        return_d["model.1.weight"], return_d["model.1.bias"] = (
            return_d["model.classifier.weight"],
            return_d["model.classifier.bias"],
        )
    return return_d
