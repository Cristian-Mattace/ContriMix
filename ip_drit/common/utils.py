"""A utilities for the common package."""
import logging
import re
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn

from ip_drit.common.metrics._base import Metric


def numel(obj: Union[torch.Tensor, List[Any]]) -> int:
    """Gets the length of an object.

    Args:
        obj: The object to get the length from.
    """
    if torch.is_tensor(obj):
        return obj.numel()
    elif isinstance(obj, list):
        return len(obj)
    else:
        raise TypeError("Invalid type for numel")


def load(module: nn.Module, path: str, device: Optional[str] = None, tries: int = 2) -> Tuple[int, Metric]:
    """Handles loading weights saved from this repo/model into an algorithm/model.

    Attempts to handle key mismatches between this module's state_dict and the loaded state_dict.

    Args:
        module: module to load parameters for
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
    except Exception as e:
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
        raise e
    return prev_epoch, best_val_metric


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
