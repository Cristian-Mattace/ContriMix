"""A utilities for the common package."""
import logging
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn

from ip_drit.common.metrics._metric import Metric


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
    """
    Handles loading weights saved from this repo/model into an algorithm/model.
    Attempts to handle key mismatches between this module's state_dict and the loaded state_dict.


    Args:
        - module: module to load parameters for
        - path: path to .pth file
        - device: device to load tensors on
        - tries: number of times to run the match_keys() function
    Returns:
        prev_epoch: model at given epoch
        best_val_metric: best_val_metric in state
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
    except:
        # Otherwise, attempt to reconcile mismatched keys and load with strict=False
        module_keys = module.state_dict().keys()
        for _ in range(tries):
            state = match_keys(state, list(module_keys))
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
