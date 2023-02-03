import itertools
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim import SGD


def initialize_optimizer(config: Dict[str, Any], models: List[nn.Module]) -> torch.optim.Optimizer:
    """Initializes an initializer.

    Args:
        config: A configuration dictionary.
        models: A list of models that shares the same optimizer.
    """
    all_params = get_parameters_from_models(models)
    if config["optimizer"] == "SGD":
        optimizer = SGD(
            all_params, lr=config["lr"], weight_decay=config["weight_decay"], **config["optimizer_kwargs"]["SGD"]
        )
    elif config["optimizer"] == "AdamW":
        optimizer = AdamW(all_params, lr=config["lr"], **config["optimizer_kwargs"]["AdamW"])

    elif config["optimizer"] == "Adam":
        optimizer = Adam(
            all_params, lr=config["lr"], weight_decay=config["weight_decay"], **config["optimizer_kwargs"]["Adam"]
        )
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not recognized.")

    return optimizer


def get_parameters_from_models(models: Iterable[nn.Module]) -> List:
    """Returns a list of parameters from all models."""
    params = []
    for model in models:
        params = params + list(model.parameters())
    return params
