import itertools
from typing import Any
from typing import Dict
from typing import List

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim import SGD


def initialize_optimizer(config: Dict[str, Any], models: List[nn.Module]) -> torch.optim.Optimizer:
    """Initializes an initializer."""
    if config["optimizer"] == "SGD":
        optimizer = SGD(
            _get_parameters_from_models(models),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            **config["optimizer_kwargs"],
        )
    elif config["optimizer"] == "AdamW":
        if "bert" in config["model"] or "gpt" in config["model"]:
            _ = ["bias", "LayerNorm.weight"]
        else:
            _ = []

        # params = [
        #    {
        #        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #        "weight_decay": config["weight_decay"],
        #    },
        #    {
        #        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #        "weight_decay": 0.0,
        #    },
        # ]
        # optimizer = AdamW(params, lr=config["lr"], **config["optimizer_kwargs"])

    elif config["optimizer"] == "Adam":
        optimizer = Adam(
            _get_parameters_from_models(models),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            **config["optimizer_kwargs"],
        )
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not recognized.")

    return optimizer


def _get_parameters_from_models(models: List[nn.Module]) -> List:
    params = []
    for model in models:
        params = params + list(model.parameters())
    return params
