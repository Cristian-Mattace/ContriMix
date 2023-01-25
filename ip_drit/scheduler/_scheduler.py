"""A module to define the schedulers."""
from typing import Any
from typing import Dict
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from transformers import get_cosine_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup

from ip_drit.common.metrics import Metric


def initialize_scheduler(
    config_dict: Dict[str, Any], optimizer: torch.optim.Optimizer, n_train_steps: int
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    """Initializes the scheduler.

    Args:
        config_dict: A configuration dictionary.
        optimizer: An optimizer.
        n_train_steps: Number of training steps.
    """
    # construct schedulers
    if config_dict["scheduler"] is None:
        return None
    elif config_dict["scheduler"] == "linear_schedule_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_training_steps=n_train_steps, **config_dict["scheduler_kwargs"]
        )
        step_every_batch = True
        use_metric = False
    elif config_dict["scheduler"] == "cosine_schedule_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=n_train_steps, **config_dict["scheduler_kwargs"]
        )
        step_every_batch = True
        use_metric = False
    elif config_dict["scheduler"] == "ReduceLROnPlateau":
        assert config_dict["scheduler_metric_name"], f"scheduler metric must be specified."
        scheduler = ReduceLROnPlateau(optimizer, **config_dict["scheduler_kwargs"])
        step_every_batch = False
        use_metric = True
    elif config_dict["scheduler"] == "StepLR":
        scheduler = StepLR(optimizer, **config_dict["scheduler_kwargs"])
        step_every_batch = False
        use_metric = False
    elif config_dict["scheduler"] == "FixMatchLR":
        scheduler = LambdaLR(
            optimizer, lambda x: (1.0 + 10 * float(x) / n_train_steps) ** -0.75, **config_dict["scheduler_kwargs"]
        )
        step_every_batch = True
        use_metric = False
    elif config_dict["scheduler"] == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, **config_dict["scheduler_kwargs"])
        step_every_batch = False
        use_metric = False
    else:
        raise ValueError(f"Unknown scheduler!")

    # add an step_every_batch field
    scheduler.step_every_batch = step_every_batch
    scheduler.use_metric = use_metric
    return scheduler


def step_scheduler(scheduler, metric: Optional[Metric] = None) -> None:
    """Steps the scheduler by 1 step.

    Args:
        scheduler: The scheduler to step.
        metric (optional): The metric that is used to step the scheduler.
    """
    if isinstance(scheduler, ReduceLROnPlateau):
        assert metric is not None
        scheduler.step(metric)
    else:
        scheduler.step()
