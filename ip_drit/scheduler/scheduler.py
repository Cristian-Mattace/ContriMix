"""A module to define the schedulers."""
import torch
from typing import Optional

from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, StepLR, CosineAnnealingLR, MultiStepLR

def step_scheduler(scheduler, metric=None):
    if isinstance(scheduler, ReduceLROnPlateau):
        assert metric is not None
        scheduler.step(metric)
    else:
        scheduler.step()
