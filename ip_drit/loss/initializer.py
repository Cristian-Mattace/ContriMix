"""A module that initializes different losses."""
import torch
import torch.nn as nn

from ip_drit.common.metrics import ElementwiseLoss
from ip_drit.common.metrics import MSE
from ip_drit.common.metrics import MultiTaskLoss


def initialize_loss(loss_type: str) -> nn.Module:
    """Initalizes the loss module."""
    if loss_type == "cross_entropy":
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction="none", ignore_index=-100))

    elif loss_type == "lm_cross_entropy":
        return MultiTaskLoss(loss_fn=nn.CrossEntropyLoss(reduction="none", ignore_index=-100))

    elif loss_type == "mse":
        return MSE(name="loss")

    elif loss_type == "multitask_bce":
        return MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction="none"))

    elif loss_type == "cross_entropy_logits":
        return ElementwiseLoss(loss_fn=cross_entropy_with_logits_loss)

    else:
        raise ValueError(f"loss type {loss_type} is not recognized")


def cross_entropy_with_logits_loss(input: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
    r"""Implementation of CrossEntropy loss using a soft target.

    This is an extension of BCEWithLogitsLoss to MCE.
    Normally, cross entropy loss is
         \sum_j 1{j == y} -log \frac{e^{s_j}}{\sum_k e^{s_k}} = -log \frac{e^{s_y}}{\sum_k e^{s_k}}
    Here we use
        \sum_j P_j *-log \frac{e^{s_j}}{\sum_k e^{s_k}}
    where 0 <= P_j <= 1
    Does not support fancy nn.CrossEntropy options (e.g. weight, size_average, ignore_index, reductions, etc.)

    Args:
        input: A tensor of size (N, K) that contains the logit value.
        soft_target: A target tensor for softmax(input); likely want to use class probabilities

    Returns:
        A tensor of size (N, 1) for the losses.
    """
    return torch.sum(-soft_target * torch.nn.functional.log_softmax(input, 1), 1)
