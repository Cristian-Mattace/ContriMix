"""A module to compute the HistauGAN loss."""
from enum import auto
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io

from ..common.metrics._base import Metric
from ..common.metrics._base import MultiTaskMetric
from ._contrimix_loss import ContriMixAggregationType
from ip_drit.algorithms._contrimix_utils import ContrimixTrainingMode


class HistauGANLoss(MultiTaskMetric):
    """A class that defines a HistauGAN loss in absorbance space.

    Because most of the loss term are calculated in the algoirhm class. This class is mainly for logging.

    Args:
        loss_params: A dictionary that defines the parameters for the loss.
        name (optional): The name of the loss. Defaults to "histaugan_loss".
    """

    def __init__(self, loss_params: Dict[str, Any], name: Optional[str] = "histaugan_loss") -> None:
        self._loss_fn = loss_params["loss_fn"]
        self._training_mode: ContrimixTrainingMode = loss_params["training_mode"]
        self._aggregation: ContriMixAggregationType = loss_params.get("aggregation", ContriMixAggregationType.MEAN)
        super().__init__(name)
        self._use_original_image_for_entropy_loss = True

    def compute(
        self, in_dict: Dict[str, torch.Tensor], return_dict: bool = True, return_loss_components: bool = True
    ) -> Union[Metric, Dict]:
        """Computes the metrics, which is a linear combination of different loss term.

        Args:
            in_dict: A dictionary from the inputs of the forward pass of contrimix, key by the name of the field.
            return_dict: Whether to return the output as a dictionary or a tensor.

        Returns:
            The value of the backbone loss.
        """
        self._is_training = in_dict["is_training"]
        self._batch_transform = getattr(in_dict, "batch_transform", None)
        entropy_loss = self._compute_backbone_loss(in_dict=in_dict, x_self_recon=None, zc=None, za_targets=None)

        total_loss = entropy_loss
        if return_dict:
            if return_loss_components:
                return {self.agg_metric_field: total_loss, "entropy_loss": entropy_loss.item()}
            else:
                return {self.agg_metric_field: total_loss}
        else:
            return total_loss

    def _compute_backbone_loss(
        self, in_dict: Dict[str, torch.Tensor], x_self_recon: torch.Tensor, za_targets: torch.Tensor, zc: torch.Tensor
    ) -> torch.Tensor:
        """Computes the cross-entropy loss for the backbone.

        Args:
            in_dict: A dictionary from the inputs of the forward pass of contrimix, key by the name of the field.
            x_self_recon: The self-reconstruction image.
            za_targets: The target attributes to borrow.
            zc: The content vectors.
            return_loss_components: If True, returns different loss components.
        """
        backbone = in_dict["backbone"]
        x_org_1 = in_dict["x_org"]
        y_true = in_dict["y_true"]

        if self._use_original_image_for_entropy_loss:
            backbone_inputs = x_org_1
            backbone_inputs_extended = [x_org_1]

        num_total_ims_to_backbone = 1

        y_true_extended = y_true.repeat(num_total_ims_to_backbone)
        backbone_inputs_extended = torch.cat(backbone_inputs_extended, dim=0)

        # Compute the prediction on the training set.
        in_dict["y_pred"] = backbone(backbone_inputs)
        losses = self._initialize_cross_entropy_loss(
            backbone_input=backbone_inputs_extended, y_true=y_true_extended, backbone=backbone
        )
        losses = losses.reshape(-1, num_total_ims_to_backbone)  # [#images, #augs]
        # The following aggregation is over the augnetaitons.
        if self._aggregation == ContriMixAggregationType.MAX:
            return torch.mean(losses.max(dim=1)[0])
        elif self._aggregation == ContriMixAggregationType.MEAN:
            return torch.mean(losses.mean(dim=1))
        elif self._aggregation == ContriMixAggregationType.AUGREG:
            return torch.mean(losses.mean(dim=1)) + self._aug_reg_variance_weight * torch.mean(torch.var(losses, dim=1))
        else:
            raise ValueError(f"Aggregation type of {self._aggregation} is not supported!")

    def _initialize_cross_entropy_loss(
        self, backbone_input: torch.Tensor, y_true: torch.Tensor, backbone: nn.Module
    ) -> torch.Tensor:
        """Returns a tuple of y_pred logits and the loss values."""
        return self._loss_fn(backbone(backbone_input).float(), y_true)
