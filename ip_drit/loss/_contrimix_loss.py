"""A module that defines the loss class for the contrimix."""
import logging
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import torch

from ..common.metrics._base import Metric
from ..common.metrics._base import MultiTaskMetric
from ..common.metrics._utils import numel
from ip_drit.models import SignalType


class ContriMixLoss(MultiTaskMetric):
    """A class that defines a multi-task loss.

    Args:
        loss_fn: A function to compute the loss from the label, excluding other ContriMix specific loss.
        loss_weights_by_name: A dictionary of loss weights by the type of the loss. The sum of the weight must not be
            larger than than 1.0. Valid keys are:
            "entropy_weight": The weights for the cross entropy loss.
            "self_recon_weight": The weights for the self-recon loss.
        name (optional): The name of the loss. Defaults to None, in which case, the default name of "contrimix_loss"
            will be used.
    """

    def __init__(
        self,
        loss_fn: Optional[Callable],
        loss_weights_by_name: Dict[str, float],
        name: Optional[str] = "contrimix_loss",
    ) -> None:
        self._loss_fn = loss_fn
        self._loss_weights_by_name = self._clean_up_loss_weight_dictionary(loss_weights_by_name)
        super().__init__(name)

    @staticmethod
    def _clean_up_loss_weight_dictionary(loss_weights_by_name: Dict[str, float]) -> Dict[str, float]:
        total_loss_weights = sum(loss_weights_by_name.values())
        if any(x < 0 for x in loss_weights_by_name.values()):
            raise ValueError("All the weights must be non-negative!")

        if total_loss_weights > 1.0:
            raise ValueError("The total weights for all the loss can't be larger than 1")
        else:
            if "entropy_loss_weight" not in loss_weights_by_name:
                loss_weights_by_name["entropy_loss_weight"] = 1.0 - total_loss_weights
        return loss_weights_by_name

    def compute(
        self, in_dict: Dict[str, Union[torch.Tensor, SignalType]], return_dict: bool = True
    ) -> Union[Metric, Dict]:
        """Computes metric. This is a wrapper around _compute.

        Args:
            in_dict: A dictionary from the inputs of the forward pass of contrimix, key by the name of the field.
            return_dict: Whether to return the output as a dictionary or a tensor.

        Returns:
             If return_dict=False:
            - metric (0-dim tensor): metric. If the inputs are empty, returns tensor(0.).
        Output (return_dict=True):
            - results (dict): Dictionary of results, mapping metric.agg_metric_field to avg_metric.
        """
        y_true = in_dict["y_true"]
        im_gen = in_dict["im_gen"]
        backbone = in_dict["backbone"]
        abs_to_trans_cvt = in_dict["abs_to_trans_cvt"]
        _ = in_dict["trans_to_abs_cvt"]
        sig_type = in_dict["sig_type"]
        zc = in_dict["zc"]
        za = in_dict["za"]
        x_org = in_dict["x_org"]

        x_abs_self_recon = im_gen(zc, za)
        x_self_recon = abs_to_trans_cvt(im_and_sig_type=(x_abs_self_recon, sig_type))[0]

        if backbone.needs_y_input:
            raise ValueError("Backbone network with y-input is not supported")
        else:
            y_pred = backbone(x_self_recon)

        in_dict["y_pred"] = y_pred

        self_recon_loss = self._self_recon_consistency_loss(self_recon_ims=x_abs_self_recon, expected_ims=x_org)
        entropy_loss = self._compute_entropy_loss_from_logits(y_pred, y_true)
        total_loss = (
            self._loss_weights_by_name.get("self_recon_weight", 0.0) * self_recon_loss
            + self._loss_weights_by_name["entropy_weight"] * entropy_loss
        )
        if return_dict:
            # Used for updating logs.
            return {
                self.agg_metric_field: total_loss,
                "self_recon_loss": self_recon_loss.item(),
                "entropy_loss": entropy_loss.item(),
            }
        else:
            # Used for updating the objective.
            return total_loss

    @staticmethod
    def _self_recon_consistency_loss(self_recon_ims: torch.Tensor, expected_ims: torch.Tensor) -> float:
        return torch.nn.L1Loss(reduction="mean")(self_recon_ims, expected_ims)

    def _compute(self, y_pred, y_true):
        flattened_metrics, _ = self._compute_flattened(y_pred, y_true, return_dict=False)
        if flattened_metrics.numel() == 0:
            return torch.tensor(0.0, device=y_true.device)
        else:
            return flattened_metrics.mean()

    def _compute_flattened(self, y_pred, y_true, return_dict=True):
        is_labeled = ~torch.isnan(y_true)
        batch_idx = torch.where(is_labeled)[0]
        flattened_y_pred = y_pred[is_labeled]
        flattened_y_true = y_true[is_labeled]
        flattened_metrics = self._compute_flattened_metrics(flattened_y_pred, flattened_y_true)
        if return_dict:
            return {self.name: flattened_metrics, "index": batch_idx}
        else:
            return flattened_metrics, batch_idx

    def _compute_entropy_loss_from_logits(self, y_pred_logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Returns the cross-entropy loss from logits."""
        is_labeled = ~torch.isnan(y_true)
        y_pred_logits = y_pred_logits[is_labeled]
        y_true = y_true[is_labeled]
        if isinstance(self._loss_fn, torch.nn.BCEWithLogitsLoss):
            y_pred_logits = y_pred_logits.float()
            y_true = y_true.float()
        elif isinstance(self._loss_fn, torch.nn.CrossEntropyLoss):
            y_true = y_true.long()
        y_true = torch.reshape(y_true, y_pred_logits.shape)
        loss = self._loss_fn(y_pred_logits, y_true)

        if loss.numel() == 0:
            return torch.tensor(0.0, device=y_true.device)
        return loss.mean()
