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
        name (optional): The name of the loss. Defaults to None, in which case, the default name of "contrimix_loss"
            will be used.
    """

    def __init__(self, loss_fn: Optional[Callable], name: Optional[str] = "contrimix_loss") -> None:
        self._loss_fn = loss_fn
        super().__init__(name)

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

        x_abs_self_recon = im_gen(zc, za)
        x_self_recon = abs_to_trans_cvt(im_and_sig_type=(x_abs_self_recon, sig_type))[0]

        if backbone.needs_y_input:
            raise ValueError("Backbone network with y-input is not supported")
        else:
            y_pred = backbone(x_self_recon)

        in_dict["y_pred"] = y_pred

        if numel(y_true) == 0:
            if hasattr(y_true, "device"):
                agg_metric = torch.tensor(0.0, device=y_true.device)
            else:
                agg_metric = torch.tensor(0.0)
        else:
            agg_metric = self._compute(y_pred, y_true)

        if return_dict:
            results = {self.agg_metric_field: agg_metric.item()}
            return results
        else:
            return agg_metric

    def _compute_flattened_metrics(
        self, flattened_y_pred: torch.Tensor, flattened_y_true: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self._loss_fn, torch.nn.BCEWithLogitsLoss):
            flattened_y_pred = flattened_y_pred.float()
            flattened_y_true = flattened_y_true.float()
        elif isinstance(self._loss_fn, torch.nn.CrossEntropyLoss):
            flattened_y_true = flattened_y_true.long()
        flattened_y_true = torch.reshape(flattened_y_true, flattened_y_pred.shape)
        flattened_loss = self._loss_fn(flattened_y_pred, flattened_y_true)
        return flattened_loss
