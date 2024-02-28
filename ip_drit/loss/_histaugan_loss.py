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
from torchvision import transforms as tfs

from ..common.metrics._base import Metric
from ..common.metrics._base import MultiTaskMetric
from ._contrimix_loss import ContriMixAggregationType
from ip_drit.algorithms._contrimix_utils import ContrimixTrainingMode

# Ported from https://github.com/sophiajw/HistAuGAN/blob/main/augmentations.py#L121C1-L145C2
_MEAN_DOMAINS = torch.Tensor(
    [
        [0.3020, -2.6476, -0.9849, -0.7820, -0.2746, 0.3361, 0.1694, -1.2148],
        [0.1453, -1.2400, -0.9484, 0.9697, -2.0775, 0.7676, -0.5224, -0.2945],
        [2.1067, -1.8572, 0.0055, 1.2214, -2.9363, 2.0249, -0.4593, -0.9771],
        [0.8378, -2.1174, -0.6531, 0.2986, -1.3629, -0.1237, -0.3486, -1.0716],
        [1.6073, 1.9633, -0.3130, -1.9242, -0.9673, 2.4990, -2.2023, -1.4109],
    ]
)

_STD_DOMAINS = torch.Tensor(
    [
        [0.6550, 1.5427, 0.5444, 0.7254, 0.6701, 1.0214, 0.6245, 0.6886],
        [0.4143, 0.6543, 0.5891, 0.4592, 0.8944, 0.7046, 0.4441, 0.3668],
        [0.5576, 0.7634, 0.7875, 0.5220, 0.7943, 0.8918, 0.6000, 0.5018],
        [0.4157, 0.4104, 0.5158, 0.3498, 0.2365, 0.3612, 0.3375, 0.4214],
        [0.6154, 0.3440, 0.7032, 0.6220, 0.4496, 0.6488, 0.4886, 0.2989],
    ]
)


class HistauGANLoss(MultiTaskMetric):
    """A class that defines a HistauGAN loss in absorbance space.

    Because most of the loss term are calculated in the algoirhm class. This class is mainly for logging.

    Args:
        loss_params: A dictionary that defines the parameters for the loss.
        name (optional): The name of the loss. Defaults to "histaugan_loss".
    """

    def __init__(self, loss_params: Dict[str, Any], name: Optional[str] = "histaugan_loss") -> None:
        self._loss_fn = loss_params["loss_fn"]
        self._nz = loss_params["nz"]
        self._upsampler = tfs.Resize(size=(216, 216), interpolation=tfs.InterpolationMode.BICUBIC)
        self._downsampler = tfs.Resize(
            size=loss_params["original_resolution"], interpolation=tfs.InterpolationMode.BICUBIC
        )
        self._original_resolution = loss_params["original_resolution"]
        self._aggregation: ContriMixAggregationType = loss_params.get("aggregation", ContriMixAggregationType.MEAN)
        super().__init__(name)

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
        entropy_loss = self._compute_backbone_loss(in_dict=in_dict)

        total_loss = entropy_loss
        if return_dict:
            if return_loss_components:
                return {self.agg_metric_field: total_loss, "entropy_loss": entropy_loss.item()}
            else:
                return {self.agg_metric_field: total_loss}
        else:
            return total_loss

    def _compute_backbone_loss(self, in_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes the cross-entropy loss for the backbone.

        Args:
            in_dict: A dictionary from the inputs of the forward pass of contrimix, key by the name of the field.
            with the following fields
                x_org: the original image with the range of (-1, 1).
        """
        backbone = in_dict["backbone"]
        y_true = in_dict["y_true"]
        x_org = in_dict["x_org"]
        backbone_inputs_extended = [x_org]

        # Compute the prediction on the training set, not doing any synthetic images here.
        in_dict["y_pred"] = backbone(x_org)
        num_total_domains_to_backbone = 1
        if self._is_training:
            xs_syn = self._generate_synthetic_images(
                enc_c=in_dict["enc_c"],
                gen=in_dict["gen"],
                x_org=x_org,
                target_domain_indices=in_dict["target_domain_indices"],
            )
            backbone_inputs_extended.append(xs_syn)
            num_total_domains_to_backbone += len(in_dict["target_domain_indices"])

        backbone_inputs_extended = torch.cat(backbone_inputs_extended, dim=0)

        losses = self._initialize_cross_entropy_loss(
            backbone_input=backbone_inputs_extended,
            y_true=y_true.repeat(num_total_domains_to_backbone),
            backbone=backbone,
        )
        losses = losses.reshape(-1, num_total_domains_to_backbone)  # [#images, #augs]
        # The following aggregation is over the augnetaitons.
        if self._aggregation == ContriMixAggregationType.MAX:
            return torch.mean(losses.max(dim=1)[0])
        elif self._aggregation == ContriMixAggregationType.MEAN:
            return torch.mean(losses.mean(dim=1))
        elif self._aggregation == ContriMixAggregationType.AUGREG:
            return torch.mean(losses.mean(dim=1)) + self._aug_reg_variance_weight * torch.mean(torch.var(losses, dim=1))
        else:
            raise ValueError(f"Aggregation type of {self._aggregation} is not supported!")

    def _generate_synthetic_images(
        self, enc_c: nn.Module, gen: nn.Module, x_org: torch.Tensor, target_domain_indices: torch.Tensor
    ) -> torch.Tensor:
        """Augment half of the image with HistAuGan.

        Ported from https://github.com/sophiajw/HistAuGAN/blob/main/README.md.
        """
        _NUM_DOMAINS = 5
        bs = x_org.size(0)
        z_c = enc_c(self._upsampler(x_org))
        synthetic_ims = []
        for target_domain_index in target_domain_indices:
            aug_target_domain_one_hots = torch.zeros((bs, _NUM_DOMAINS), device=x_org.device)
            aug_target_domain_one_hots[:, target_domain_index] = 1.0
            z_attr = (torch.randn((bs, 8)) * _STD_DOMAINS[target_domain_index] + _MEAN_DOMAINS[target_domain_index]).to(
                x_org.device
            )
            synthetic_ims.append(gen(z_c, z_attr, aug_target_domain_one_hots))
        return self._downsampler(torch.cat(synthetic_ims, dim=0))

    def _initialize_cross_entropy_loss(
        self, backbone_input: torch.Tensor, y_true: torch.Tensor, backbone: nn.Module
    ) -> torch.Tensor:
        """Returns a tuple of y_pred logits and the loss values."""
        return self._loss_fn(backbone(backbone_input).float(), y_true)
