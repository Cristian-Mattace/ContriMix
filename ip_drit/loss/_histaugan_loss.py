"""A module to compute the HistauGAN loss."""
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
from torchvision import transforms as tfs

from ..common.metrics._base import Metric
from ..common.metrics._base import MultiTaskMetric

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

_NUM_DOMAINS = 5
_NUM_ATTRS = 8
_CROP_SIZE = 216


class HistauGANLoss(MultiTaskMetric):
    """A class that defines a HistauGAN loss in absorbance space.

    Because most of the loss term are calculated in the algoirhm class. This class is mainly for logging.

    Args:
        loss_params: A dictionary that defines the parameters for the loss.
        name (optional): The name of the loss. Defaults to "histaugan_loss".
    """

    def __init__(self, loss_params: Dict[str, Any], name: Optional[str] = "histaugan_loss") -> None:
        self._loss_fn = loss_params["loss_fn"]
        self._upsampler = tfs.Resize(size=(_CROP_SIZE, _CROP_SIZE), interpolation=tfs.InterpolationMode.BILINEAR)
        self._downsampler = tfs.Resize(
            size=loss_params["original_resolution"], interpolation=tfs.InterpolationMode.BILINEAR
        )
        self._original_resolution = loss_params["original_resolution"]
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
        x_org = in_dict["x"]

        # Compute the prediction on the training set, not doing any synthetic images here.
        in_dict["y_pred"] = backbone(x_org)
        if self._is_training:
            x_aug = self._replace_half_with_synthetic_images(
                enc_c=in_dict["enc_c"],
                gen=in_dict["gen"],
                x_org=x_org,
                target_domain_indices=in_dict["target_domain_indices"],
            )
        else:
            x_aug = x_org

        losses = self._initialize_cross_entropy_loss(backbone_input=x_aug, y_true=in_dict["y_true"], backbone=backbone)
        losses = losses.reshape(-1, 1)
        return torch.mean(losses.mean(dim=1))

    def _replace_half_with_synthetic_images(
        self, enc_c: nn.Module, gen: nn.Module, x_org: torch.Tensor, target_domain_indices: torch.Tensor
    ) -> torch.Tensor:
        """Augment half of the image with HistAuGan.

        Ported from https://github.com/sophiajw/HistAuGAN/blob/main/README.md.
        """
        x_augmented = x_org.clone()
        bs = x_org.size(0)
        indices = torch.randint(2, (bs,), device=x_org.device)  # augmentations are applied with probability 0.5
        num_aug = indices.sum()

        if num_aug > 0:
            num_target_domains = target_domain_indices.size(0)
            aug_target_domain_indices = target_domain_indices[
                torch.randint(low=0, high=num_target_domains, size=(num_aug,))
            ]

            aug_target_domain_one_hots = torch.eye(_NUM_DOMAINS, device=x_org.device)[aug_target_domain_indices]
            z_attr = (
                torch.randn((num_aug, _NUM_ATTRS)) * _STD_DOMAINS[aug_target_domain_indices]
                + _MEAN_DOMAINS[aug_target_domain_indices]
            ).to(x_org.device)

            # compute content encoding
            z_c = enc_c(self._upsampler(x_augmented[indices.bool()]))
            # generate augmentations
            x_augmented[indices.bool()] = self._downsampler(
                gen(z_c, z_attr, aug_target_domain_one_hots).detach()
            )  # in range [-1, 1]
        return x_augmented

    def _initialize_cross_entropy_loss(
        self, backbone_input: torch.Tensor, y_true: torch.Tensor, backbone: nn.Module
    ) -> torch.Tensor:
        """Returns a tuple of y_pred logits and the loss values."""
        return self._loss_fn(backbone(backbone_input).float(), y_true)
