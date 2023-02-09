"""A module that defines the loss class for the contrimix."""
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn as nn

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
        sig_type = in_dict["sig_type"]
        zc = in_dict["zc"]
        za = in_dict["za"]
        za_targets = in_dict["za_targets"]
        x_org = in_dict["x_org"]

        x_abs_self_recon = im_gen(zc, za)
        x_self_recon = abs_to_trans_cvt(im_and_sig_type=(x_abs_self_recon, sig_type))[0]

        y_pred = backbone(x_self_recon)
        in_dict["y_pred"] = y_pred

        # TODO: investigate whether the consistency should be in the absorbance or transmittance space.
        self_recon_loss = self._self_recon_consistency_loss(self_recon_ims=x_self_recon, expected_ims=x_org)
        # self_recon_loss = self._self_recon_consistency_loss(
        #    self_recon_ims=x_abs_self_recon, expected_ims=in_dict["x_abs_org"]
        # )

        num_mixings = za_targets.shape[0]
        entropy_losses = [self._compute_entropy_loss_from_logits(y_pred, y_true)]
        attr_cons_losses: List[float] = [
            self._attribute_consistency_loss(
                x_abs_cross_translation_im=x_abs_self_recon, expected_za=za, attr_enc=in_dict["attr_enc"]
            )
        ]

        cont_cons_losses: List[float] = [
            self._content_consistency_loss(
                x_abs_cross_translation_im=x_abs_self_recon, expected_zc=zc, cont_enc=in_dict["cont_enc"]
            )
        ]

        # x1 = x_org[0].clone().detach().cpu().numpy().transpose(1, 2, 0)

        for mix_idx in range(num_mixings):
            za_target = za_targets[mix_idx]
            x_abs_cross_translation = im_gen(zc, za_target)
            x_cross_translation = abs_to_trans_cvt(im_and_sig_type=(x_abs_cross_translation, sig_type))[0]

            # x2 = x_cross_translation[0].clone().detach().cpu().numpy().transpose(1, 2, 0)
            entropy_losses.append(self._compute_entropy_loss_from_logits(backbone(x_cross_translation), y_true))

            attr_cons_losses.append(
                self._attribute_consistency_loss(
                    x_abs_cross_translation_im=x_abs_cross_translation,
                    expected_za=za_target,
                    attr_enc=in_dict["attr_enc"],
                )
            )

            cont_cons_losses.append(
                self._content_consistency_loss(
                    x_abs_cross_translation_im=x_abs_cross_translation, expected_zc=zc, cont_enc=in_dict["cont_enc"]
                )
            )

        attr_cons_loss = torch.mean(torch.stack(attr_cons_losses, dim=0))
        cont_cons_loss = torch.mean(torch.stack(cont_cons_losses, dim=0))

        # TODO: test with max here.
        entropy_loss = torch.mean(torch.stack(entropy_losses, dim=0))

        total_loss = (
            self._loss_weights_by_name["self_recon_weight"] * self_recon_loss
            + self._loss_weights_by_name["entropy_weight"] * entropy_loss
            + self._loss_weights_by_name["attr_cons_weight"] * attr_cons_loss
            + self._loss_weights_by_name["cont_cons_weight"] * cont_cons_loss
        )
        if return_dict:
            # Used for updating logs.
            return {
                self.agg_metric_field: total_loss,
                "self_recon_loss": self_recon_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "attribute_consistency_loss": attr_cons_loss.item(),
                "content_consistency_loss": cont_cons_loss.item(),
            }
        else:
            # Used for updating the objective.
            return total_loss

    @staticmethod
    def _self_recon_consistency_loss(self_recon_ims: torch.Tensor, expected_ims: torch.Tensor) -> float:
        return torch.nn.L1Loss(reduction="mean")(self_recon_ims, expected_ims)

    def _compute_entropy_loss_from_logits(self, y_pred_logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Returns the cross-entropy loss from logits."""
        is_labeled = ~torch.isnan(y_true)
        y_pred_logits = y_pred_logits[is_labeled]
        y_true = y_true[is_labeled]
        if isinstance(self._loss_fn, torch.nn.BCEWithLogitsLoss):
            y_pred_logits = y_pred_logits.float()
            y_true = y_true.float()
        y_true = torch.reshape(y_true, y_pred_logits.shape)
        loss = self._loss_fn(y_pred_logits, y_true)

        if loss.numel() == 0:
            return torch.tensor(0.0, device=y_true.device)
        return loss.mean()

    @staticmethod
    def _attribute_consistency_loss(
        x_abs_cross_translation_im: torch.Tensor, expected_za: torch.Tensor, attr_enc: nn.Module
    ) -> float:
        """Computes the attribute consistency loss.

        Args:
            x_abs_cross_translation_im: The (absorbance) translated images that we used for augmentations.
            expected_za: The expected attribute tensor.
            attr_enc: The attribute encoder.

        Returns:
            A floating point value for the loss.
        """
        za_recon = attr_enc(x_abs_cross_translation_im)
        return torch.nn.L1Loss(reduction="mean")(za_recon, expected_za)

    @staticmethod
    def _content_consistency_loss(
        x_abs_cross_translation_im: torch.Tensor, expected_zc: torch.Tensor, cont_enc: nn.Module
    ) -> float:
        """Computes the attribute consistency loss.

        Args:
            x_abs_cross_translation_im: The (absorbance) translated images that we used for augmentations.
            expected_zc: The expcted content tensor.
            cont_enc: The content encoder.

        Returns:
            A floating point value for the loss.
        """
        zc_recon = cont_enc(x_abs_cross_translation_im)
        return torch.nn.L1Loss(reduction="mean")(zc_recon, expected_zc)

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
