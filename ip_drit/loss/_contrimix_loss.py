"""A module that defines the loss class for the contrimix."""
import logging
from enum import auto
from enum import Enum
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from ..common.metrics._base import Metric
from ..common.metrics._base import MultiTaskMetric
from ..common.metrics._utils import numel
from ip_drit.models import SignalType


class ContriMixAggregationType(Enum):
    """An enum class that defines how we should aggregates over the mixings."""

    MEAN = auto()
    AUGREG = auto()


class ContriMixLoss(MultiTaskMetric):
    """A class that defines a multi-task loss.

    Args:
        loss_fn: A function to compute the loss from the label, excluding other ContriMix specific loss.
        contrimix_loss_weights_by_name: A dictionary of loss weights for Contrimix terms only, excluding the cross-
            entropy term. The sum of the weights must be equal to 1.0.
        name (optional): The name of the loss. Defaults to None, in which case, the default name of "contrimix_loss"
            will be used.
        save_images_for_debugging (optional): If True, different mixing images will be save for debugging. Defaults to
            False.
        aggregation (optional): The type of the aggregation on the backbone loss. Defaults to
            ContriMixAggregationType.AUGREG, in which augmentation regularization will be used for image consistency.
        aug_reg_variance_weight (optional): The factor that is used to penalize the variance of the model performance.
            Defaults to 10.0.
        weight_decay_rate (float): The rate that the contrixmix weights should be decay. Defaults to 0.95.
        max_cross_entropy_loss_weight (float): The maximum values for the cross-entropy loss weight. Defaults to 0.2.
    """

    def __init__(
        self,
        loss_fn: Optional[Callable],
        contrimix_loss_weights_by_name: Dict[str, float],
        name: Optional[str] = "contrimix_loss",
        save_images_for_debugging: bool = False,
        aggregation: ContriMixAggregationType = ContriMixAggregationType.AUGREG,
        aug_reg_variance_weight: float = 10.0,
        weight_decay_rate: float = 0.95,
        max_cross_entropy_loss_weight: float = 0.2,
    ) -> None:
        self._loss_fn = loss_fn
        self._contrimix_loss_weights_by_name = self._clean_up_loss_weight_dictionary(contrimix_loss_weights_by_name)
        self._save_images_for_debugging = save_images_for_debugging
        self._debug_image: Optional[np.ndarray] = None
        self._aug_reg_variance_weight: float = aug_reg_variance_weight

        self._aggregation: ContriMixAggregationType = aggregation
        super().__init__(name)
        self._weight_decay_rate: float = weight_decay_rate
        self._max_epoch_to_update_weight: int = int(
            np.log10(1 - max_cross_entropy_loss_weight) / np.log10(self._weight_decay_rate)
        )
        self._epoch: int = 0

    @staticmethod
    def _clean_up_loss_weight_dictionary(loss_weights_by_name: Dict[str, float]) -> Dict[str, float]:
        total_loss_weights = sum(loss_weights_by_name.values())
        if any(x < 0 for x in loss_weights_by_name.values()):
            raise ValueError("All the weights must be non-negative!")

        if total_loss_weights > 1.0:
            raise ValueError("The total weights for all the loss can't be larger than 1")
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
        backbone = in_dict["backbone"]
        abs_to_trans_cvt = in_dict["abs_to_trans_cvt"]
        trans_to_abs_cvt = in_dict["trans_to_abs_cvt"]

        x_org = in_dict["x_org"]
        unlabeled_x_org = in_dict["unlabeled_x_org"]
        all_target_image_indices = in_dict["all_target_image_indices"]
        cont_enc = in_dict["cont_enc"]
        attr_enc = in_dict["attr_enc"]
        im_gen = in_dict["im_gen"]

        x_abs_org = trans_to_abs_cvt(im_and_sig_type=(x_org, SignalType.TRANS))[0]
        zc = cont_enc(x_abs_org)
        za = attr_enc(x_abs_org)
        x_abs_self_recon = im_gen(zc, za)
        x_self_recon = abs_to_trans_cvt(im_and_sig_type=(x_abs_self_recon, SignalType.ABS))[0]

        # We should not evaluate the error in the absorbance space because it does not uniformly across the range.
        self_recon_losses = [self._self_recon_consistency_loss(self_recon_ims=x_self_recon, expected_ims=x_org)]

        if unlabeled_x_org is not None:
            unlabeled_x_abs_org = trans_to_abs_cvt(im_and_sig_type=(unlabeled_x_org, SignalType.TRANS))[0]
            unlabeled_za = attr_enc(unlabeled_x_abs_org)
            self_recon_losses.append(
                self._self_recon_consistency_loss(
                    self_recon_ims=abs_to_trans_cvt(
                        im_and_sig_type=(im_gen(cont_enc(unlabeled_x_abs_org), unlabeled_za), SignalType.ABS)
                    )[0],
                    expected_ims=unlabeled_x_org,
                )
            )
        else:
            unlabeled_za = None

        self_recon_loss = torch.mean(torch.stack(self_recon_losses, dim=0))

        za_targets = self._generate_za_targets(za=za, unlabeled_za=za, all_mix_target_im_idxs=all_target_image_indices)

        y_pred = backbone(x_self_recon)
        in_dict["y_pred"] = y_pred

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

        if self._save_images_for_debugging:
            save_images: List[np.ndarray] = [x_self_recon[0].clone().detach().cpu().numpy().transpose(1, 2, 0)]
            target_images: List[np.ndarray] = [x_org[0].clone().detach().cpu().numpy().transpose(1, 2, 0)]

        for mix_idx in range(num_mixings):
            za_target = za_targets[mix_idx]
            x_abs_cross_translation = im_gen(zc, za_target)
            x_cross_translation = abs_to_trans_cvt(im_and_sig_type=(x_abs_cross_translation, SignalType.ABS))[0]

            if self._save_images_for_debugging:
                target_index = all_target_image_indices[:, mix_idx][0]
                target_images.append(x_org[target_index].clone().detach().cpu().numpy().transpose(1, 2, 0))
                save_images.append(x_cross_translation[0].clone().detach().cpu().numpy().transpose(1, 2, 0))

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

        entropy_losses = torch.stack(entropy_losses, dim=0)
        entropy_loss_mean_avg = torch.mean(entropy_losses)
        if self._aggregation == ContriMixAggregationType.MEAN:
            entropy_loss = entropy_loss_mean_avg
        elif self._aggregation == ContriMixAggregationType.AUGREG:
            entropy_loss = entropy_loss_mean_avg + self._aug_reg_variance_weight * torch.mean(
                torch.var(entropy_losses, dim=0)
            )
        else:
            raise ValueError(f"Aggregation type of {self._aggregation} is not supported!")

        total_loss = (
            self._all_loss_weights_by_name["self_recon_weight"] * self_recon_loss
            + self._all_loss_weights_by_name["entropy_weight"] * entropy_loss
            + self._all_loss_weights_by_name["attr_cons_weight"] * attr_cons_loss
            + self._all_loss_weights_by_name["cont_cons_weight"] * cont_cons_loss
        )

        if self._save_images_for_debugging:
            target_image = np.concatenate(target_images, axis=1)
            augmented_image = np.concatenate(save_images, axis=1)
            self._debug_image = np.concatenate([target_image, augmented_image], axis=0)

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
    def _generate_za_targets(
        za: torch.Tensor, unlabeled_za: Optional[torch.Tensor], all_mix_target_im_idxs: torch.Tensor
    ) -> torch.Tensor:
        """Generates a tensor that contains the target attributes to mix.

        Args:
            za: The attributes from the labeled images.
            unlabeled_za: The attributes from the unlabeled images.
            all_mix_target_im_idxs: A 2D tensor of size minibatch size x # mixings that contains the indices to use.
        """
        za_targets: List[torch.Tensor] = []
        for mix_idx in range(all_mix_target_im_idxs.size(1)):
            target_im_idxs = all_mix_target_im_idxs[:, mix_idx]
            if unlabeled_za is None:
                za_targets.append(za[target_im_idxs])
            else:
                # Either borrow from the label or unlabel dataset
                if float(torch.rand(1)) > 0.5:
                    za_targets.append(unlabeled_za[target_im_idxs])
                else:
                    za_targets.append(za[target_im_idxs])
        return torch.stack(za_targets, dim=0)

    @staticmethod
    def _self_recon_consistency_loss(self_recon_ims: torch.Tensor, expected_ims: torch.Tensor) -> float:
        return torch.nn.L1Loss(reduction="mean")(self_recon_ims, expected_ims)

    def _compute_entropy_loss_from_logits(self, y_pred_logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Returns the cross-entropy loss from logits (not averaging over the minibatch samples)."""
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
        return loss

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

    @property
    def debug_image(self) -> np.ndarray:
        if self._debug_image is None:
            raise RuntimeError(f"Debug image was not saved, set save_images_for_debugging to True to get it")
        return self._debug_image

    def update_contrimix_loss_weights_for_current_epoch(self, epoch: int) -> None:
        """Calculates a dictionary of the contrimix loss weights."""
        if epoch > self._max_epoch_to_update_weight:
            return
        lambdaa = self._weight_decay_rate**epoch
        weight_dict = {k: lambdaa * v for k, v in self._contrimix_loss_weights_by_name.items()}
        weight_dict["entropy_weight"] = 1.0 - lambdaa
        logging.info(f"   Contrimix loss weights = {weight_dict}")
        self._all_loss_weights_by_name = weight_dict
