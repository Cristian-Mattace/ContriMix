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
from ip_drit.models import SignalType
from ip_drit.patch_transform import CutMixJointTensorTransform
from ip_drit.patch_transform import ZeroMeanUnitStdNormalizer


class ContriMixAggregationType(Enum):
    """An enum class that defines how we should aggregates over the mixings."""

    MEAN = auto()
    AUGREG = auto()


class ContriMixLoss(MultiTaskMetric):
    """A class that defines a Contrimix loss in absorbance space.

    Args:
        loss_fn: A function to compute the loss from the label, excluding other ContriMix specific loss.
        loss_weights_by_name: A dictionary of loss weights for Contrimix terms only, excluding the cross-
            entropy term. The sum of the weights must be equal to 1.0.
        name (optional): The name of the loss. Defaults to None, in which case, the default name of "contrimix_loss"
            will be used.
        save_images_for_debugging (optional): If True, different mixing images will be save for debugging. Defaults to
            False.
        aggregation (optional): The type of the aggregation on the backbone loss. Defaults to
            ContriMixAggregationType.MEAN.
        aug_reg_variance_weight (optional): The factor that is used to penalize the variance of the model performance.
            Defaults to 10.0.
        weight_ramp_up_steps (int): The number of steps that the weights of cross-entropy should ramp up. Defaults to 1.
        max_cross_entropy_loss_weight (float): The maximum values for the cross-entropy loss weight. Defaults to 0.2.
        use_cut_mix (optional): If True, the CutMix transform will be used before computing the cross-entropy loss.
        normalize_signals_into_to_backbone (bool): If True, the input signal into the backbone will be normalized.
            Defaults to True.
    """

    def __init__(
        self,
        loss_fn: Optional[Callable],
        loss_weights_by_name: Dict[str, float],
        name: Optional[str] = "contrimix_loss",
        save_images_for_debugging: bool = False,
        aggregation: ContriMixAggregationType = ContriMixAggregationType.MEAN,
        aug_reg_variance_weight: float = 10.0,
        weight_ramp_up_steps: int = 1,
        use_cut_mix: bool = True,
        normalize_signals_into_to_backbone: bool = True,
    ) -> None:
        self._loss_fn = loss_fn
        self._loss_weights_by_name = self._clean_up_loss_weight_dictionary(loss_weights_by_name)
        self._save_images_for_debugging = save_images_for_debugging
        self._debug_image: Optional[np.ndarray] = None
        self._aug_reg_variance_weight: float = aug_reg_variance_weight
        self._backbone_input_normalizer = ZeroMeanUnitStdNormalizer() if normalize_signals_into_to_backbone else None
        self._aggregation: ContriMixAggregationType = aggregation
        super().__init__(name)
        self._weight_ramp_up_steps: float = weight_ramp_up_steps
        self._epoch: int = 0
        self._cut_mix_transform = CutMixJointTensorTransform() if use_cut_mix else None

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
        self._abs_to_trans_cvt = in_dict["abs_to_trans_cvt"]
        self._trans_to_abs_cvt = in_dict["trans_to_abs_cvt"]
        self._batch_transform = in_dict["batch_transform"]
        self._is_training = in_dict["is_training"]
        self._validate_batch_transform(batch_transform=self._batch_transform)

        x_org_1 = in_dict["x_org"]
        unlabeled_x_org_1 = in_dict["unlabeled_x_org"]
        all_target_image_indices = in_dict["all_target_image_indices"]
        cont_enc = in_dict["cont_enc"]
        attr_enc = in_dict["attr_enc"]
        im_gen = in_dict["im_gen"]
        is_training = in_dict["is_training"]

        # The index 2 is to tell what space the original image is going to, which can be absorbance, or transmittance.
        x_org_2 = self._one_to_two_conversion(im_1=x_org_1)
        zc = cont_enc(x_org_2)
        za = attr_enc(x_org_2)
        x_self_recon_2 = im_gen(zc, za)
        x_self_recon_1 = self._two_to_one_conversion(im_2=x_self_recon_2)

        # We should not evaluate the error in the absorbance space because it does not uniformly across the range.
        self_recon_losses = [self._self_recon_consistency_loss(self_recon_ims=x_self_recon_1, expected_ims=x_org_1)]

        if unlabeled_x_org_1 is not None:
            unlabeled_x_org_2 = self._one_to_two_conversion(im_1=unlabeled_x_org_1)
            unlabeled_za = attr_enc(unlabeled_x_org_2)
            self_recon_losses.append(
                self._self_recon_consistency_loss(
                    self_recon_ims=self._two_to_one_conversion(im_2=im_gen(cont_enc(unlabeled_x_org_2), unlabeled_za)),
                    expected_ims=unlabeled_x_org_1,
                )
            )
        else:
            unlabeled_za = None

        self_recon_loss = torch.mean(torch.stack(self_recon_losses, dim=0))

        za_targets = self._generate_za_targets(za=za, unlabeled_za=za, all_mix_target_im_idxs=all_target_image_indices)

        in_dict["y_pred"] = backbone(self._compute_backbone_input(x_self_recon_1))

        num_mixings = za_targets.shape[0]

        entropy_losses = [
            self._compute_entropy_loss_from_logits(
                self._compute_backbone_input(x_self_recon_1), y_true, backbone=backbone
            )
        ]

        attr_cons_losses: List[float] = [
            self._attribute_consistency_loss(
                x_cross_translation=x_self_recon_2, expected_za=za, attr_enc=in_dict["attr_enc"]
            )
        ]

        cont_cons_losses: List[float] = [
            self._content_consistency_loss(
                x_cross_translation=x_self_recon_2, expected_zc=zc, cont_enc=in_dict["cont_enc"]
            )
        ]

        if self._save_images_for_debugging:
            save_images: List[np.ndarray] = [x_self_recon_1[0].clone().detach().cpu().numpy().transpose(1, 2, 0)]
            target_images: List[np.ndarray] = [x_org_1[0].clone().detach().cpu().numpy().transpose(1, 2, 0)]

        for mix_idx in range(num_mixings):
            za_target = za_targets[mix_idx]
            x_cross_translation_2 = im_gen(zc, za_target)
            x_cross_translation_1 = self._two_to_one_conversion(im_2=x_cross_translation_2)

            if self._save_images_for_debugging:
                target_index = all_target_image_indices[:, mix_idx][0]
                target_images.append(x_org_1[target_index].clone().detach().cpu().numpy().transpose(1, 2, 0))
                save_images.append(x_cross_translation_1[0].clone().detach().cpu().numpy().transpose(1, 2, 0))

            entropy_losses.append(
                self._compute_entropy_loss_from_logits(
                    self._compute_backbone_input(x_cross_translation_1), y_true, backbone=backbone
                )
            )

            attr_cons_losses.append(
                self._attribute_consistency_loss(
                    x_cross_translation=x_cross_translation_2, expected_za=za_target, attr_enc=in_dict["attr_enc"]
                )
            )

            cont_cons_losses.append(
                self._content_consistency_loss(
                    x_cross_translation=x_cross_translation_2, expected_zc=zc, cont_enc=in_dict["cont_enc"]
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

        all_loss_weights = self._loss_weights_by_name
        if is_training:
            all_loss_weights = self._loss_weights_from_epoch()
        else:
            all_loss_weights = self._loss_weights_by_name

        total_loss = (
            all_loss_weights["self_recon_weight"] * self_recon_loss
            + all_loss_weights["entropy_weight"] * entropy_loss
            + all_loss_weights["attr_cons_weight"] * attr_cons_loss
            + all_loss_weights["cont_cons_weight"] * cont_cons_loss
        )

        if self._save_images_for_debugging:
            target_image = np.concatenate(target_images, axis=1)
            augmented_image = np.concatenate(save_images, axis=1)
            self._debug_image = np.clip(np.concatenate([target_image, augmented_image], axis=0), a_min=0.0, a_max=1.0)

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

    def _one_to_two_conversion(self, im_1: torch.Tensor) -> torch.Tensor:
        return self._trans_to_abs_cvt(im_trans=im_1) if self._trans_to_abs_cvt is not None else im_1

    def _two_to_one_conversion(self, im_2: torch.Tensor) -> torch.Tensor:
        return self._abs_to_trans_cvt(im_abs=im_2) if self._abs_to_trans_cvt is not None else im_2

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

    def _compute_entropy_loss_from_logits(
        self, x: torch.Tensor, y_true: torch.Tensor, backbone: nn.Module
    ) -> torch.Tensor:
        """Returns the cross-entropy loss from logits (not averaging over the minibatch samples)."""
        if self._cut_mix_transform is not None and self._is_training:
            x, y_true = self._cut_mix_transform(x, y_true)

        y_pred_logits = backbone(x)
        if isinstance(self._loss_fn, torch.nn.BCEWithLogitsLoss):
            y_pred_logits = y_pred_logits.float()
            y_true = y_true.float()
        loss = self._loss_fn(y_pred_logits, y_true)

        if loss.numel() == 0:
            return torch.tensor(0.0, device=y_true.device)
        return loss

    @staticmethod
    def _attribute_consistency_loss(
        x_cross_translation: torch.Tensor, expected_za: torch.Tensor, attr_enc: nn.Module
    ) -> float:
        """Computes the attribute consistency loss.

        Args:
            x_cross_translation: The cross translation images that we used for augmentations.
            expected_za: The expected attribute tensor.
            attr_enc: The attribute encoder.

        Returns:
            A floating point value for the loss.
        """
        za_recon = attr_enc(x_cross_translation)
        return torch.nn.L1Loss(reduction="mean")(za_recon, expected_za)

    @staticmethod
    def _content_consistency_loss(
        x_cross_translation: torch.Tensor, expected_zc: torch.Tensor, cont_enc: nn.Module
    ) -> float:
        """Computes the attribute consistency loss.

        Args:
            x_cross_translation: The cross translation images that we used for augmentations.
            expected_zc: The expcted content tensor.
            cont_enc: The content encoder.

        Returns:
            A floating point value for the loss.
        """
        zc_recon = cont_enc(x_cross_translation)
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

    def update_epoch_index(self, epoch: int) -> None:
        """Updates the current epoch index."""
        self._epoch = epoch

    def _loss_weights_from_epoch(self) -> Dict[str, float]:
        if self._epoch >= self._weight_ramp_up_steps:
            return self._loss_weights_by_name
        cur_cross_entropy_weight = (
            self._epoch / self._weight_ramp_up_steps * self._loss_weights_by_name["entropy_weight"]
        )
        total_contrix_weights = (
            self._loss_weights_by_name["self_recon_weight"]
            + self._loss_weights_by_name["attr_cons_weight"]
            + self._loss_weights_by_name["cont_cons_weight"]
        )
        target_total_contrimix_weights = 1.0 - cur_cross_entropy_weight
        weight_dict = {"entropy_weight": cur_cross_entropy_weight}
        weight_dict.update(
            {
                k: self._loss_weights_by_name[k] * (target_total_contrimix_weights / total_contrix_weights)
                for k in ("self_recon_weight", "attr_cons_weight", "cont_cons_weight")
            }
        )

        return weight_dict

    def _validate_batch_transform(self, batch_transform: Callable) -> None:
        if isinstance(batch_transform, CutMixJointTensorTransform) and (
            self._aggregation == ContriMixAggregationType.AUGREG
        ):
            raise ValueError("Can't use CutMix with AugReg. Turn off one of them!")

    def _compute_backbone_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_training:
            # Only apply random rotations, adding noise during training.
            if self._batch_transform is not None:
                return self._batch_transform(self._normalize_signal_into_backbone(x))
        return self._normalize_signal_into_backbone(x)

    def _normalize_signal_into_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if self._backbone_input_normalizer is None:
            return x
        return self._backbone_input_normalizer(x)
