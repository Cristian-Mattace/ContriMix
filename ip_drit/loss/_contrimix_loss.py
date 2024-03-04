"""A module that defines the loss class for the contrimix."""
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
from ip_drit.algorithms._contrimix_utils import ContrimixTrainingMode
from ip_drit.patch_transform import CutMixJointTensorTransform
from ip_drit.patch_transform import ZeroMeanUnitStdNormalizer
from ip_drit.visualization import visualize_content_channels


class ContriMixAggregationType(Enum):
    """An enum class that defines how we should aggregates over the mixings."""

    MAX = auto()
    MEAN = auto()
    AUGREG = auto()


class ContriMixLoss(MultiTaskMetric):
    """A class that defines a Contrimix loss in absorbance space.

    Args:
        d_out: The dimension of the model output.
        loss_weights_by_name: A dictionary of loss weights for Contrimix terms only, excluding the cross-
            entropy term. The sum of the weights must be equal to 1.0.
        loss_params: A dictionary that defines the parameters for the loss. It has the following fields
            -loss_fn: A function to compute the loss from the label, excluding other ContriMix specific loss.
            - normalize_signals_into_to_backbone: If True, the input signal into the backbone will be normalized.
            - aggregation (optional): The type of the aggregation on the backbone loss. Defaults to
                ContriMixAggregationType.MEAN.
            - use_cut_mix (optional): If True, the CutMix transform will be used before computing the CE loss.
                Defaults to False.
            - cut_mix_alpha (optional): The hyper-parameter of cutmix. Defaults to 1.0.
            - use_original_image_for_entropy_loss (optional): If True, the original image will be use for the entropy
                loss.
        name (optional): The name of the loss. Defaults to None, in which case, the default name of "contrimix_loss"
            will be used.
        save_images_for_debugging (optional): If True, different mixing images will be save for debugging. Defaults to
            False.
        aug_reg_variance_weight (optional): The factor that is used to penalize the variance of the model performance.
            Defaults to 10.0.
        max_cross_entropy_loss_weight (optional): The maximum values for the cross-entropy loss weight. Defaults to 0.2.
    """

    def __init__(
        self,
        d_out: int,
        loss_weights_by_name: Dict[str, float],
        loss_params: Dict[str, Any],
        name: Optional[str] = "contrimix_loss",
        save_images_for_debugging: bool = True,
        aug_reg_variance_weight: float = 10.0,
    ) -> None:
        self._loss_fn = loss_params["loss_fn"]
        self._ddp_local_rank = loss_params.get("ddp_local_rank", None)
        self._loss_weights_by_name = loss_weights_by_name
        self._save_images_for_debugging = save_images_for_debugging
        self._debug_image: Optional[np.ndarray] = None
        self._aug_reg_variance_weight: float = aug_reg_variance_weight
        self._backbone_input_normalizer = (
            ZeroMeanUnitStdNormalizer() if loss_params.get("normalize_signals_into_to_backbone", False) else None
        )
        self._aggregation: ContriMixAggregationType = loss_params.get("aggregation", ContriMixAggregationType.MEAN)
        self._training_mode: ContrimixTrainingMode = loss_params["training_mode"]
        super().__init__(name)
        self._epoch: int = 0
        self._cut_mix_transform = (
            CutMixJointTensorTransform(
                num_classes=d_out, alpha=loss_params.get("cut_mix_alpha", 1.0), exclude_num_first_samples=None
            )
            if loss_params.get("use_cut_mix", False)
            else None
        )
        self._use_original_image_for_entropy_loss: bool = loss_params.get("use_original_image_for_entropy_loss", False)

    def compute_self_reconstruction_im(self, im_gen: nn.Module, za: torch.Tensor, zc: torch.Tensor) -> torch.Tensor:
        return im_gen(zc, za)

    def compute(
        self, in_dict: Dict[str, torch.Tensor], return_dict: bool = True, return_loss_components: bool = True
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
        self._is_training = in_dict["is_training"]
        self._abs_to_trans_cvt = in_dict["abs_to_trans_cvt"]
        self._trans_to_abs_cvt = in_dict["trans_to_abs_cvt"]
        self._batch_transform = getattr(in_dict, "batch_transform", None)

        all_target_image_indices = in_dict["all_target_image_indices"]
        cont_enc = in_dict["cont_enc"]
        attr_enc = in_dict["attr_enc"]
        im_gen = in_dict["im_gen"]

        if in_dict["x_org"] is not None:  # labeled data only!
            x_org_1 = in_dict["x_org"]
        else:
            x_org_1 = in_dict["unlabeled_x_org"]

        x_org_2 = self._one_to_two_conversion(im_1=x_org_1)
        zc = cont_enc(x_org_2)
        za = attr_enc(x_org_2)
        x_self_recon_2 = im_gen(zc, za)
        x_self_recon_1 = self._two_to_one_conversion(im_2=x_self_recon_2)
        za_targets = self._generate_z_targets(za=za, all_mix_target_im_idxs=all_target_image_indices)

        if in_dict["y_true"] is not None:
            entropy_loss = self._compute_backbone_loss(
                in_dict=in_dict, x_self_recon=x_self_recon_1, zc=zc, za_targets=za_targets
            )

        if self._save_images_for_debugging and return_loss_components and self._ddp_local_rank == 0:
            self._visualize_contrimix_res(
                x_org_1=x_org_1,
                in_dict=in_dict,
                x_self_recon=x_self_recon_1,
                zc=zc,
                za_targets=za_targets,
                return_loss_components=return_loss_components,
            )

        if self._training_mode == ContrimixTrainingMode.ENCODERS:
            loss_dict_by_name = self._compute_partial_contrimix_loss(in_dict=in_dict)
            all_loss_weights = self._loss_weights_by_name
            total_loss = (
                all_loss_weights["self_recon_weight"] * loss_dict_by_name["self_recon_loss"]
                + all_loss_weights["attr_similarity_weight"] * loss_dict_by_name["attr_similarity_loss"]
                + all_loss_weights["attr_cons_weight"] * loss_dict_by_name["attr_cons_loss"]
                + all_loss_weights["cont_cons_weight"] * loss_dict_by_name["cont_cons_loss"]
            )

            if return_dict:
                if return_loss_components:
                    return {
                        self.agg_metric_field: total_loss,
                        "self_recon_loss": loss_dict_by_name["self_recon_loss"].item(),
                        "attr_similarity_loss": loss_dict_by_name["attr_similarity_loss"].item(),
                        "attribute_consistency_loss": loss_dict_by_name["attr_cons_loss"].item(),
                        "content_consistency_loss": loss_dict_by_name["cont_cons_loss"].item(),
                    }
                else:
                    return {self.agg_metric_field: total_loss}
            else:
                # Used for updating the objective.
                return total_loss

        elif self._training_mode == ContrimixTrainingMode.BACKBONE:
            total_loss = entropy_loss
            if return_dict:
                if return_loss_components:
                    return {self.agg_metric_field: total_loss, "entropy_loss": entropy_loss.item()}
                else:
                    return {self.agg_metric_field: total_loss}
            else:
                return total_loss

        elif self._training_mode == ContrimixTrainingMode.JOINTLY:
            loss_dict_by_name = self._compute_partial_contrimix_loss(in_dict=in_dict)
            all_loss_weights = self._loss_weights_by_name
            total_loss = (
                all_loss_weights["self_recon_weight"] * loss_dict_by_name["self_recon_loss"]
                + all_loss_weights["attr_similarity_weight"] * loss_dict_by_name["attr_similarity_loss"]
                + all_loss_weights["attr_cons_weight"] * loss_dict_by_name["attr_cons_loss"]
                + all_loss_weights["cont_cons_weight"] * loss_dict_by_name["cont_cons_loss"]
                + all_loss_weights["entropy_weight"] * entropy_loss
            )

            if return_dict:
                if return_loss_components:
                    return {
                        self.agg_metric_field: total_loss,
                        "entropy_loss": entropy_loss.item(),
                        "self_recon_loss": loss_dict_by_name["self_recon_loss"].item(),
                        "attr_similarity_loss": loss_dict_by_name["attr_similarity_loss"].item(),
                        "attribute_consistency_loss": loss_dict_by_name["attr_cons_loss"].item(),
                        "content_consistency_loss": loss_dict_by_name["cont_cons_loss"].item(),
                    }
                else:
                    return {self.agg_metric_field: total_loss}
            else:
                return total_loss
        else:
            raise ValueError(f"Contrimix training mode of {self._training_mode} is not supported!")

    def _compute_partial_contrimix_loss(self, in_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self._abs_to_trans_cvt = in_dict["abs_to_trans_cvt"]
        self._trans_to_abs_cvt = in_dict["trans_to_abs_cvt"]
        self._batch_transform = getattr(in_dict, "batch_transform", None)

        all_target_image_indices = in_dict["all_target_image_indices"]
        cont_enc = in_dict["cont_enc"]
        attr_enc = in_dict["attr_enc"]
        im_gen = in_dict["im_gen"]

        x_org_1 = in_dict["x_org"]
        if x_org_1 is not None:
            x_org_2 = self._one_to_two_conversion(im_1=x_org_1)
            zc = cont_enc(x_org_2)
            za = attr_enc(x_org_2)
            x_self_recon_2 = im_gen(zc, za)
            x_self_recon_1 = self._two_to_one_conversion(im_2=x_self_recon_2)
            za_targets = self._generate_z_targets(za=za, all_mix_target_im_idxs=all_target_image_indices)
            self_recon_losses = [self._self_recon_consistency_loss(self_recon_ims=x_self_recon_1, expected_ims=x_org_1)]

        # The index 2 is to tell what space the original image is going to, which can be abs, or trans.
        unlabeled_x_org_1 = in_dict["unlabeled_x_org"]
        if unlabeled_x_org_1 is not None:
            unlabeled_x_org_2 = self._one_to_two_conversion(im_1=unlabeled_x_org_1)
            unlabeled_zc = cont_enc(unlabeled_x_org_2)
            unlabeled_za = attr_enc(unlabeled_x_org_2)
            unlabeled_x_self_recon_2 = im_gen(unlabeled_zc, unlabeled_za)
            unlabeld_x_self_recon_1 = self._two_to_one_conversion(im_2=unlabeled_x_self_recon_2)
            unlabeled_za_targets = self._generate_z_targets(
                za=unlabeled_za, all_mix_target_im_idxs=all_target_image_indices
            )
            self_recon_losses = [
                self._self_recon_consistency_loss(
                    self_recon_ims=unlabeld_x_self_recon_1, expected_ims=unlabeled_x_org_1
                )
            ]

        self_recon_loss = torch.max(torch.stack(self_recon_losses, dim=0))

        if x_org_1 is not None:
            attr_similarity_loss = self._attribute_similarity_loss(za=za, za_targets=za_targets)
            combined_ims_for_attr_cont_loss_calc = torch.cat(
                [x_self_recon_2, *[im_gen(zc, za_target) for za_target in za_targets]], dim=0
            )

            attr_cons_loss = self._attribute_consistency_loss(
                x_cross_translation_ims=combined_ims_for_attr_cont_loss_calc,
                expected_zas=torch.cat([za, za_targets.reshape(-1, za_targets.size(2), za_targets.size(3))], dim=0),
                attr_enc=in_dict["attr_enc"],
            )

            cont_cons_loss = self._content_consistency_loss(
                x_cross_translation_ims=combined_ims_for_attr_cont_loss_calc,
                expected_zc=zc,
                cont_enc=in_dict["cont_enc"],
                num_ims=za_targets.size(0) + 1,
            )

        if unlabeled_x_org_1 is not None:
            attr_similarity_loss = self._attribute_similarity_loss(za=unlabeled_za, za_targets=unlabeled_za_targets)
            combined_ims_for_attr_cont_loss_calc = torch.cat(
                [
                    unlabeled_x_self_recon_2,
                    *[im_gen(unlabeled_zc, unlabeled_za_target) for unlabeled_za_target in unlabeled_za_targets],
                ],
                dim=0,
            )

            attr_cons_loss = self._attribute_consistency_loss(
                x_cross_translation_ims=combined_ims_for_attr_cont_loss_calc,
                expected_zas=torch.cat(
                    [
                        unlabeled_za,
                        unlabeled_za_targets.reshape(-1, unlabeled_za_targets.size(2), unlabeled_za_targets.size(3)),
                    ],
                    dim=0,
                ),
                attr_enc=in_dict["attr_enc"],
            )

            cont_cons_loss = self._content_consistency_loss(
                x_cross_translation_ims=combined_ims_for_attr_cont_loss_calc,
                expected_zc=unlabeled_zc,
                cont_enc=in_dict["cont_enc"],
                num_ims=unlabeled_za_targets.size(0) + 1,
            )

        return {
            "self_recon_loss": self_recon_loss,
            "attr_similarity_loss": attr_similarity_loss,
            "attr_cons_loss": attr_cons_loss,
            "cont_cons_loss": cont_cons_loss,
        }

    def _one_to_two_conversion(self, im_1: torch.Tensor) -> torch.Tensor:
        return self._trans_to_abs_cvt(im_trans=im_1) if self._trans_to_abs_cvt is not None else im_1

    def _two_to_one_conversion(self, im_2: torch.Tensor) -> torch.Tensor:
        return self._clip_tensor_range(
            self._abs_to_trans_cvt(im_abs=im_2) if self._abs_to_trans_cvt is not None else im_2
        )

    @staticmethod
    def _clip_tensor_range(x: torch.Tensor) -> torch.Tensor:
        return torch.clip(x, min=0.0, max=1.0)

    @staticmethod
    def _generate_z_targets(za: torch.Tensor, all_mix_target_im_idxs: torch.Tensor) -> Optional[torch.Tensor]:
        """Generates a tensor that contains the target attributes to mix.

        Args:
            za: The (content/attribute) attributes from the labeled images.
            all_mix_target_im_idxs: A 2D tensor of size minibatch size x # mixings that contains the indices to use.
        """
        num_mixings = all_mix_target_im_idxs.size(1)
        if num_mixings > 0:
            za_targets: List[torch.Tensor] = []
            for mix_idx in range(num_mixings):
                target_im_idxs = all_mix_target_im_idxs[:, mix_idx]
                za_targets.append(za[target_im_idxs])
            return torch.stack(za_targets, dim=0)
        return None

    @staticmethod
    def _self_recon_consistency_loss(self_recon_ims: torch.Tensor, expected_ims: torch.Tensor) -> float:
        # return torch.nn.L1Loss(reduction="mean")(self_recon_ims, expected_ims)
        # Use 90 pct of the error to avoid the noise affecting the max.
        diff = torch.abs(self_recon_ims - expected_ims)
        diff_reshaped = diff.view((diff.size(0), -1))
        return 0.2 * torch.quantile(diff_reshaped, q=0.90, dim=1) + 0.8 * torch.nn.L1Loss(reduction="mean")(
            self_recon_ims, expected_ims
        )

    @staticmethod
    def _attribute_consistency_loss(
        x_cross_translation_ims: torch.Tensor, expected_zas: torch.Tensor, attr_enc: nn.Module
    ) -> float:
        """Computes the attribute similarity loss.

        Args:
            x_cross_translation_ims: The cross translation images that we used for augmentation.
            expected_zas: The expected attribute tensor.
            attr_enc: The attribute encoder.

        Returns:
            A floating point value for the loss.
        """
        zas_recon = attr_enc(x_cross_translation_ims)
        return torch.nn.L1Loss(reduction="mean")(zas_recon, expected_zas)

    @staticmethod
    def _content_consistency_loss(
        x_cross_translation_ims: torch.Tensor, expected_zc: torch.Tensor, cont_enc: nn.Module, num_ims: int
    ) -> float:
        """Computes the content consistency loss.

        Args:
            x_cross_translation: The cross translation images that we used for augmentation. This is from all images.
            expected_zc: The expected content tensor. This is for 1 image.
            cont_enc: The content encoder.
            num_ims: The number of images in x_cross_translation.

        Returns:
            A floating point value for the loss.
        """
        zc_recons = cont_enc(x_cross_translation_ims).reshape(num_ims, -1, *expected_zc.shape[1:])
        return torch.nn.L1Loss(reduction="mean")(zc_recons, expected_zc.unsqueeze(0).repeat(num_ims, 1, 1, 1, 1))

    @staticmethod
    def _attribute_similarity_loss(za: torch.Tensor, za_targets: torch.Tensor) -> torch.Tensor:
        """Computes the attribute similarity loss to minimize the similarity between different zas used for a single im.

        Args:
            za: The attribute tensor of shape (#samples, #attr dims, #attrs) extracted from different patches. We would
                like to maximize the distance between these tensors.
            za_targets: The target attributes of shape (#augs, #samples, #attr dims, #attrs) of the target attributes.
        """
        _TEMPERATURE_PARAM = 0.1
        za_expanded = za.unsqueeze(0)
        all_zas = torch.cat([za_expanded, za_targets], dim=0)  # (augs+1, #samples, #attr dims, #attrs)
        all_zas = all_zas.permute((1, 0, 2, 3))  # (#samples, augs+1, #attr dims, #attrs)
        all_zas_reshaped = all_zas.reshape(*all_zas.shape[:2], -1)  # (#samples, augs+1, #attr dims * #attrs)
        # Don't want to penalize the high correlation due to the common mode.
        dist_matrix = F.pairwise_distance(all_zas_reshaped[:, :, None, :], all_zas_reshaped[:, None, :, :])
        # Mask the diagonal elements
        diag_mask = torch.eye(dist_matrix.shape[1], dtype=torch.bool, device=dist_matrix.device).expand(
            [dist_matrix.size(0), -1, -1]
        )
        return torch.exp(-dist_matrix / _TEMPERATURE_PARAM).masked_fill(diag_mask, 0).mean()

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
        else:
            backbone_inputs = x_self_recon
            backbone_inputs_extended = [x_self_recon]

        num_total_ims_to_backbone = 1

        # Add synthetic images from the mixing on training only.
        if self._is_training:
            if za_targets is not None:
                num_total_ims_to_backbone = za_targets.size(0) + 1
                backbone_inputs_extended.extend(
                    [self._two_to_one_conversion(im_2=in_dict["im_gen"](zc, za_target)) for za_target in za_targets]
                )

        if len(y_true.shape) != 1:
            raise ValueError("y_true is not a class index vector. Can't proceed!")

        y_true_extended = y_true.repeat(num_total_ims_to_backbone)
        backbone_inputs_extended = torch.cat(backbone_inputs_extended, dim=0)

        if self._is_training and self._cut_mix_transform is not None:
            backbone_inputs_extended, y_true_extended = self._cut_mix_transform(
                backbone_inputs_extended, y_true_extended
            )

        # Compute the prediction on the training set.

        backbone_inputs_extended = self._process_backbone_input(backbone_inputs_extended)
        backbone_inputs = self._process_backbone_input(backbone_inputs)

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

    def _visualize_contrimix_res(
        self,
        x_org_1: torch.Tensor,
        in_dict: Dict[str, torch.Tensor],
        x_self_recon: torch.Tensor,
        za_targets: torch.Tensor,
        zc: torch.Tensor,
        save_debug_im: Optional[bool] = True,
        return_loss_components: bool = False,
    ) -> None:
        # Visualizing images
        if self._save_images_for_debugging and return_loss_components:
            im_gen = in_dict["im_gen"]
            im_idx_to_visualize = 5
            all_target_image_indices = in_dict["all_target_image_indices"]
            save_images: List[np.ndarray] = [
                x_self_recon[im_idx_to_visualize].clone().detach().cpu().numpy().transpose(1, 2, 0)
            ]
            self_recon_images: List[np.ndarray] = [
                x_self_recon[im_idx_to_visualize].clone().detach().cpu().numpy().transpose(1, 2, 0)
            ]
            target_images: List[np.ndarray] = [
                x_org_1[im_idx_to_visualize].clone().detach().cpu().numpy().transpose(1, 2, 0)
            ]

            if za_targets is not None:
                for mix_idx, za_target in enumerate(za_targets):
                    x_cross_translation_2 = im_gen(zc, za_target)
                    x_cross_translation_1 = self._two_to_one_conversion(im_2=x_cross_translation_2)

                    target_index = all_target_image_indices[:, mix_idx][0]
                    target_images.append(x_org_1[target_index].clone().detach().cpu().numpy().transpose(1, 2, 0))
                    self_recon_images.append(
                        x_self_recon[target_index].clone().detach().cpu().numpy().transpose(1, 2, 0)
                    )
                    save_images.append(
                        x_cross_translation_1[im_idx_to_visualize].clone().detach().cpu().numpy().transpose(1, 2, 0)
                    )

            target_image = np.concatenate(target_images, axis=1)
            self_recon_image = np.concatenate(self_recon_images, axis=1)
            augmented_image = np.concatenate(save_images, axis=1)
            self._debug_image = np.clip(
                np.concatenate([target_image, self_recon_image, augmented_image], axis=0), a_min=0.0, a_max=1.0
            )
            if save_debug_im:
                io.imsave("debug_image.png", (self.debug_image * 255.0).astype(np.uint8))

            if in_dict["y_true"] is not None:
                visualize_content_channels(org_ims=x_org_1, zcs=zc)

    def _validate_batch_transform(self, batch_transform: Callable) -> None:
        if isinstance(batch_transform, CutMixJointTensorTransform) and (
            self._aggregation == ContriMixAggregationType.AUGREG
        ):
            raise ValueError("Can't use CutMix with AugReg. Turn off one of them!")

    def _process_backbone_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_training:
            # Only apply random rotations, adding noise during training.
            if self._batch_transform is not None:
                return self._batch_transform(self._normalize_signal_into_backbone(x))
        return self._normalize_signal_into_backbone(x)

    def _normalize_signal_into_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if self._backbone_input_normalizer is not None:
            x = self._backbone_input_normalizer(x)
        return x
