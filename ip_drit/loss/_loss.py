from typing import Dict
from typing import List

import torch
import torch.nn as nn

from ._consistency_loss import AbstractImageConsistencyLoss


class Loss:
    """A class that defines the loss for the DRIT.

    Args:
        real_fake_weight (optional): The weight for penalizing a real vs. fake image. Defaults to 1.0.
        recon_weight (optional): The weight for the reconstruction losses. Defaults to 10.0.
        content_consistency_weight (optional): The weight for L2 content consistency loss. Defaults to 1.0.
        attr_consistency_weight (optional): The weight for the latent regression loss. Defaults to 10.0.
        mode_seeking_loss_weight (optional): The weight for the mode seeking loss. Defaults to 1.0.
        content_channel_covariance_loss_weight (optional): The weight for the l1 norm of the content covariance loss.
            Defaults to 1.0.
    """

    def __init__(
        self,
        real_fake_weight: float = 1.0,
        recon_weight: float = 1.0,
        content_consistency_weight: float = 1.0,
        attr_consistency_weight: float = 10.0,
        mode_seeking_loss_weight: float = 1.0,
        content_channel_covariance_loss_weight: float = 1.0,
    ) -> None:
        self._real_fake_weight: float = real_fake_weight
        self._recon_weight: float = recon_weight
        self._content_consistency_weight: float = content_consistency_weight
        self._attr_consistency_weight: float = attr_consistency_weight
        self._mode_seeking_loss_weight: float = mode_seeking_loss_weight
        self._content_channel_covariance_loss_weight: float = content_channel_covariance_loss_weight

    def compute_generator_and_encoder_losses(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        real_vs_cross_translation_disc: nn.Module,
        num_permutations_for_fake_images: int,
    ) -> Dict[str, torch.Tensor]:
        """Computes a dictionary of the losses, keyed by the loss types.

        Args:
            forward_outputs: A dictionary of the output tensors from the forward pass, keyed by the name of the outputs.
            real_vs_cross_translation_disc: The discriminator that discrimates between the real and the fake
                cross-translation images.
            num_permutations_for_fake_images: The number of permutation used to generate fake images.
        """
        fake_one_time_ims_pred_logits_with_real_vs_cross_trans_disc = real_vs_cross_translation_disc.forward(
            forward_outputs["fake_one_time_cross_translation_ims"]
        )
        rf_adv_loss_cross_trans_ims = (
            self._real_fake_loss_for_images(
                predicted_logits=fake_one_time_ims_pred_logits_with_real_vs_cross_trans_disc,
                target_disc_label=1,  # Ref: https://github.com/HsinYingLee/MDMM/blob/master/model.py#L292.
            )
            / num_permutations_for_fake_images
        )

        cont_consistency_loss = (
            self._latent_regression_loss(forward_outputs["swapped_zc"], forward_outputs["z_c_from_fake_ims"])
            + self._latent_regression_loss(forward_outputs["z_cs"], forward_outputs["z_c_from_recon_ims"])
        ) / (num_permutations_for_fake_images + 1)

        attribute_consistency_loss = (
            self._latent_regression_loss(
                forward_outputs["z_as"].repeat(num_permutations_for_fake_images, 1, 1),
                forward_outputs["z_a_from_fake_ims"],
            )
            + self._latent_regression_loss(forward_outputs["z_as"], forward_outputs["z_a_from_recon_ims"])
        ) / (num_permutations_for_fake_images + 1)

        mode_seeking_loss = self._mode_seeking_regularization_loss(
            zas=forward_outputs["z_as"], ims=forward_outputs["real_ims"]
        )

        content_channel_covariance_loss = self._content_channel_covariance_loss(zcs=forward_outputs["z_cs"])
        total_loss = (
            self._real_fake_weight * rf_adv_loss_cross_trans_ims
            + self._content_consistency_weight * cont_consistency_loss
            + self._attr_consistency_weight * attribute_consistency_loss
            + self._mode_seeking_loss_weight * mode_seeking_loss
            + self._content_channel_covariance_loss_weight * content_channel_covariance_loss
        )

        return {
            "loss": total_loss,
            "real_fake_adv_loss_cross_trans_ims_with_real_vs_cross_trans_disc": rf_adv_loss_cross_trans_ims,
            "cont_consistency_loss": cont_consistency_loss,
            "attribute_consistency_loss": attribute_consistency_loss,
            "mode_seeking_loss": mode_seeking_loss,
            "content_channel_covariance_loss": content_channel_covariance_loss,
            "batch_size": torch.tensor(
                forward_outputs["real_ims"].size(0), device=content_channel_covariance_loss.device
            ),
        }

    @staticmethod
    def _latent_regression_loss(z1: torch.Tensor, z2: torch.Tensor) -> float:
        """Computes the L1loss between the two tensor z1 and z2."""
        return nn.L1Loss(reduction="mean")(z1, z2) * z1.size(1)

    @staticmethod
    def _mode_seeking_regularization_loss(zas: torch.Tensor, ims: torch.Tensor) -> float:
        half_batch_size = zas.size(0) // 2
        za_dist = nn.L1Loss(reduction="sum")(zas[:half_batch_size], zas[half_batch_size:]) / zas.size(1)
        im_dist = nn.L1Loss(reduction="sum")(ims[:half_batch_size], ims[half_batch_size:]) / (
            ims.size(1) * ims.size(2) * ims.size(3)
        )
        return im_dist / (za_dist + 1e-9)

    @staticmethod
    def _content_channel_covariance_loss(zcs: torch.Tensor) -> float:
        """Returns the covvariance loss, which is the sum of the off-diagonal entries of the covariance matrices.

        Reference:
            VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning
        """
        zcs = zcs.view(zcs.size(0), zcs.size(1), -1)

        # Remove the mean of each channel.
        zc_diff = zcs - zcs.mean(dim=2, keepdim=True)  # #samples x # channels # chanel
        num_samples = zc_diff.size(0)
        zc_diff_t = torch.transpose(zc_diff, 1, 2)
        cov_mat = torch.bmm(zc_diff, zc_diff_t).mean(dim=0) / (num_samples - 1)
        num_bands = cov_mat.size(0)
        select_mask = 1.0 - torch.eye(num_bands, device=cov_mat.device)

        # Use the square loss to penalize large correlation more.
        loss_val = ((cov_mat * select_mask).pow_(2).sum() / (num_bands * (num_bands - 1))).sqrt()
        return loss_val

    def compute_real_fake_discriminator_losses(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        real_vs_cross_translation_disc: nn.Module,
        num_permutations_for_fake_images: int,
    ) -> Dict[str, torch.Tensor]:
        """Computes a dictionary of the losses for the real vs. fake discriminators, keyed by the loss types.

        Args:
            forward_outp_vs_cross_translation_disc: The discriminator that discrimates between the real and the fake
                cross-translation images.
            num_permutations_for_fake_images: The number of permutation used to generate fake images.
        """
        pred_logits_real_vs_cross_trans_disc = real_vs_cross_translation_disc.forward(
            torch.cat(
                [forward_outputs["real_ims"].detach(), forward_outputs["fake_one_time_cross_translation_ims"].detach()],
                dim=0,
            )
        )

        num_real_samples = forward_outputs["real_ims"].size(0)

        rf_loss_real_ims_with_real_vs_cross_trans_disc = self._real_fake_loss_for_images(
            predicted_logits=pred_logits_real_vs_cross_trans_disc[:num_real_samples], target_disc_label=1
        )

        rf_loss_cross_trans_ims_real_vs_cross_disc = 0
        for fake_image_batch_idx in range(num_permutations_for_fake_images):
            rf_loss_cross_trans_ims_real_vs_cross_disc += self._real_fake_loss_for_images(
                predicted_logits=pred_logits_real_vs_cross_trans_disc[
                    num_real_samples * (fake_image_batch_idx + 1) : num_real_samples * (fake_image_batch_idx + 2)
                ],
                target_disc_label=0,
            )
        # We need to account for the fake that we have more fake images.
        rf_loss_cross_trans_ims_real_vs_cross_disc = (
            rf_loss_cross_trans_ims_real_vs_cross_disc / num_permutations_for_fake_images
        )

        total_loss = rf_loss_real_ims_with_real_vs_cross_trans_disc + rf_loss_cross_trans_ims_real_vs_cross_disc
        return {
            "loss": total_loss,
            "real_fake_loss_real_ims_with_real_vs_cross_trans_disc": rf_loss_real_ims_with_real_vs_cross_trans_disc,
            "real_fake_loss_cross_trans_ims_with_real_vs_cross_trans_disc": rf_loss_cross_trans_ims_real_vs_cross_disc,
        }

    @staticmethod
    def _real_fake_loss_for_images(predicted_logits: torch.Tensor, target_disc_label: int) -> float:
        """Computes the adversarial loss when the discriminator tries to discrimate between the real and fake images.

        The loss is given as -{Sum_over_real_images [log(sigmoid(disc(real)))] +
            Sum_over_fake_images{[log(1 - sigmoid(disc(real)))]}
        The loss is per-pixel averaged and summed over instances in the mini-batch.

        Args:
            predicted_logits: The outputs logits predicted from a discriminatot to tell if an image is real or fake.
            target_disc_label: The target label that we want the output of the discriminator to be. This can be 0 for
                Fake and 1 for real image.
        """
        if target_disc_label == 0:
            target_labels = torch.zeros_like(predicted_logits, device=predicted_logits.device)
        else:
            target_labels = torch.ones_like(predicted_logits, device=predicted_logits.device)
        return nn.BCEWithLogitsLoss(reduction="sum")(predicted_logits, target_labels) / (
            predicted_logits.size(2) * predicted_logits.size(3)
        )
