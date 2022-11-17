import itertools
import logging
import os
from collections import defaultdict
from tempfile import TemporaryDirectory
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from ._utils import load_trained_model_from_checkpoint
from ip_drit.datamodule import MultiDomainDataset
from ip_drit.loss import ImageConsistencyLossType
from ip_drit.loss import L1ImageConsistencyLoss
from ip_drit.loss import Loss
from ip_drit.loss import LPipsImageConsistencyLoss
from ip_drit.models import AbsorbanceImGenerator
from ip_drit.models import AbsorbanceImGeneratorWithConvTranspose
from ip_drit.models import AbsorbanceImGeneratorWithInterpolation
from ip_drit.models import AttributeEncoder
from ip_drit.models import ContentEncoder
from ip_drit.models import GeneratorType
from ip_drit.models import RealFakeDiscriminator
from ip_drit.patch_transform import AbsorbanceToTransmittance

class MutliClassTrainingModule(pl.LightningModule):
    """A class that defines the training module for multi-class image translation.

    Args:
        num_input_channels: The number of the input channels.
        num_stain_vectors: The number of stain vector.
        num_attribute_enc_out_chans: The number of output channels for the attribute encoder.
        train_hyperparams: A dictionary that defines different hyperparameters for the training.
        test_dataset (optional): The dataset that contains images that is used to visualize the
            performance of the network over epochs of the training. Defaults to None.
        generator_type (optional): The type of the generator to use. Defaults to GeneratorType.WithPixelShuffle.
        image_consistency_loss_type (optional): The type of the image consistency loss. Defaults to
            ImageConsistencyLossType.L1ImageConsistencyLoss.
    """

    _RANDOM_SEED = 1
    _NUM_DISC_UPDATE_PER_ITERATION = 1
    _NUM_PERMUTATION_FOR_FAKE_IMAGES = 2

    def __init__(
        self,
        num_input_channels: int,
        num_stain_vectors: int,
        num_attribute_enc_out_chans: int,
        train_hyperparams: Dict[str, Any],
        test_dataset: Optional[MultiDomainDataset] = True,
        generator_type: GeneratorType = GeneratorType.WithPixelShuffle,
        image_consistency_loss_type: ImageConsistencyLossType = ImageConsistencyLossType.L1ImageConsistencyLoss,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._num_input_channels: int = num_input_channels
        self._train_hyperparams = train_hyperparams
        self._number_gen_optimization_steps_to_update_disc = train_hyperparams[
            "number_gen_optimization_steps_to_update_disc"
        ]

        if self._number_gen_optimization_steps_to_update_disc > 1:
            logging.info(
                f"Please make sure that you have at least {self._number_gen_optimization_steps_to_update_disc} "
                + " minibatches so that the discrimonator are updated!"
            )
        self._enc_c = ContentEncoder(in_channels=3, num_stain_vectors=num_stain_vectors)
        self._enc_a = AttributeEncoder(
            in_channels=3, out_channels=num_attribute_enc_out_chans, num_stain_vectors=num_stain_vectors
        )

        self._gen = {
            GeneratorType.WithPixelShuffle: AbsorbanceImGenerator(downsampling_factor=4),
            GeneratorType.WithInterpolation: AbsorbanceImGeneratorWithInterpolation(
                in_channels=num_attribute_enc_out_chans, downsampling_factor=4
            ),
            GeneratorType.WithConvTranspose2d: AbsorbanceImGeneratorWithConvTranspose(
                in_channels=num_attribute_enc_out_chans
            ),
        }[generator_type]

        if image_consistency_loss_type == ImageConsistencyLossType.L1ImageConsistencyLoss:
            self._image_consistency_loss = L1ImageConsistencyLoss()
        elif image_consistency_loss_type == ImageConsistencyLossType.LPipsImageConsistencyLoss:
            self._image_consistency_loss = LPipsImageConsistencyLoss()
        else:
            raise ValueError(f"Unknow image_consistency_loss_type ({image_consistency_loss_type})!")

        self._recon_weight = train_hyperparams["loss_weights_by_name"]["recon_weight"]

        self._loss = Loss(**train_hyperparams["loss_weights_by_name"])

        self._encoders_gen_params = itertools.chain(
            [*self._enc_c.parameters(), *self._enc_a.parameters(), *self._gen.parameters()]
        )
        self._disc1 = RealFakeDiscriminator(in_channels=num_input_channels)

        if train_hyperparams["pretrained_model_path"] is not None:
            self._enc_c = load_trained_model_from_checkpoint(
                train_hyperparams["pretrained_model_path"], network=self._enc_c, starts_str="_enc_c."
            )
            self._enc_a = load_trained_model_from_checkpoint(
                train_hyperparams["pretrained_model_path"], network=self._enc_a, starts_str="_enc_a."
            )
            self._gen = load_trained_model_from_checkpoint(
                train_hyperparams["pretrained_model_path"], network=self._gen, starts_str="_gen."
            )
            self._disc1 = load_trained_model_from_checkpoint(
                train_hyperparams["pretrained_model_path"], network=self._disc1, starts_str="_disc1."
            )
        self._discs_params = self._disc1.parameters()

        self._test_dataset = test_dataset
        if test_dataset is not None:
            self._temp_save_folder: str = TemporaryDirectory()
            logging.info(f"Training results will be saved to {self._temp_save_folder.name}")

    def training_step(self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int) -> Dict[str, torch.Tensor]:
        outputs = self._compute_network_forward_outputs(batch)
        if optimizer_idx == 0:
            self._set_requires_gradients([self._enc_c, self._enc_a, self._gen], requires_grad=True)
            self._set_requires_gradients([self._disc1], requires_grad=False)
            loss_dict = self._loss.compute_generator_and_encoder_losses(
                forward_outputs=outputs,
                real_vs_cross_translation_disc=self._disc1,
                num_permutations_for_fake_images=MutliClassTrainingModule._NUM_PERMUTATION_FOR_FAKE_IMAGES,
            )

            # We have to compute the image consistency loss here to avoid parameters of LPIPS modules exists on 1 GPU
            # while the images exists on another GPUs. It is dirty not to be able to move the LPIPs loss into the Loss()
            # module. Why?
            image_consistency_loss = self._compute_image_consistency_loss(
                real_ims=outputs["real_ims"], target_ims=outputs["self_recon_ims"]
            )
            loss_dict["loss"] = loss_dict["loss"] + self._recon_weight * image_consistency_loss
            loss_dict["self_recon_loss"] = image_consistency_loss
            return loss_dict

        if optimizer_idx == 1:
            self._set_requires_gradients([self._enc_c, self._enc_a, self._gen], requires_grad=False)
            self._set_requires_gradients([self._disc1], requires_grad=True)
            return self._loss.compute_real_fake_discriminator_losses(
                forward_outputs=outputs,
                real_vs_cross_translation_disc=self._disc1,
                num_permutations_for_fake_images=MutliClassTrainingModule._NUM_PERMUTATION_FOR_FAKE_IMAGES,
            )

    def _compute_network_forward_outputs(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        minibatch_size = batch.shape[0]
        self._ensure_minibatch_size_is_even(minibatch_size)
        real_ims = batch
        z_cs = self._enc_c(real_ims)
        z_as = self._enc_a(real_ims)

        num_permutations = MutliClassTrainingModule._NUM_PERMUTATION_FOR_FAKE_IMAGES
        swapped_zc = self._generate_permutations(z_cs, num_permutations=num_permutations)
        # Cross-translation image generation.
        input_z_cs = torch.cat([z_cs, swapped_zc], dim=0)
        input_z_as = torch.cat([z_as] * (num_permutations + 1), dim=0)

        out_fakes = self._gen(input_z_cs, input_z_as)
        self_recon_ims, fake_one_time_cross_translation_ims = torch.split(
            out_fakes, [minibatch_size, minibatch_size * num_permutations], dim=0
        )

        z_c_from_recon_ims, z_c_from_fake_ims = torch.split(
            self._enc_c(out_fakes), [minibatch_size, minibatch_size * num_permutations], dim=0
        )

        z_a_from_recon_ims, z_a_from_fake_ims = torch.split(
            self._enc_a(out_fakes), [minibatch_size, minibatch_size * num_permutations], dim=0
        )
        return {
            "real_ims": real_ims,
            "z_cs": z_cs,
            "swapped_zc": swapped_zc,
            "z_c_from_fake_ims": z_c_from_fake_ims,
            "z_c_from_recon_ims": z_c_from_recon_ims,
            "self_recon_ims": self_recon_ims,
            "fake_one_time_cross_translation_ims": fake_one_time_cross_translation_ims,
            "z_as": z_as,
            "z_a_from_recon_ims": z_a_from_recon_ims,
            "z_a_from_fake_ims": z_a_from_fake_ims,
        }

    @staticmethod
    def _generate_permutations(minibatch_samples: torch.Tensor, num_permutations: int = 1) -> torch.Tensor:
        num_samples = minibatch_samples.size(0)
        all_permutations: List[torch.Tensor] = []
        for _ in range(num_permutations):
            sample_permutation = torch.randperm(num_samples, device=minibatch_samples.device)
            sample_permutation = sample_permutation.view(num_samples, 1, 1, 1)
            sample_permutation = sample_permutation.repeat(
                1, minibatch_samples.size(1), minibatch_samples.size(2), minibatch_samples.size(3)
            )
            all_permutations.append(torch.gather(minibatch_samples, 0, sample_permutation))
        return torch.cat(all_permutations, dim=0)

    @staticmethod
    def _sample_attribute_vectors_from_gaussian_distribution(
        mus: torch.Tensor, logvars: torch.Tensor, over_sampling_factor: int = 1
    ) -> torch.Tensor:
        std = logvars.mul(0.5).exp()
        z = MutliClassTrainingModule._sample_normal_distrbution(
            num_samples=mus.size(0) * over_sampling_factor, attr_dim=mus.size(1), target_device=mus.device
        )
        z = torch.randn(mus.shape, device=mus.device)
        z = z.mul(std).add(mus)
        return z.view(z.size(0), z.size(1), 1, 1)

    def _compute_image_consistency_loss(self, real_ims: torch.Tensor, target_ims: torch.Tensor) -> float:
        """Computes the  consistency loss between the real images and the cycle reconstructed images.

        Args:
            real_ims: The real images.
            target_ims: The target images to compare to.

        """
        return self._image_consistency_loss(real_ims, target_ims)

    @staticmethod
    def _aggregate_losses_from_generator_and_discrimonator_loss_dicts(
        gen_loss_dict: Dict[str, torch.Tensor], disc_loss_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Returns a new dictionary of loss terms which is the combination of losses from two loss dictionaries."""
        losses_by_name: Dict[str, torch.Tensor] = {
            "gen_loss": gen_loss_dict["loss"],
            "disc_loss": disc_loss_dict["loss"],
        }

        for loss_dict in [gen_loss_dict, disc_loss_dict]:
            for k, v in loss_dict.items():
                if k != "loss":
                    losses_by_name[k] = v
        if set(gen_loss_dict.keys()).intersection(disc_loss_dict.keys()) != {"loss", "batch_size"}:
            raise ValueError("Generator loss dictionary and discriminator loss dictionary have overlapping names!")
        return losses_by_name

    def training_step_end(self, workers_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Aggregates the results from the training steps across multiple workers in the same batch.

        This is required for gradient descent to work. Otherwise, we would get "RuntimeError: grad can be implicitly
        created only for scalar outputs" because the loss is not a scalar (i.e., tensor with one element); instead it
        would be a tensor with size equal to the number of workers.

        Args:
            workers_outputs: A dictionary that contains the outputs from multiple workers.
        """
        return {k: v.sum() for k, v in workers_outputs.items()}

    def training_epoch_end(self, batch_outputs: List[List[Dict[str, float]]]) -> None:
        """Combines the loss from all batches.

        Args:
            batch_outputs: A list of list in which one item is for one optimizer idx. Each item in the inner
                lists is a Dictionary with key equals to 'loss'.
        """
        self._set_random_seed()
        loss_dict = defaultdict(list)

        for batch_output in batch_outputs:
            # Generator and encoders
            for k, value in batch_output[0].items():
                if k == "loss":
                    k = "generator_and_encoder_loss"
                loss_dict[k].append(value)

            # Discriminator losses.
            for k, value in batch_output[1].items():
                if k == "loss":
                    k = "train_disc_total_loss"
                loss_dict[k].append(value)

        for k, values in loss_dict.items():
            loss_dict[k] = torch.stack(values).sum()

        total_num_samples = loss_dict["batch_size"]
        avg_generator_and_encoder_loss = loss_dict["generator_and_encoder_loss"] / total_num_samples
        avg_real_fake_adv_loss_cross_trans_ims_with_real_vs_cross_trans_disc = (
            loss_dict["real_fake_adv_loss_cross_trans_ims_with_real_vs_cross_trans_disc"] / total_num_samples
        )
        avg_self_recon_loss = loss_dict["self_recon_loss"] / total_num_samples
        avg_cont_consistency_loss = loss_dict["cont_consistency_loss"] / total_num_samples
        avg_attribute_consistency_loss = loss_dict["attribute_consistency_loss"] / total_num_samples
        avg_mode_seeking_loss = loss_dict["mode_seeking_loss"] / total_num_samples
        avg_content_channel_covariance_loss = loss_dict["content_channel_covariance_loss"] / total_num_samples

        avg_train_disc_total_loss = loss_dict["train_disc_total_loss"] / total_num_samples
        avg_real_fake_loss_real_ims_with_real_vs_cross_trans_disc = (
            loss_dict["real_fake_loss_real_ims_with_real_vs_cross_trans_disc"] / total_num_samples
        )
        avg_real_fake_loss_cross_trans_ims_with_real_vs_cross_trans_disc = (
            loss_dict["real_fake_loss_cross_trans_ims_with_real_vs_cross_trans_disc"] / total_num_samples
        )

        logging.info(
            f"\n-> Epoch {self.current_epoch}: train_encoders_generators_total_loss: "
            + f" {avg_generator_and_encoder_loss:.3f}, \n        BCE[D_1(cross_tran), "
            + f"'real'): {avg_real_fake_adv_loss_cross_trans_ims_with_real_vs_cross_trans_disc:.3f}"
            + "  -> Expected value: -log(0.5) = 0.6931."
            f"\n        self_recon_loss: {avg_self_recon_loss:.3f} / "
            + f"cont_consistency_loss: {avg_cont_consistency_loss:.3f} / "
            + f"attribute_consistency_loss: {avg_attribute_consistency_loss:.3f} / "
            + f"mode_seeking_loss: {avg_mode_seeking_loss:.3f} / "
            + f"content_channel_covariance_loss: {avg_content_channel_covariance_loss}"
        )

        logging.info(
            f"-> train_disc_total_loss: {avg_train_disc_total_loss:.3f}"
            + f"\n     BCE[D_1(real_image), 'real'): {avg_real_fake_loss_real_ims_with_real_vs_cross_trans_disc:.3f} / "
            + f"BCEL[D_1(cross_trans), 'fake']: {avg_real_fake_loss_cross_trans_ims_with_real_vs_cross_trans_disc:.3f}"
            + " -> Expected value: -log(0.5) = 0.6931."
        )

        self.log("train_encoders_generators_total_loss", avg_generator_and_encoder_loss)
        self.log("train_disc_total_loss", avg_train_disc_total_loss)
        if self._test_dataset is not None and self.current_epoch % 3 == 0:
            self._generate_inference_results()

    @staticmethod
    def _set_random_seed():
        # Make sure that random is the same in the begining of each optimization epoch.
        torch.manual_seed(MutliClassTrainingModule._RANDOM_SEED)
        torch.cuda.manual_seed(MutliClassTrainingModule._RANDOM_SEED)

    def _generate_inference_results(self) -> None:
        """Generates the inference results for debugging."""
        tensor_im_0 = self._test_dataset.get_item_with_domain_idx(sample_idx=0, domain_idx=0)
        tensor_im_1 = self._test_dataset.get_item_with_domain_idx(sample_idx=0, domain_idx=1)
        im = tensor_im_0.numpy().transpose(1, 2, 0)
        gpu_device = torch.device("cuda:0")
        tensor_im_0 = tensor_im_0[None, ...].to(gpu_device)
        tensor_im_1 = tensor_im_1[None, ...].to(gpu_device)

        with torch.no_grad():
            z_c = self._enc_c.to(gpu_device)(tensor_im_0)
            z_a_0 = self._enc_a.to(gpu_device)(tensor_im_0)
            z_a_1 = self._enc_a.to(gpu_device)(tensor_im_1)

            recon_im = self._gen.to(gpu_device)(z_c, z_a_0)
            cross_recon_im = self._gen.to(gpu_device)(z_c, z_a_1)

        num_total_images = 4
        num_rows, num_cols, num_chans = im.shape
        out_ims = np.zeros((num_rows, num_total_images * num_cols, num_chans), dtype=np.float)
        out_ims[:, :num_cols] = 10 ** (-im)
        out_ims[:, num_cols : 2 * num_cols] = 10 ** (-recon_im.cpu().numpy()[0].transpose(1, 2, 0))
        out_ims[:, 2 * num_cols : 3 * num_cols] = 10 ** (-tensor_im_1[0].cpu().numpy().transpose(1, 2, 0))
        out_ims[:, 3 * num_cols :] = 10 ** (-cross_recon_im.cpu().numpy()[0].transpose(1, 2, 0))
        out_ims = (np.clip(out_ims, 0.0, 1.0) * 255.0).astype(np.uint8)
        image_name = os.path.join(self._temp_save_folder.name, f"training_res_epoch_{self.current_epoch}.png")
        cv2.imwrite(image_name, out_ims)
        logging.info(f"Save debug image {image_name}")

    @staticmethod
    def _set_requires_gradients(networks: List[nn.Module], requires_grad: bool) -> None:
        """Sets the status for gradient calculation for the networks.

        Args:
            networks: A list of neural networks to set the gradient.
            requires_grad: If False, gradient will not be calculated, otherwise, do gradient calc.
        """
        for net in networks:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def configure_optimizers(self):
        # Optimizer configuration
        # See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        encoders_generator_opt = torch.optim.SGD(
            self._encoders_gen_params, lr=self._train_hyperparams["gen_learning_rate"], weight_decay=0.0001
        )
        discs_opt = torch.optim.SGD(
            self._discs_params, lr=self._train_hyperparams["disc_learning_rate"], weight_decay=0.0001
        )
        return {
            "optimizer": encoders_generator_opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(encoders_generator_opt, 0.95, verbose=True),
                "interval": "epoch",
                "frequency": self._train_hyperparams["number_of_steps_to_update_lr"],
            },
        }, {
            "optimizer": discs_opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(discs_opt, 0.95, verbose=True),
                "interval": "epoch",
                "frequency": self._train_hyperparams["number_of_steps_to_update_lr"],
            },
        }

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ) -> None:
        # See https://pytorch-lightning.readthedocs.io/en/latest/common/optimization.html on how to se this correctly.
        if optimizer_idx == 0:
            if (batch_idx + 1) % self._number_gen_optimization_steps_to_update_disc == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()

        # Update the discriminator for each iteration
        if optimizer_idx == 1:
            optimizer.step(closure=optimizer_closure)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, float]:
        outputs = self._compute_network_forward_outputs(batch)
        loss_dict = self._loss.compute_generator_and_encoder_losses(
            forward_outputs=outputs,
            real_vs_cross_translation_disc=self._disc1,
            num_permutations_for_fake_images=MutliClassTrainingModule._NUM_PERMUTATION_FOR_FAKE_IMAGES,
        )

        # We have to compute the image consistency loss here to avoid parameters in the LPIPS modules exists on 1 GPU
        # while the images exists on another GPUs. It is dirty not to be able to move the LPIPs loss into the Loss()
        # module. Why?
        image_consistency_loss = self._compute_image_consistency_loss(
            real_ims=outputs["real_ims"], target_ims=outputs["self_recon_ims"]
        )
        loss_dict["loss"] = loss_dict["loss"] + self._recon_weight * image_consistency_loss
        loss_dict["self_recon_loss"] = image_consistency_loss
        return loss_dict

    @staticmethod
    def _ensure_minibatch_size_is_even(minibatch_size: int) -> None:
        if not minibatch_size % 2 == 0:
            raise ValueError(
                f"The size of the minibatch must be even! The requested minibatch size is {minibatch_size}."
            )

    @staticmethod
    def _sample_normal_distrbution(num_samples: int, attr_dim: int, target_device: torch.device) -> torch.Tensor:
        return torch.randn((num_samples, attr_dim, 1, 1), device=target_device)

    def validation_step_end(self, workers_outputs: Dict[str, torch.Tensor]):
        """Combines the loss from all workers."""
        return {k: v.sum() for k, v in workers_outputs.items()}

    def validation_epoch_end(self, batch_outputs: List[Dict[str, float]]):
        """Combines the loss from all batches.

        Args:
            batch_outputs: A list in which each item is a Dictionary that contains information from each batch.
        """
        self._set_random_seed()
        total_num_samples = torch.stack([b["batch_size"] for b in batch_outputs]).sum()
        val_avg_generator_and_encoder_loss = torch.stack([b["loss"] for b in batch_outputs]).sum() / total_num_samples

        avg_real_fake_adv_loss_cross_trans_ims_with_real_vs_cross_trans_disc = (
            torch.stack(
                [b["real_fake_adv_loss_cross_trans_ims_with_real_vs_cross_trans_disc"] for b in batch_outputs]
            ).sum()
            / total_num_samples
        )

        avg_self_recon_loss = torch.stack([b["self_recon_loss"] for b in batch_outputs]).sum() / total_num_samples
        avg_cont_consistency_loss = (
            torch.stack([b["cont_consistency_loss"] for b in batch_outputs]).sum() / total_num_samples
        )
        avg_attribute_consistency_loss = (
            torch.stack([b["attribute_consistency_loss"] for b in batch_outputs]).sum() / total_num_samples
        )
        avg_mode_seeking_loss = torch.stack([b["mode_seeking_loss"] for b in batch_outputs]).sum() / total_num_samples
        avg_content_channel_covariance_loss = (
            torch.stack([b["content_channel_covariance_loss"] for b in batch_outputs]).sum() / total_num_samples
        )

        logging.info(
            f"\n-> Val: train_encoders_generators_total_loss: {val_avg_generator_and_encoder_loss:.3f}, "
            + f"\n        real_fake_adv_loss_cross_trans_ims_with_real_vs_cross_trans_disc: "
            + f"{avg_real_fake_adv_loss_cross_trans_ims_with_real_vs_cross_trans_disc:.3f}"
            + f"\n        self_recon_loss: {avg_self_recon_loss:.3f} / cont_consistency_loss: "
            + f"{avg_cont_consistency_loss:.3f} / attribute_consistency_loss: "
            + f"{avg_attribute_consistency_loss:.3f} / mode_seeking_loss: {avg_mode_seeking_loss:.3f}. "
            + f"/ content_channel_covariance_loss: {avg_content_channel_covariance_loss:.3f}"
        )
        self.log("val_encoders_generators_total_loss", val_avg_generator_and_encoder_loss)
