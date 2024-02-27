"""This modules implements the ContriMix Augmentation algorithm."""
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from skimage import io
from torchvision import transforms as tfs

from ._utils import move_to
from .multi_model_algorithm import MultimodelAlgorithm
from ip_drit.algorithms._contrimix_utils import ContriMixMixingType
from ip_drit.algorithms._contrimix_utils import ContrimixTrainingMode
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.metrics import Metric
from ip_drit.loss import ContriMixLoss
from ip_drit.loss import HistauGANLoss
from ip_drit.models import get_non_linearity
from ip_drit.models import InitializationType
from ip_drit.models import Initializer
from ip_drit.models import MD_Dis
from ip_drit.models import MD_Dis_content
from ip_drit.models import MD_E_attr_concat
from ip_drit.models import MD_E_content
from ip_drit.models import MD_G_multi_concat
from ip_drit.models import SignalType
from ip_drit.models.wild_model_initializer import initialize_model_from_configuration
from ip_drit.patch_transform import PostContrimixTransformPipeline


class HistauGAN(MultimodelAlgorithm):
    """A class that implements the HistauGAN algorithm.

    Args:
        config: A dictionary that defines the configuration to load the model.
        d_out: The dimension of the model output.
        grouper: A grouper object that defines the groups for which we compute/log statistics for.
        loss: The loss module.
        metric: The metric to use.
        n_train_steps: The number of training steps.
        convert_to_absorbance_in_between (optional): If True (default), the input image will be converted to absorbance
            before decomposing into content and attribute.
        batch_transform (optional): A module perform batch processing. Defaults to None, in which case, no batch
            processing will be performed.
        num_mixing_per_image (optional): The number of mixing images for each original image. Defaults to 5.
        num_attr_vectors (optional): The number of stain vectors. Defaults to 4
        contrimix_mixing_type (optional): The type of Contrimix mixing. Defaults to ContriMixMixingTyoe.RANDOM.
        local_rank (optional): The local rank used for training with DDP. Defaults to None.
        input_zc_logits (bool): If True, the content encoder will output logits, not activation output.
            The label associated contrimix should have this one set to True. Defaults to False.
        training_mode (optional): If True, only the Contrimix encoder will be trained. Defaults to
            ContrimixTrainingMode.JOINTLY, meaning both the backbone and the encoders will be trained at the same time.
        logged_fields (optional): A list of fields to log. Defaults to None, in which case, the default field names will
            be used.

    Reference:
        https://github.com/sophiajw/HistAuGAN
    """

    _NUM_INPUT_CHANNELS = 3

    def __init__(
        self,
        config: Dict[str, Any],
        d_out: int,
        grouper: AbstractGrouper,
        loss: HistauGANLoss,
        metric: Metric,
        n_train_steps: int,
        algorithm_parameters: Dict[str, Any],
        num_mixing_per_image: int = 5,
        batch_transforms: Optional[PostContrimixTransformPipeline] = None,
        contrimix_mixing_type: ContriMixMixingType = ContriMixMixingType.RANDOM,
        training_mode: ContrimixTrainingMode = ContrimixTrainingMode.JOINTLY,
        logged_fields: Optional[List[str]] = None,
    ) -> None:
        if not isinstance(loss, HistauGANLoss):
            raise ValueError(f"The specified loss module is of type {type(loss)}, not HistauGANLoss!")

        self._model_type = config["model"]
        self._d_out = d_out
        self._nz = algorithm_parameters["nz"]
        self._d_iter = algorithm_parameters[
            "d_iter"
        ]  # The number of epochs to update the discriminator before updating the generator.
        self._concat = True
        self._lambda_cls = algorithm_parameters["lambda_cls"]
        self._lambda_cls_G = algorithm_parameters["lambda_cls_G"]
        self._lambda_rec = algorithm_parameters["lambda_rec"]
        self._saving_freq_iters = 20

        backbone_network = initialize_model_from_configuration(
            self._model_type,
            self._d_out,
            output_classifier=False,
            use_pretrained_backbone=config["model_kwargs"]["pretrained"],
        )

        if logged_fields is None:
            logged_fields = self._update_log_fields_based_on_training_mode(training_mode=training_mode)

        super().__init__(
            config=config,
            models_by_names=self._initialize_histaugan_networks(
                backbone_network=backbone_network, algorithm_options=algorithm_parameters
            ),
            grouper=grouper,
            loss=loss,
            logged_fields=logged_fields,
            metric=metric,
            n_train_steps=n_train_steps,
            batch_transform=batch_transforms,
            training_mode=training_mode,
        )

        self._configure_optimizers(lr=1e-4)
        self._use_unlabeled_y = config["use_unlabeled_y"]
        self._num_mixing_per_image = num_mixing_per_image
        self._use_amp = getattr(config, "use_amp", False)
        self._cls_loss = nn.BCEWithLogitsLoss()
        self._gan_loss = None
        self._gan_cls_loss = None
        self._z_L1_loss = None
        self._l1_self_rec_loss = None
        self._l1_cc_rec_loss = None

    def _configure_optimizers(self, lr: float) -> None:
        """Configures the optimizers.

        Args:
            lr: The learning rate.
        """
        if self._training_mode == ContrimixTrainingMode.ENCODERS:
            self._dis1_opt = torch.optim.Adam(
                self._models_by_names["dis1"].parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001
            )
            self._dis2_opt = torch.optim.Adam(
                self._models_by_names["dis2"].parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001
            )
            self._enc_c_opt = torch.optim.Adam(
                self._models_by_names["enc_c"].parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001
            )
            self._enc_a_opt = torch.optim.Adam(
                self._models_by_names["enc_a"].parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001
            )
            self._gen_opt = torch.optim.Adam(
                self._models_by_names["gen"].parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001
            )
            self._disContent_opt = torch.optim.Adam(
                self._models_by_names["disContent"].parameters(), lr=lr / 2.5, betas=(0.5, 0.999), weight_decay=0.0001
            )
        elif self._training_mode == ContrimixTrainingMode.BACKBONE:
            self._backbone_opt = torch.optim.Adam(
                self._models_by_names["backbone"].parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001
            )

    def update(
        self,
        labeled_batch: Tuple[torch.Tensor, ...],
        unlabeled_batch: Optional[Tuple[torch.Tensor, ...]] = None,
        is_epoch_end: bool = False,
        epoch: Optional[int] = None,
        return_loss_components: bool = True,
        batch_idx: Optional[int] = None,
    ):
        """Process the batch, update the log, and update the model.

        Args:
            labeled_batch: A batch of data yielded by data loaders.
            unlabeled_batch (optional): A batch of data yielded by unlabeled data loader or None.
            is_epoch_end (optional): Whether this batch is the last batch of the epoch. If so, force optimizer to step,
                regardless of whether this batch idx divides self.gradient_accumulation_steps evenly. Defaults to False.
            epoch (optional): The index of the epoch.
            return_loss_components (optional): If True, the component of the loss will be return.
            batch_idx (optional): The index of the current batch. Defaults to None.

        Returns:
            A dictionary of the results, keyed by the field names. There are following fields.
                g: Groups indices for samples in the for batch.
                y_true: Ground truth labels for batch.
                metadata: Metadata for batch.
                y_pred: model output for batch.
                outputs: A tensor for the output.
                objective: The value of the objective.
        """
        if not self._is_training:
            raise RuntimeError("Can't upddate the model parameters because the algorithm is not in the training mode!")

        input_dict = self._parse_inputs(labeled_batch, split="train")

        images = input_dict["x"]
        c_org = input_dict["g"]

        batch_results = {
            "g": c_org,
            "metadata": input_dict["metadata"],
            "y_true": input_dict["y_true"],
            "backbone": self._models_by_names["backbone"],
            "is_training": self._is_training,
            "x_org": images,
            "gen": self._models_by_names["gen"],
            "enc_c": self._models_by_names["enc_c"],
            "enc_a": self._models_by_names["enc_a"],
        }

        if self._training_mode == ContrimixTrainingMode.ENCODERS:
            if (batch_idx + 1) % self._d_iter != 0:
                self._update_D_content(image=images, c_org=c_org)
            else:
                self._update_D(image=images, c_org=c_org)
                self._update_EG(image=images, c_org=c_org)

            batch_results.update(
                {
                    "gan_loss": self._gan_loss,
                    "gan_cls_loss": self._gan_cls_loss,
                    "z_L1_loss": self._z_L1_loss,
                    "l1_self_rec_loss": self._l1_self_rec_loss,
                    "l1_cc_rec_loss": self._l1_cc_rec_loss,
                }
            )

            # Hack to avoid missing the y_pred
            self.objective(batch_results, return_loss_components=return_loss_components)
            if batch_idx > 0 and batch_idx % self._saving_freq_iters == 0:
                self._save_display_image()

        else:
            # Computes the backbone loss.
            self._backbone_opt.zero_grad()
            entropy_loss = self.objective(batch_results, return_loss_components=return_loss_components)[0]
            entropy_loss.backward()
            self._backbone_opt.step()
            batch_results["entropy_loss"] = entropy_loss.detach().item()

        if return_loss_components:
            self.update_log(batch_results)

        if is_epoch_end:
            self._batch_idx = 0
        else:
            self._batch_idx += 1

        return self._sanitize_dict(batch_results)

    @staticmethod
    def _update_log_fields_based_on_training_mode(training_mode: ContrimixTrainingMode):
        log_fields = []
        if training_mode in (ContrimixTrainingMode.BACKBONE, ContrimixTrainingMode.JOINTLY):
            log_fields.append("entropy_loss")
        elif training_mode in (ContrimixTrainingMode.ENCODERS, ContrimixTrainingMode.JOINTLY):
            log_fields.extend(["gan_loss", "gan_cls_loss", "z_L1_loss", "l1_self_rec_loss", "l1_cc_rec_loss"])
        return log_fields

    def _initialize_histaugan_networks(
        self, backbone_network: nn.Module, algorithm_options: Dict[str, Any]
    ) -> Dict[str, nn.Module]:
        """Initialize the HistauGAN networks.

        Reference:
            https://github.com/sophiajw/HistAuGAN/blob/main/histaugan/model.py#L14
        """
        return {
            "backbone": backbone_network,
            "dis1": Initializer(init_type=InitializationType.NORMAL)(
                MD_Dis(
                    input_dim=algorithm_options["input_dim"],
                    norm=algorithm_options["dis_norm"],
                    sn=algorithm_options["dis_spectral_norm"],
                    c_dim=algorithm_options["num_domains"],
                    image_size=algorithm_options["crop_size"],
                )
            ),
            "dis2": Initializer(init_type=InitializationType.NORMAL)(
                MD_Dis(
                    input_dim=algorithm_options["input_dim"],
                    norm=algorithm_options["dis_norm"],
                    sn=algorithm_options["dis_spectral_norm"],
                    c_dim=algorithm_options["num_domains"],
                    image_size=algorithm_options["crop_size"],
                )
            ),
            "enc_c": Initializer(init_type=InitializationType.NORMAL)(
                MD_E_content(input_dim=algorithm_options["input_dim"])
            ),
            "enc_a": Initializer(init_type=InitializationType.NORMAL)(
                MD_E_attr_concat(
                    input_dim=algorithm_options["input_dim"],
                    output_nc=self._nz,
                    c_dim=algorithm_options["num_domains"],
                    norm_layer=None,
                    nl_layer=get_non_linearity(layer_type="lrelu"),
                )
            ),
            "gen": Initializer(init_type=InitializationType.NORMAL)(
                MD_G_multi_concat(
                    output_dim=algorithm_options["input_dim"], c_dim=algorithm_options["num_domains"], nz=self._nz
                )
            ),
            "disContent": Initializer(init_type=InitializationType.NORMAL)(
                MD_Dis_content(c_dim=algorithm_options["num_domains"])
            ),
        }

    def _parse_inputs(self, labeled_batch: Optional[Tuple[torch.Tensor, ...]], split: str) -> Dict[str, torch.Tensor]:
        x, y_true = None, None
        if labeled_batch is not None:
            x, y_true, metadata = labeled_batch
            x = move_to(x, self._device)
            y_true = move_to(y_true, self._device)

        group_indices = move_to(
            self._convert_group_index_to_one_hot(
                train_group_idxs=self._grouper.group_indices_by_split_name[split],
                group_idx_values=self._grouper.metadata_to_group(metadata),
            ),
            self._device,
        )
        return {"x": x, "g": group_indices, "y_true": y_true, "metadata": metadata}

    def _convert_group_index_to_one_hot(
        self, train_group_idxs: torch.Tensor, group_idx_values: torch.Tensor
    ) -> torch.Tensor:
        out_group_idx_values = group_idx_values.clone()
        for idx, group_idx in enumerate(train_group_idxs):
            out_group_idx_values[group_idx_values == group_idx] = idx
        return F.one_hot(out_group_idx_values, num_classes=train_group_idxs.size(0)).float()

    def _update_D_content(self, image: torch.Tensor, c_org: torch.Tensor) -> None:
        """Update the content discriminator.

        Adapted from:
            https://github.com/sophiajw/HistAuGAN/blob/main/histaugan/model.py#L239
        """
        self._z_content = self._models_by_names["enc_c"].forward(image)
        self._disContent_opt.zero_grad()
        pred_cls = self._models_by_names["disContent"].forward(self._z_content.detach())
        loss_D_content = self._cls_loss(pred_cls, c_org)
        loss_D_content.backward()
        self._disContent_loss = loss_D_content.item()
        nn.utils.clip_grad_norm_(self._models_by_names["disContent"].parameters(), 5)
        self._disContent_opt.step()

    def _update_D(self, image: torch.Tensor, c_org: torch.Tensor):
        self.forward(image=image, c_org=c_org)
        self._dis1_opt.zero_grad()
        self._D1_gan_loss, self._D1_cls_loss = self._backward_D(
            self._models_by_names["dis1"], real=image, fake=self._fake_encoded_img, c_org=c_org
        )
        self._dis1_opt.step()

        self._dis2_opt.zero_grad()
        self._D2_gan_loss, self._D2_cls_loss = self._backward_D(
            self._models_by_names["dis2"], real=image, fake=self._fake_random_img, c_org=c_org
        )
        self._dis2_opt.step()

    def forward(self, image: torch.Tensor, c_org: torch.Tensor):
        # input images
        if not image.size(0) % 2 == 0:
            print("Need to be even QAQ")
            input()
        half_size = image.size(0) // 2
        self._real_A = image[0:half_size]
        self._real_B = image[half_size:]
        c_org_A = c_org[0:half_size]
        c_org_B = c_org[half_size:]

        # get encoded z_c
        self._real_img = torch.cat((self._real_A, self._real_B), 0)
        self._z_content = self._models_by_names["enc_c"].forward(self._real_img)
        self._z_content_a, self._z_content_b = torch.split(self._z_content, half_size, dim=0)

        # get encoded z_a
        if self._concat:
            self._mu, self._logvar = self._models_by_names["enc_a"].forward(self._real_img, c_org)

            std = torch.exp(self._logvar.mul(0.5))
            eps = self._get_z_random(std.size(0), std.size(1)).to(image.device)
            self._z_attr = torch.add(eps.mul(std), self._mu)
        else:
            self._z_attr = self._models_by_names["enc_a"].forward(self._real_img, c_org)

        self._z_attr_a, self._z_attr_b = torch.split(self._z_attr, half_size, dim=0)
        # get random z_a
        self._z_random = self._get_z_random(half_size, self._nz).to(image.device)

        # first cross translation
        input_content_forA = torch.cat((self._z_content_b, self._z_content_a, self._z_content_b), 0)
        input_content_forB = torch.cat((self._z_content_a, self._z_content_b, self._z_content_a), 0)
        input_attr_forA = torch.cat((self._z_attr_a, self._z_attr_a, self._z_random), 0)
        input_attr_forB = torch.cat((self._z_attr_b, self._z_attr_b, self._z_random), 0)

        input_c_forA = torch.cat((c_org_A, c_org_A, c_org_A), 0)
        input_c_forB = torch.cat((c_org_B, c_org_B, c_org_B), 0)
        output_fakeA = self._models_by_names["gen"].forward(input_content_forA, input_attr_forA, input_c_forA)
        output_fakeB = self._models_by_names["gen"].forward(input_content_forB, input_attr_forB, input_c_forB)
        self._fake_A_encoded, self._fake_AA_encoded, self._fake_A_random = torch.split(
            output_fakeA, self._z_content_a.size(0), dim=0
        )
        self._fake_B_encoded, self._fake_BB_encoded, self._fake_B_random = torch.split(
            output_fakeB, self._z_content_a.size(0), dim=0
        )

        # get reconstructed encoded z_c
        self._fake_encoded_img = torch.cat((self._fake_A_encoded, self._fake_B_encoded), 0)
        self._z_content_recon = self._models_by_names["enc_c"].forward(self._fake_encoded_img)
        self._z_content_recon_b, self._z_content_recon_a = torch.split(self._z_content_recon, half_size, dim=0)

        # get reconstructed encoded z_a
        if self._concat:
            self._mu_recon, self._logvar_recon = self._models_by_names["enc_a"].forward(self._fake_encoded_img, c_org)
            std_recon = torch.exp(self._logvar_recon.mul(0.5))
            eps_recon = self._get_z_random(std_recon.size(0), std_recon.size(1)).to(image.device)
            self._z_attr_recon = torch.add(eps_recon.mul(std_recon), self._mu_recon)
        else:
            self._z_attr_recon = self._models_by_names["enc_a"].forward(self._fake_encoded_img, c_org)
        self._z_attr_recon_a, self._z_attr_recon_b = torch.split(self._z_attr_recon, half_size, dim=0)

        # second cross translation
        self._fake_A_recon = self._models_by_names["gen"].forward(
            self._z_content_recon_a, self._z_attr_recon_a, c_org_A
        )
        self._fake_B_recon = self._models_by_names["gen"].forward(
            self._z_content_recon_b, self._z_attr_recon_b, c_org_B
        )

        # for latent regression
        self._fake_random_img = torch.cat((self._fake_A_random, self._fake_B_random), 0)
        if self._concat:
            self._mu2, _ = self._models_by_names["enc_a"].forward(self._fake_random_img, c_org)
            self._mu2_a, self._mu2_b = torch.split(self._mu2, half_size, 0)
        else:
            self._z_attr_random = self._models_by_names["enc_a"].forward(self._fake_random_img, c_org)
            self._z_attr_random_a, self._z_attr_random_b = torch.split(self._z_attr_random, half_size, 0)

    def _get_z_random(self, batchSize: int, nz: int):
        z = torch.randn(batchSize, nz)
        return z

    def _backward_D(
        self, netD: nn.Module, real: torch.Tensor, fake: torch.Tensor, c_org: torch.Tensor
    ) -> Tuple[float, float]:
        pred_fake, pred_fake_cls = netD.forward(fake.detach())
        pred_real, pred_real_cls = netD.forward(real)

        loss_D_gan = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all0 = torch.zeros_like(out_fake)
            all1 = torch.ones_like(out_real)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_D_gan += ad_true_loss + ad_fake_loss

        loss_D_cls = self._cls_loss(pred_real_cls, c_org)
        loss_D = loss_D_gan + self._lambda_cls * loss_D_cls
        loss_D.backward()
        self._D_loss = loss_D.item()
        return loss_D_gan, loss_D_cls

    def _update_EG(self, image: torch.Tensor, c_org: torch.Tensor):
        # update G, Ec, Ea
        self._enc_c_opt.zero_grad()
        self._enc_a_opt.zero_grad()
        self._gen_opt.zero_grad()
        self._backward_EG(image=image, c_org=c_org)
        self._backward_G_alone(c_org=c_org)
        self._enc_c_opt.step()
        self._enc_a_opt.step()
        self._gen_opt.step()

    def _backward_EG(self, image: torch.Tensor, c_org: torch.Tensor):
        # content Ladv for generator
        loss_G_GAN_content = self._backward_G_GAN_content(self._z_content, c_org=c_org)

        # Ladv for generator
        pred_fake, pred_fake_cls = self._models_by_names["dis1"].forward(self._fake_encoded_img)
        loss_G_GAN = 0
        for out_a in pred_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake)
            loss_G_GAN += nn.functional.binary_cross_entropy(outputs_fake, all_ones)

        # classification
        loss_G_cls = self._cls_loss(pred_fake_cls, c_org) * self._lambda_cls_G

        # self and cross-cycle recon
        loss_G_L1_self = (
            torch.mean(torch.abs(image - torch.cat((self._fake_AA_encoded, self._fake_BB_encoded), 0)))
            * self._lambda_rec
        )
        loss_G_L1_cc = (
            torch.mean(torch.abs(image - torch.cat((self._fake_A_recon, self._fake_B_recon), 0))) * self._lambda_rec
        )

        # KL loss - z_c
        loss_kl_zc = self._l2_regularize(self._z_content) * 0.01

        # KL loss - z_a
        if self._concat:
            kl_element = torch.add((-torch.add(self._mu.pow(2), self._logvar.exp())) + 1, self._logvar)
            loss_kl_za = torch.sum(kl_element) * (-0.5) * 0.01
        else:
            loss_kl_za = self._l2_regularize(self._z_attr) * 0.01

        loss_G = loss_G_GAN + loss_G_cls + loss_G_L1_self + loss_G_L1_cc + loss_kl_zc + loss_kl_za
        loss_G += loss_G_GAN_content
        loss_G.backward(retain_graph=True)

        self._gan_loss = loss_G_GAN.item()
        self._gan_cls_loss = loss_G_cls.item()
        self._gan_loss_content = loss_G_GAN_content.item()
        self._kl_loss_zc = loss_kl_zc.item()
        self._kl_loss_za = loss_kl_za.item()
        self._l1_self_rec_loss = loss_G_L1_self.item()
        self._l1_cc_rec_loss = loss_G_L1_cc.item()
        self._G_loss = loss_G.item()

    def _backward_G_GAN_content(self, data: torch.Tensor, c_org: torch.Tensor) -> float:
        pred_cls = self._models_by_names["disContent"].forward(data)
        loss_G_content = self._cls_loss(pred_cls, 1 - c_org)
        return loss_G_content

    def _l2_regularize(self, mu: torch.Tensor) -> torch.Tensor:
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def _backward_G_alone(self, c_org: torch.Tensor) -> None:
        # Ladv for generator
        pred_fake, pred_fake_cls = self._models_by_names["dis2"].forward(self._fake_random_img)
        loss_G_GAN2 = 0
        for out_a in pred_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake)
            loss_G_GAN2 += nn.functional.binary_cross_entropy(outputs_fake, all_ones)

        # classification
        loss_G_cls2 = self._cls_loss(pred_fake_cls, c_org) * self._lambda_cls_G

        # latent regression loss
        if self._concat:
            loss_z_L1_a = torch.mean(torch.abs(self._mu2_a - self._z_random)) * 10
            loss_z_L1_b = torch.mean(torch.abs(self._mu2_b - self._z_random)) * 10
        else:
            loss_z_L1_a = torch.mean(torch.abs(self._z_attr_random_a - self._z_random)) * 10
            loss_z_L1_b = torch.mean(torch.abs(self._z_attr_random_b - self._z_random)) * 10

        loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2 + loss_G_cls2
        loss_z_L1.backward()
        self._l1_recon_z_loss = loss_z_L1_a.item() + loss_z_L1_b.item()
        self._gan2_loss = loss_G_GAN2.item()
        self._gan2_cls_loss = loss_G_cls2.item()
        self._z_L1_loss = loss_z_L1.item()

    def objective(
        self, in_dict: Dict[str, Union[torch.Tensor, SignalType]], return_loss_components: bool
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns a tuple of objective that can be backpropagate and a dictionary of all loss term."""
        if self._use_amp:
            with torch.cuda.amp.autocast():
                loss_dict = self._loss.compute(
                    in_dict=in_dict, return_dict=True, return_loss_components=return_loss_components
                )
        else:
            loss_dict = self._loss.compute(
                in_dict=in_dict, return_dict=True, return_loss_components=return_loss_components
            )

        objective_loss_name = self._loss.agg_metric_field
        labeled_loss = loss_dict[objective_loss_name]
        non_objective_loss_by_name = {k: v for k, v in loss_dict.items() if k != objective_loss_name}

        if self._use_unlabeled_y and "unlabeled_y_true" in in_dict:
            unlabeled_loss = self._loss.compute(
                in_dict["unlabeled_y_pred"], in_dict["unlabeled_y_true"], return_dict=False
            )
            lab_size = len(in_dict["y_pred"])
            unl_size = len(in_dict["unlabeled_y_pred"])
            return (lab_size * labeled_loss + unl_size * unlabeled_loss) / (
                lab_size + unl_size
            ), non_objective_loss_by_name
        else:
            return labeled_loss, non_objective_loss_by_name

    def _process_batch(
        self,
        labeled_batch: Optional[Tuple[torch.Tensor, ...]],
        split: str,
        unlabeled_batch: Optional[Tuple[torch.Tensor, ...]] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        input_dict = self._parse_inputs(labeled_batch, split=split)
        y_true, metadata = input_dict["y_true"], input_dict["metadata"]
        # out_dict = self._get_model_output(x, y_true, unlabeled_x)
        if self._training_mode == ContrimixTrainingMode.ENCODERS:
            return {
                "y_true": y_true,
                "metadata": metadata,
                "is_training": self._is_training,
                "g": input_dict["g"],
                "backbone": self._models_by_names["backbone"],
                "x_org": input_dict["x"],
                # These losses are not available in Histogan for the evaluation set.
                "gan_loss": -1.0,
                "gan_cls_loss": -1.0,
                "z_L1_loss": -1.0,
                "l1_self_rec_loss": -1.0,
                "l1_cc_rec_loss": -1.0,
            }
        elif self._training_mode == ContrimixTrainingMode.BACKBONE:
            return {
                "y_true": y_true,
                "metadata": metadata,
                "is_training": self._is_training,
                "g": input_dict["g"],
                "backbone": self._models_by_names["backbone"],
                "x_org": input_dict["x"],
                "gen": self._models_by_names["gen"],
                "enc_c": self._models_by_names["enc_c"],
                "enc_a": self._models_by_names["enc_a"],
            }
        else:
            raise ValueError("Unknown training model")

    def _save_display_image(self):
        # for display
        image_display = torch.cat(
            (
                self._real_A[0:1].detach().cpu(),
                self._fake_B_encoded[0:1].detach().cpu(),
                self._fake_B_random[0:1].detach().cpu(),
                self._fake_AA_encoded[0:1].detach().cpu(),
                self._fake_A_recon[0:1].detach().cpu(),
                self._real_B[0:1].detach().cpu(),
                self._fake_A_encoded[0:1].detach().cpu(),
                self._fake_A_random[0:1].detach().cpu(),
                self._fake_BB_encoded[0:1].detach().cpu(),
                self._fake_B_recon[0:1].detach().cpu(),
            ),
            dim=0,
        )

        image_dis = torchvision.utils.make_grid(image_display, nrow=image_display.size(0) // 2) / 2 + 0.5
        io.imsave("image_display.png", (image_dis.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))
