"""This modules implements the ContriMix Augmentation algorithm."""
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from ._utils import move_to
from .multi_model_algorithm import MultimodelAlgorithm
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.metrics import ContriMixLoss
from ip_drit.common.metrics import ElementwiseLoss
from ip_drit.common.metrics import Metric
from ip_drit.models import AbsorbanceImGenerator
from ip_drit.models import AbsorbanceToTransmittance
from ip_drit.models import AttributeEncoder
from ip_drit.models import ContentEncoder
from ip_drit.models import SignalType
from ip_drit.models import TransmittanceToAbsorbance
from ip_drit.models.wild_model_initializer import initialize_model_from_configuration


class ContriMix(MultimodelAlgorithm):
    """A class that implements the ContriMix algorithm.

    Args:
        config: A dictionary that defines the configuration to load the model.
        d_out: The dimension of the model output.
        grouper: A grouper object that defines the groups for which we compute/log statistics for.
        loss: The loss module.
        metric: The metric to use.
        n_train_steps: The number of training steps.
        convert_to_absorbance_in_between (optional): If True (default), the input image will be converted to absorbance
            before decomposing into content and attribute.
        num_mxing_per_image (optional): The number of mixing images for each original image. Defaults to 5.
    """

    _NUM_INPUT_CHANNELS = 3
    _NUM_STAIN_VECTORS = 8
    _DOWNSAMPLING_FACTOR = 4

    def __init__(
        self,
        config: Dict[str, Any],
        d_out: int,
        grouper: AbstractGrouper,
        loss: ContriMixLoss,
        metric: Metric,
        n_train_steps: int,
        convert_to_absorbance_in_between: bool = True,
        num_mixing_per_image: int = 15,
    ) -> None:
        if not isinstance(loss, ContriMixLoss):
            raise ValueError(f"The specified loss module is of type {type(loss)}, not ContriMixLoss!")

        backbone_network = initialize_model_from_configuration(config, d_out, output_classifier=True)

        if convert_to_absorbance_in_between:
            self._trans_to_abs_converter = TransmittanceToAbsorbance()
            self._abs_to_trans_converter = AbsorbanceToTransmittance()
        else:
            raise ValueError("ContriMix without converting to absorbance in between is not supported yet!")

        super().__init__(
            config=config,
            models_by_names={
                "backbone": backbone_network,
                "cont_enc": ContentEncoder(
                    in_channels=self._NUM_INPUT_CHANNELS, num_stain_vectors=self._NUM_STAIN_VECTORS
                ),
                "attr_enc": AttributeEncoder(
                    in_channels=self._NUM_INPUT_CHANNELS, num_stain_vectors=self._NUM_STAIN_VECTORS
                ),
                "im_gen": AbsorbanceImGenerator(),
            },
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self._use_unlabeled_y = config["use_unlabeled_y"]
        self._num_mixing_per_image = num_mixing_per_image
        self._convert_to_absorbance_in_between = convert_to_absorbance_in_between

    def _process_batch(
        self, batch: Tuple[torch.Tensor, ...], unlabeled_batch: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Dict[str, torch.Tensor]:
        x, y_true, metadata = batch
        x = move_to(x, self._device)
        y_true = move_to(y_true, self._device)
        group_indices = move_to(self._grouper.metadata_to_group_indices(metadata), self._device)

        out_dict = self._get_model_output(x, y_true)

        results = {"g": group_indices, "y_true": y_true, "metadata": metadata, **out_dict}

        if unlabeled_batch is not None:
            raise ValueError("ContriMix does not support unlabeled data yet!")
        return results

    def _get_model_output(self, x: torch.Tensor, y_true: torch.Tensor) -> Dict[str, Union[torch.Tensor, SignalType]]:
        """Computes the model outputs.

        Args:
            x: The tensor of the input image.
            y_true: The groundtruth label of x.

        Returns:
            A dictionary of tensors, keyed by the name of the tensor.
        """
        cont_enc = self._models_by_names["cont_enc"]
        attr_enc = self._models_by_names["attr_enc"]
        all_target_image_indices = self._select_random_image_indices_by_image_index(batch_size=x.shape[0])
        all_target_image_indices = torch.stack(all_target_image_indices, dim=0)  # (Minibatch dim x #augmentations)
        if self._convert_to_absorbance_in_between:
            x_abs, signal_type = self._trans_to_abs_converter(im_and_sig_type=(x, SignalType.TRANS))
            zc = cont_enc(x_abs)
            za = attr_enc(x_abs)

            za_targets: List[torch.Tensor] = []
            for mix_idx in range(self._num_mixing_per_image):
                # Extract attributes of the target patches.
                target_im_idxs = all_target_image_indices[:, mix_idx]
                za_targets.append(za[target_im_idxs])
                # x_abs_cross_translation = im_gen(zc, za_target)
                # _ = self._abs_to_trans_converter(im_and_sig_type=(x_abs_cross_translation, signal_type))[0]
            za_targets = torch.cat(za_targets, dim=0)

        return {"zc": zc, "za": za, "za_targets": za_targets, "x_org": x, "sig_type": signal_type, "y_true": y_true}

    def _select_random_image_indices_by_image_index(self, batch_size: int) -> List[torch.Tensor]:
        """Returns a list of tensors that contains target image indices to sample from.

        Args:
            batch_size: The size of the training batch.

        Returns:
            A list of tensors in which each is the index of the images in the minibatch that we can use for ContriMix.
        """
        return [torch.randint(low=0, high=batch_size, size=(self._num_mixing_per_image,)) for _ in range(batch_size)]

    def objective(self, in_dict: Dict[str, Union[torch.Tensor, SignalType]]):
        im_gen = self._models_by_names["im_gen"]
        backbone = self._models_by_names["backbone"]
        x_abs_self_recon = im_gen(in_dict["zc"], in_dict["za"])
        x_self_recon = self._abs_to_trans_converter(im_and_sig_type=(x_abs_self_recon, in_dict["sig_type"]))[0]

        if backbone.needs_y_input:
            raise ValueError("Backbone network with y-input is not supported")
        else:
            y_pred = backbone(x_self_recon)
        in_dict["y_pred"] = y_pred

        labeled_loss = self._loss.compute(in_dict=in_dict, return_dict=False)

        if self._use_unlabeled_y and "unlabeled_y_true" in in_dict:
            unlabeled_loss = self._loss.compute(
                in_dict["unlabeled_y_pred"], in_dict["unlabeled_y_true"], return_dict=False
            )
            lab_size = len(in_dict["y_pred"])
            unl_size = len(in_dict["unlabeled_y_pred"])
            return (lab_size * labeled_loss + unl_size * unlabeled_loss) / (lab_size + unl_size)
        else:
            return labeled_loss
