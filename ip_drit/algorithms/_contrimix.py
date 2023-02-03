"""This modules implements the ContriMix Augmentation algorithm."""
import logging
from enum import auto
from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch

from ._utils import move_to
from .multi_model_algorithm import MultimodelAlgorithm
from ip_drit.common.grouper import AbstractGrouper
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
    """

    _NUM_INPUT_CHANNELS = 3
    _NUM_STAIN_VECTORS = 8
    _DOWNSAMPLING_FACTOR = 4

    def __init__(
        self,
        config: Dict[str, Any],
        d_out: int,
        grouper: AbstractGrouper,
        loss: ElementwiseLoss,
        metric: Metric,
        n_train_steps: int,
        convert_to_absorbance_in_between: bool = True,
    ) -> None:
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
        self._convert_to_absorbance_in_between = convert_to_absorbance_in_between

    def _process_batch(
        self, batch: Tuple[torch.Tensor, ...], unlabeled_batch: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Dict[str, torch.Tensor]:
        x, y_true, metadata = batch
        x = move_to(x, self._device)
        y_true = move_to(y_true, self._device)
        group_indices = move_to(self._grouper.metadata_to_group_indices(metadata), self._device)

        y_pred = self._get_model_output(x, y_true)

        results = {"g": group_indices, "y_true": y_true, "y_pred": y_pred, "metadata": metadata}

        if unlabeled_batch is not None:
            if self.use_unlabeled_y:  # expect loaders to return x,y,m
                x, y, metadata = unlabeled_batch
                y = move_to(y, self.device)
            else:
                x, metadata = unlabeled_batch
            x = move_to(x, self.device)
            results["unlabeled_metadata"] = metadata
            if self.use_unlabeled_y:
                results["unlabeled_y_pred"] = self._get_model_output(x, y)
                results["unlabeled_y_true"] = y
            results["unlabeled_g"] = self.grouper.metadata_to_group(metadata).to(self.device)
        return results

    def _get_model_output(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Computes the model outputs."""
        cont_enc = self._models_by_names["cont_enc"]
        attr_enc = self._models_by_names["attr_enc"]
        im_gen = self._models_by_names["im_gen"]
        backbone = self._models_by_names["backbone"]
        if self._convert_to_absorbance_in_between:
            x_abs, signal_type = self._trans_to_abs_converter(im_and_sig_type=(x, SignalType.TRANS))
            zc = cont_enc(x_abs)
            za = attr_enc(x_abs)
            x_abs_self_recon = im_gen(zc, za)
            x_self_recon = self._abs_to_trans_converter(im_and_sig_type=(x_abs_self_recon, signal_type))

        if backbone.needs_y_input:
            if self.training:
                y_pred = backbone(x_self_recon, y_true)
            else:
                y_pred = backbone(x, None)
        else:
            y_pred = backbone(x)
        return y_pred

    def objective(self, results):
        labeled_loss = self._loss.compute(results["y_pred"], results["y_true"], return_dict=False)
        if self._use_unlabeled_y and "unlabeled_y_true" in results:
            unlabeled_loss = self._loss.compute(
                results["unlabeled_y_pred"], results["unlabeled_y_true"], return_dict=False
            )
            lab_size = len(results["y_pred"])
            unl_size = len(results["unlabeled_y_pred"])
            return (lab_size * labeled_loss + unl_size * unlabeled_loss) / (lab_size + unl_size)
        else:
            return labeled_loss
