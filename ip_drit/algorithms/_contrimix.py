"""This modules implements the ContriMix Augmentation algorithm."""
import logging
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
from ip_drit.models import AttributeEncoder
from ip_drit.models import ContentEncoder
from ip_drit.models import ImageGenerator
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
        cont_enc = ContentEncoder(in_channels=self._NUM_INPUT_CHANNELS, num_stain_vectors=self._NUM_STAIN_VECTORS)
        attr_enc = AttributeEncoder(
            in_channels=self._NUM_INPUT_CHANNELS, num_stain_vectors=self._NUM_STAIN_VECTORS, out_channels=d_out
        )
        gen = ImageGenerator(convert_to_absorbance_in_between=convert_to_absorbance_in_between)

        super().__init__(
            config=config,
            models=[backbone_network, cont_enc, attr_enc, gen],
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self._use_unlabeled_y = config["use_unlabeled_y"]

    def process_batch(
        self, batch: Tuple[torch.Tensor, ...], unlabeled_batch: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Dict[str, torch.Tensor]:
        x, y_true, metadata = batch
        x = move_to(x, self._device)
        y_true = move_to(y_true, self._device)
        _ = move_to(self._grouper.metadata_to_group_indices(metadata), self._device)

    def get_model_output(self, x: torch.Tensor, y_true: torch.Tensor):
        if self._model.needs_y_input:
            if self.training:
                outputs = self._model(x, y_true)
            else:
                outputs = self._model(x, None)
        else:
            outputs = self._model(x)
        return outputs
