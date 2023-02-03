"""This modules implements the ContriMix Augmentation algorithm."""
import logging
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch

from ._utils import move_to
from .single_model_algorithm import SingleModelAlgorithm
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.metrics import ElementwiseLoss
from ip_drit.common.metrics import Metric
from ip_drit.models import AttributeEncoder
from ip_drit.models import ContentEncoder
from ip_drit.models.wild_model_initializer import initialize_model_from_configuration


class ContriMix(SingleModelAlgorithm):
    """A class that implements the ContriMix algorithm.

    Args:
        config: A dictionary that defines the configuration to load the model.
        d_out: The dimension of the model output.
        grouper: A grouper object that defines the groups for which we compute/log statistics for.
        loss: The loss module.
        metric: The metric to use.

    References:
        https://binhu7.github.io/courses/ECE598/Spring2019/files/Lecture4.pdf
    """

    def __init__(
        self,
        config: Dict[str, Any],
        d_out: int,
        grouper: AbstractGrouper,
        loss: ElementwiseLoss,
        metric: Metric,
        n_train_steps: int,
    ) -> None:
        backbone_network = initialize_model_from_configuration(config, d_out, output_classifier=True)
        super().__init__(
            config=config,
            model=backbone_network,
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
