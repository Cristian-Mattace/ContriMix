from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch

from ._utils import move_to
from .single_model_algorithm import SingleModelAlgorithm
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.metrics import Metric
from ip_drit.loss import ElementwiseLoss
from ip_drit.models.wild_model_initializer import initialize_model_from_configuration


class ERM(SingleModelAlgorithm):
    """A class that implements the ERM algorithm.

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
        model = initialize_model_from_configuration(config["model"], d_out, output_classifier=False)
        super().__init__(
            config=config, model=model, grouper=grouper, loss=loss, metric=metric, n_train_steps=n_train_steps
        )
        self._use_unlabeled_y = config["use_unlabeled_y"]

    def objective(self, results: Dict[str, Any]) -> float:
        return self._loss.compute(results["y_pred"], results["y_true"], return_dict=False)

    def _process_unlabeled_batch(self, unlabeled_batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        raise RuntimeError("ERM algorithm does not support the use of unlabeled data.")
