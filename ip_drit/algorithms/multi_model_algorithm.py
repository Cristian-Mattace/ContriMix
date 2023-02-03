"""A module that defines the algorithm with multiple underlying models."""
import logging
from typing import Any
from typing import Dict
from typing import List

import torch
from torch.nn import DataParallel

from ._group_algorithm import GroupAlgorithm
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.metrics import Metric
from ip_drit.optimizer import initialize_optimizer
from ip_drit.scheduler import initialize_scheduler


class MultimodelAlgorithm(GroupAlgorithm):
    """An algorithm that contains many underlying models.

    Args:
        config: A configuration dictionary.
        model:s A list of models used in the algorithm.
        grouper: A grouper object that defines the groups for which we compute/log statistics for.
        loss: The loss object.
        metric: The metric to use.
        n_train_steps: The number of training steps.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        models: List[torch.nn.Module],
        grouper: AbstractGrouper,
        loss,
        metric: Metric,
        n_train_steps: int,
    ):
        self._loss = loss
        logged_metrics = [self._loss]
        if metric is not None:
            self._metric = metric
            logged_metrics.append(self._metric)
        else:
            self._metric = None

        if config["use_data_parallel"]:
            parallelized_models = [DataParallel(m) for m in models]
        else:
            parallelized_models = models

        if not hasattr(self, "optimizer") or self.optimizer is None:
            self._optimizer = initialize_optimizer(config, models=models)
        self._max_grad_norm = config["max_grad_norm"]

        logging.info(f"Using device {config['device']} for training.")
        for m in parallelized_models:
            m.to(config["device"])

        self._batch_idx = 0
        self._gradient_accumulation_steps = config["gradient_accumulation_steps"]

        super().__init__(
            device=config["device"],
            grouper=grouper,
            logged_metrics=logged_metrics,
            logged_fields=["objective"],
            schedulers=[initialize_scheduler(config, self._optimizer, n_train_steps)],
            scheduler_metric_names=[config["scheduler_metric_name"]],
        )
