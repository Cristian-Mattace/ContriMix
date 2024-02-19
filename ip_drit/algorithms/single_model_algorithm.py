"""A module that defines an algorithm with a model."""
from abc import abstractmethod
from enum import auto
from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_

from ._group_algorithm import GroupAlgorithm
from ._utils import move_to
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.metrics import Metric
from ip_drit.optimizer import initialize_optimizer
from ip_drit.patch_transform import AbstractJointTensorTransform
from ip_drit.scheduler import initialize_scheduler


class ModelAlgorithm(Enum):
    """A class that defines the algorithm for the model."""

    ERM = auto()
    CONTRIMIX = auto()
    HISTAUGAN = auto()
    LABEL_ASSOCIATED_CONTENT_CONTRIMIX = auto()


class SingleModelAlgorithm(GroupAlgorithm):
    """An class for algorithm that has an underlying model.

    Args:
        config: A configuration dictionary.
        model: The underlying backbone model used for the algorithm.
        grouper: A grouper object that defines the groups for which we compute/log statistics for.
        loss: The loss object.
        metric: The metric to use.
        n_train_steps: The number of training steps.
        batch_transform (optional): A module perform batch processing. Defaults to None, in which case, no batch
            processing will be performed.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module,
        grouper: AbstractGrouper,
        loss,
        metric: Metric,
        n_train_steps: int,
        batch_transform: Optional[AbstractJointTensorTransform] = None,
    ):
        self._loss = loss
        self._has_log = False
        logged_metrics = [self._loss]
        if metric is not None:
            self._metric = metric
            logged_metrics.append(self._metric)
        else:
            self._metric = None

        # initialize models, optimizers, and schedulers
        if not hasattr(self, "optimizer") or self.optimizer is None:
            self._optimizer = initialize_optimizer(config, models=[model])
        self._max_grad_norm = config["max_grad_norm"]

        if not config["use_ddp_over_dp"]:
            parallelized_model = DataParallel(model)
        else:
            raise ValueError("DDP is not supported in single-model algorithms!")

        print(f"Using device {config['device']} for training.")
        parallelized_model.to(config["device"])
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

        # The parallelized_model does not contains the needs_y_input. We need to copy here before it gets cleaned up.
        parallelized_model.needs_y_input = model.needs_y_input

        self._model = parallelized_model
        self._batch_transform = batch_transform

    def _get_model_output(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self._model.needs_y_input:
            if self.training:
                outputs = self._model(x, y_true)
            else:
                outputs = self._model(x, None)
        else:
            outputs = self._model(x)
        return outputs

    def _process_batch(
        self, labeled_batch: Tuple[torch.Tensor, ...], unlabeled_batch: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Dict[str, Any]:
        """Process a single batch of data.

        ERM defines its own process_batch to handle if self.use_unlabeled_y is true.

        Args:
            batch: A tuple of tensor for a batch of data yielded by the labeled data loaders.
            unlabeled_batch (optional): A batch of data yielded by unlabeled data loader. Defaults to None.

        Returns
            A dictionary result of tensors, keyed by the name of the fields, which are the followings
                y_true: Groundtruth labels for batch
                g: Groups for batch
                metadata: Metadata for batch
                y_pred: model output for batch
                unlabeled_g: Groups for unlabeled batch
                unlabeled_metadata: Metadata for unlabeled batch
                unlabeled_y_pred: Predictions for unlabeled batch for fully-supervised ERM experiments
                unlabeled_y_true: True labels for unlabeled batch for fully-supervised ERM experiments
        """
        x, y_true, metadata = labeled_batch
        x = move_to(x, self._device)
        y_true = move_to(y_true, self._device)
        if self._batch_transform is not None:
            x, y_true = self._batch_transform.transform(x, y_true)
        g = move_to(self._grouper.metadata_to_group(metadata), self._device)

        results = {"g": g, "y_true": y_true, "y_pred": self._get_model_output(x, y_true), "metadata": metadata}

        if unlabeled_batch is not None:
            results.update(self._process_unlabeled_batch(unlabeled_batch=unlabeled_batch))
        return results

    @abstractmethod
    def _process_unlabeled_batch(self, unlabeled_batch: Tuple[torch.Tensor]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def objective(self, results: Dict[str, Any]) -> float:
        """Returnst the value of the objective function."""
        raise NotImplementedError

    def evaluate(
        self,
        batch: Tuple[torch.Tensor, ...],
        ddp_params: Optional[Dict[str, Any]] = None,
        return_loss_components: bool = True,
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """Process the batch and update the log, without updating the model.

        Args:
            batch: A batch of data yielded by data loaders.
            ddp_params: A dictionary that contains the parameters for DDP.
            return_loss_components (optional): If True, returns different components of the loss dictionary.. Defaults
                to True.

        Returns:
            A dictionary of the results, keyed by the field names. There are following fields.
                g: Groups for batch
                y_true: Ground truth labels for batch
                metadata: Metadata for batch
                y_pred: model output for batch
                outputs: A tensor for the output
                objective: The value of the objective.
        """
        assert not self._is_training
        results = self._process_batch(batch)
        results["objective"] = self.objective(results, return_loss_components=return_loss_components).item()
        self.update_log(results)
        return self._sanitize_dict(results)

    def update(
        self,
        labeled_batch: Tuple[torch.Tensor, ...],
        return_loss_components: bool,
        unlabeled_batch: Optional[Tuple[torch.Tensor, ...]] = None,
        is_epoch_end: bool = False,
        epoch: Optional[int] = None,
    ):
        """Process the batch, update the log, and update the model.

        Args:
            labeled_batch: A batch of data yielded by data loaders.
            unlabeled_batch (optional): A batch of data yielded by unlabeled data loader or None.
            is_epoch_end (optional): Whether this batch is the last batch of the epoch. If so, force optimizer to step,
                regardless of whether this batch idx divides self.gradient_accumulation_steps evenly. Defaults to False.
            epoch (optional): The index of the current epoch.
            return_loss_components (optional): Returns different components of the loss.
            return_loss_components (optional): If True, the component of the loss will be return.

        Returns:
            A dictionary of the results, keyed by the field names. There are following fields.
                g: Groups indices for samples in the for batch.
                y_true: Ground truth labels for batch.
                metadata: Metadata for batch.
                y_pred: model output for batch.
                outputs: A tensor for the output.
                objective: The value of the objective.
        """
        assert self._is_training

        # process this batch
        results = self._process_batch(labeled_batch, unlabeled_batch)

        # update running statistics and update model if we've reached end of effective batch
        self._update(
            results,
            should_step=(((self._batch_idx + 1) % self._gradient_accumulation_steps == 0) or (is_epoch_end),),
            return_loss_components=return_loss_components,
        )

        if return_loss_components:
            self.update_log(results)

        if is_epoch_end:
            self._batch_idx = 0
        else:
            self._batch_idx += 1

        # return only this batch's results
        return self._sanitize_dict(results)

    def _update(self, results: Dict[str, Any], should_step: bool, return_loss_components: bool) -> None:
        """Computes the objective and updates the model.

        Also updates the results dictionary yielded by process_batch().
        Should be overridden to change algorithm update beyond modifying the objective.
        """
        objective = self.objective(results, return_loss_components=return_loss_components)
        results["objective"] = objective.item()
        objective.backward()

        if should_step:
            if self._max_grad_norm:
                clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()
            self.step_schedulers(is_epoch=False, metrics=self._log_dict, log_access=False)
            self._model.zero_grad()

    def save_metric_for_logging(self, results, metric, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                results[metric] = value.item()
            else:
                raise ValueError(f"Metric value can only be a number or single-element tensor. value={value}")
        else:
            results[metric] = value

    @property
    def model(self) -> nn.Module:
        return self._model

    def update_loss_weight_based_on_epoch(self, epoch: int) -> None:
        """Update the weights of the loss terms for each epoch."""
        pass
