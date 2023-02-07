from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch

from ._metric import ElementwiseMetric
from ._metric import Metric
from ._metric import MultiTaskMetric
from ._utils import maximum


class Loss(Metric):
    """A class for the loss metric.

    Args:
        loss_fn: A function to compute the loss.
        name (optional): The name of the metric. Defaults to None.
    """

    def __init__(self, loss_fn: Optional[Callable], name: Optional[str] = None):
        self.loss_fn = loss_fn
        if name is None:
            name = "loss"
        super().__init__(name=name)

    def _compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Helper for computing element-wise metric, implemented for each metric.

        Args:
            y_pred: Predicted targets or model output
            y_true: True targets

        Returns:
            An element_wise_metrics tensor of size (batch_size, ).
        """
        return self.loss_fn(y_pred, y_true)

    def worst(self, metrics: Union[List, np.ndarray, torch.Tensor]) -> float:
        """Computes the worst-case metric.

        Args:
            metrics: Metrics

        Returns
            The worst-case metric
        """
        return maximum(metrics)


class ElementwiseLoss(ElementwiseMetric):
    """A class for element-wise loss.

    Args:
        loss_fn: A function to compute the loss.
        name (optional): The name of the metric. Defaults to None.
    """

    def __init__(self, loss_fn: Optional[Callable], name: Optional[str] = None):
        self.loss_fn = loss_fn
        if name is None:
            name = "loss"
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Helper for computing element-wise metric, implemented for each metric.

        Args:
            y_pred: Predicted targets or model output
            y_true: True targets

        Returns:
            An element_wise_metrics tensor of size (batch_size, ).
        """
        return self.loss_fn(y_pred, y_true)

    def worst(self, metrics: Union[List, np.ndarray, torch.Tensor]) -> float:
        """Computes the worst-case metric.

        Args:
            metrics: Metrics

        Returns
            The worst-case metric
        """
        return maximum(metrics)


class MultiTaskLoss(MultiTaskMetric):
    """A class for multi-task loss.

    Args:
        loss_fn: A function to compute the loss.
        name (optional): The name of the metric. Defaults to None.
    """

    def __init__(self, loss_fn: Optional[Callable], name: Optional[str] = None) -> None:
        self.loss_fn = loss_fn  # should be elementwise
        if name is None:
            name = "loss"
        super().__init__(name=name)

    def _compute_flattened_metrics(
        self, flattened_y_pred: torch.Tensor, flattened_y_true: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            flattened_y_pred = flattened_y_pred.float()
            flattened_y_true = flattened_y_true.float()
        elif isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
            flattened_y_true = flattened_y_true.long()
        flattened_y_true = torch.reshape(flattened_y_true, flattened_y_pred.shape)
        flattened_loss = self.loss_fn(flattened_y_pred, flattened_y_true)
        return flattened_loss

    def worst(self, metrics: Union[List, np.ndarray, torch.Tensor]) -> float:
        """Computes the worst-case metric.

        Args:
            metrics: Metrics

        Returns
            The worst-case metric
        """
        return maximum(metrics)
