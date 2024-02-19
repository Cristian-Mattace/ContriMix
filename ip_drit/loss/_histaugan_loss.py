"""A module to compute the HistauGAN loss."""
from enum import auto
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io

from ..common.metrics._base import Metric
from ..common.metrics._base import MultiTaskMetric


class HistauGANLoss(MultiTaskMetric):
    """A class that defines a HistauGAN loss in absorbance space.

    Because most of the loss term are calculated in the algoirhm class. This class is mainly for logging.

    Args:
        loss_params: A dictionary that defines the parameters for the loss.
        name (optional): The name of the loss. Defaults to "histaugan_loss".
    """

    def __init__(self, loss_params: Dict[str, Any], name: Optional[str] = "histaugan_loss") -> None:
        super().__init__(name)

    def compute(
        self, in_dict: Dict[str, torch.Tensor], return_dict: bool = True, return_loss_components: bool = True
    ) -> Union[Metric, Dict]:
        """Computes metric. This is a wrapper around _compute.

        Args:
            in_dict: A dictionary from the inputs of the forward pass of contrimix, key by the name of the field.
            return_dict: Whether to return the output as a dictionary or a tensor.

        Returns:
             If return_dict=False:
            - metric (0-dim tensor): metric. If the inputs are empty, returns tensor(0.).
        Output (return_dict=True):
            - results (dict): Dictionary of results, mapping metric.agg_metric_field to avg_metric.
        """
        print("Done")
