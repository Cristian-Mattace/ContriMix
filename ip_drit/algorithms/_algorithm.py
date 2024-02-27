from typing import Any
from typing import Dict
from typing import Tuple

import torch
import torch.nn as nn

from ._utils import detach_and_clone
from ._utils import move_to


class Algorithm(nn.Module):
    """A class that defines the algorithm to run on.

    Args:
        device: The device to run the algorithm on
    """

    def __init__(self, device: str):
        super().__init__()
        self._device = device
        self._out_device = "cpu"
        self._has_log = False
        self.reset_log()

    def update(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process the batch, update the log, and update the model.

        Args:
            A batch of data yielded by data loaders

        Returns:
            A dictionary of metadata about the batch.
        """
        raise NotImplementedError

    def evaluate(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process the batch and update the log, without updating the model.

        Args:
            batch: A batch of data yielded by data loaders

        Returns:
            A dictionary about the batch, such as:
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
        """
        raise NotImplementedError

    def train(self, is_training_mode: bool = True) -> None:
        """Switch to train mode.

        Args:
            is_training_mode (optional): If True, in the training mode. Defaults to True.
        """
        self._is_training = is_training_mode
        super().train(is_training_mode)
        self.reset_log()

    @property
    def has_log(self) -> bool:
        return self._has_log

    def reset_log(self) -> None:
        """Resets log by clearing out the internal log, Algorithm.log_dict."""
        self._has_log = False
        self._log_dict = {}

    def _update_log(self, results: Dict[str, Any]) -> None:
        """Updates the internal log, Algorithm.log_dict.

        Args:
            results: The dictionary result.
        """
        raise NotImplementedError

    def _get_log(self) -> Dict[str, Any]:
        """Sanitizes the internal log (Algorithm.log_dict) and outputs it."""
        raise NotImplementedError

    def get_pretty_log_str(self) -> str:
        """Gets a pretty log string for the algorithm."""
        raise NotImplementedError

    def step_schedulers(self, is_epoch: bool, metrics: Dict[str, Any] = {}, log_access: bool = False) -> None:
        """Updates all relevant schedulers.

        Args:
            is_epoch: epoch-wise update if set to True, batch-wise update otherwise
            metrics: a dictionary of metrics that can be used for scheduler updates
            log_access: whether metrics from self.get_log() can be used to update schedulers
        """
        raise NotImplementedError

    def _sanitize_dict(self, in_dict: Dict[str, Any], to_out_device: bool = True) -> Dict[str, Any]:
        """A helper function to sanitize an put dictionary.

        The sanitization happens by:
            Moving to the specified output device
            Removing any gradient information
            Detaching and cloning the tensors

        Args:
            in_dict: The input dictionary to sanitize.
            to_out_device (optional): If True, the sanitized dictionary will be moved to the output device.. Defaults to
                True.

        Output:
            A sanitized version of in_dict.
        """
        out_dict = {k: v for k, v in in_dict.items() if type(v) in (torch.Tensor, list, float, int, Tuple)}
        out_dict = detach_and_clone(out_dict)
        if to_out_device:
            out_dict = move_to(out_dict, self._out_device)
        return out_dict
