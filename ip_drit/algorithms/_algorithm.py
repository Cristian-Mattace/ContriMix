import torch
import torch.nn as nn
from typing import Any
from typing import Tuple
from typing import Dict
from ._utils import move_to, detach_and_clone

class Algorithm(nn.Module):
    """
    A class that defines the algorithm to run on.
    """
    def __init__(self, device: torch.device):
        super().__init__()
        self._device = device
        self._out_device = 'cpu'
        self._has_log = False
        self.reset_log()

    def update(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process the batch, update the log, and update the model
        Args:
            - batch: A batch of data yielded by data loaders
        Output:
            - results: A dictionary of metadata about the batch.
        """
        raise NotImplementedError

    def evaluate(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process the batch and update the log, without updating the model
        Args:
            - batch: A batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
        """
        raise NotImplementedError

    def train(self, mode=True):
        """
        Switch to train mode
        """
        self._is_training = mode
        super().train(mode)
        self.reset_log()

    @property
    def has_log(self) -> bool:
        return self._has_log

    def reset_log(self) -> None:
        """
        Resets log by clearing out the internal log, Algorithm.log_dict
        """
        self._has_log = False
        self._log_dict = {}

    def _update_log(self, results: Dict[str, Any]) -> None:
        """
        Updates the internal log, Algorithm.log_dict
        Args:
            - results (dictionary)
        """
        raise NotImplementedError

    def _get_log(self) -> Dict[str, Any]:
        """
        Sanitizes the internal log (Algorithm.log_dict) and outputs it.
        """
        raise NotImplementedError

    def get_pretty_log_str(self) -> str:
        raise NotImplementedError

    def step_schedulers(self, is_epoch: bool, metrics={}, log_access=False) -> None:
        """
        Update all relevant schedulers
        Args:
            - is_epoch (bool): epoch-wise update if set to True, batch-wise update otherwise
            - metrics (dict): a dictionary of metrics that can be used for scheduler updates
            - log_access (bool): whether metrics from self.get_log() can be used to update schedulers
        """
        raise NotImplementedError

    def sanitize_dict(self, in_dict: Dict[str, Any], to_out_device: bool=True) -> None:
        """
        Helper function that sanitizes dictionaries by:
            - moving to the specified output device
            - removing any gradient information
            - detaching and cloning the tensors
        Args:
            - in_dict (dictionary)
        Output:
            - out_dict (dictionary): sanitized version of in_dict
        """
        out_dict = detach_and_clone(in_dict)
        if to_out_device:
            out_dict = move_to(out_dict, self._out_device)
        return out_dict


    def reset_log(self) -> None:
        """
        Resets log by clearing out the internal log, Algorithm.log_dict
        """
        self._has_log = False
        self._log_dict = {}