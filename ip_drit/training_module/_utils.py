from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn


def load_trained_model_from_checkpoint(checkpoint_path: Path, network: nn.Module, starts_str: str) -> nn.Module:
    """Loads the model from a checkpoint.

    Args:
        checkpoint_path: The path to the check point that stored the trained model.
        network: A network to load the checkpoint parameters to.
        starts_str: A first few letters of the network variable to load the checkpoint from.

    Returns:
        The loaded model.
    """
    state_dict = torch.load(str(checkpoint_path))["state_dict"]
    state_dict = OrderedDict([k[len(starts_str) :], v] for k, v in state_dict.items() if k.startswith(starts_str))
    network.load_state_dict(state_dict)
    return network
