from typing import Any
from typing import Dict
from typing import List
from typing import Union

import torch


def move_to(obj: Union[Dict, List, float, int], device: str) -> Union[Union[dict, List[None], int], Any]:
    """Moves an object named `obj` to a device specified by a `device` string."""
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        # Assume obj is a Tensor or other type
        # (like Batch, for MolPCBA) that supports .to(device)
        return obj.to(device)


def detach_and_clone(obj: Union[torch.Tensor, Dict, List, float, int]) -> Union[torch.Tensor, Dict, List, float, int]:
    """Detaches an object from the computing graph, clones, and returns it."""
    if torch.is_tensor(obj):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: detach_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_clone(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        raise TypeError("Invalid type for detach_and_clone")
