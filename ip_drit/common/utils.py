import torch
from typing import Any
from typing import List
from typing import Union

def numel(obj: Union[torch.Tensor, List[Any]]) -> int:
    """Gets the length of an object.

    Args:
        obj: The object to get the length from.
    """
    if torch.is_tensor(obj):
        return obj.numel()
    elif isinstance(obj, list):
        return len(obj)
    else:
        raise TypeError("Invalid type for numel")