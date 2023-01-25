from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch_scatter
from pandas.api.types import CategoricalDtype


def minimum(
    numbers: Union[torch.Tensor, np.ndarray, Iterable], ret_val_when_empty: float = 0.0
) -> Union[torch.Tensor, np.ndarray, Iterable]:
    """Returns the minimum value of a tensor, an array, or an iterable.

    Args:
        numbers: The tensor or an array or an iterable to find the min from.
        ret_val_when_empty: The return value when 'numbers' is empty.

    Returns:
        A value for the minimum for numbers with the same type as `numbers`.
    """
    if isinstance(numbers, torch.Tensor):
        if numbers.numel() == 0:
            return torch.tensor(ret_val_when_empty, device=numbers.device)
        else:
            return numbers[~torch.isnan(numbers)].min()
    elif isinstance(numbers, np.ndarray):
        if numbers.size == 0:
            return np.array(ret_val_when_empty)
        else:
            return np.nanmin(numbers)
    else:
        if len(numbers) == 0:
            return ret_val_when_empty
        else:
            return min(numbers)


def maximum(
    numbers: Union[torch.Tensor, np.ndarray, Iterable], ret_val_when_empty=0.0
) -> Union[torch.Tensor, np.ndarray, Iterable]:
    """Returns the maximum value of a tensor, an array, or an iterable.

    Args:
        numbers: The tensor or an array or an iterable to find the max from.
        ret_val_when_empty: The return value when 'numbers' is empty.

    Returns:
        A value for the maximum for numbers with the same type as `numbers`.
    """
    if isinstance(numbers, torch.Tensor):
        if numbers.numel() == 0:
            return torch.tensor(ret_val_when_empty, device=numbers.device)
        else:
            return numbers[~torch.isnan(numbers)].max()
    elif isinstance(numbers, np.ndarray):
        if numbers.size == 0:
            return np.array(ret_val_when_empty)
        else:
            return np.nanmax(numbers)
    else:
        if len(numbers) == 0:
            return ret_val_when_empty
        else:
            return max(numbers)


def split_into_groups(g: torch.Tensor) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
    """Splits a tensor into multiple group.

    Args:
        g: A tensor to split/

    Returns:
        A unique `groups` present in g
        A List of Tensors for the group index, where the i-th tensor is the indices of the elements of g that equal
        groups[i]. Has the same length as len(groups).
        A tensor of element count in groups. It has the same length as len(groups).
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts


def get_counts(g: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Returns a count tensor for `n_groups` groups.

    This differs from split_into_groups() in how it handles missing groups. This function always returns a count Tensor
    of length n_groups, whereas split_into_groups() returns a unique_counts Tensor whose length is the number of unique
    groups present in g.

    Args:
        g: A tensor of groups.
        n_groups: The number of groups to get the count.

    Returns:
        A tensor that contains the length count of each group.
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    counts = torch.zeros(n_groups, device=g.device)
    counts[unique_groups] = unique_counts.float()
    return counts


def avg_over_groups(v: torch.Tensor, g: torch.Tensor, n_groups: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the average over the groups.

    Args:
        v: A tensor that contains the quantity to average over.
        g: A tensor of the same length as v, containing group information.
        n_groups: The number of groups to average over.

    Returns:
        A tensor of group average of length n_groups.
        A tensor of group counts
    """
    assert v.device == g.device
    assert v.numel() == g.numel()
    group_count = get_counts(g, n_groups)
    group_avgs = torch_scatter.scatter(src=v, index=g, dim_size=n_groups, reduce="mean")
    return group_avgs, group_count


def threshold_at_recall(y_pred: torch.Tensor, y_true: torch.Tensor, global_recall: float = 60.0) -> float:
    """Calculate the model threshold to use to achieve a desired global_recall level.

    Args:
        y_true: A vector of the true binary labels.
    """
    return np.percentile(y_pred[y_true == 1], 100 - global_recall)


def numel(obj: Union[torch.Tensor, List]) -> int:
    """Get the number of elements of an object."""
    if torch.is_tensor(obj):
        return obj.numel()
    elif isinstance(obj, list):
        return len(obj)
    else:
        raise TypeError("Invalid type for numel")
