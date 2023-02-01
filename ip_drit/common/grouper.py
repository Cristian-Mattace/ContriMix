"""A module that defines the grouper."""
import copy
import logging
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch

from ip_drit.datasets import AbstractPublicDataset


class AbstractGrouper(ABC):
    """A class that group data points together based on their metadata.

    This group is used in training and evaluation to measure the accuracies of different group
    of data.
    """

    def __init__(self) -> None:
        self._n_groups = 0

    @property
    def n_groups(self) -> int:
        return self._n_groups

    @abstractmethod
    def metadata_to_group(
        self, metadata: torch.Tensor, return_counts: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Converts the metadata tensor to the group index.

        Args:
            metadata: An n x d matrix containing the (integer) values for d metadata fields for n different data points.
            return_counts: If True, return group counts as well.

        Returns:
            An n-length vector of the groups indices for all the datapoints.

            If return_counts == True, further returns
                An n_group-length vector of integers containing the numbers of data points in each group.
        """
        raise NotImplementedError


class CombinatorialGrouper(AbstractGrouper):
    """A grouper that form groups by taking all possible combinations of the groupby_fields in lexicographical order.

    For example, if:
        dataset.metadata_fields = ['country', 'time', 'y']
        groupby_fields = ['country', 'time']

    and if in dataset.metadata, country is in {0, 1} and time is in {0, 1, 2},
    then the grouper will assign groups in the following way:
        country = 0, time = 0 -> group 0
        country = 1, time = 0 -> group 1
        country = 0, time = 1 -> group 2
        country = 1, time = 1 -> group 3
        country = 0, time = 2 -> group 4
        country = 1, time = 2 -> group 5


    Args:
        dataset: A dataset that contains the all data points.
        groupby_fields (optional): A list of strings that specifies what fields to group. Defaults to None, in which
            case all data points are assigned to group 0.
    """

    def __init__(self, dataset: AbstractPublicDataset, groupby_fields: Optional[List[str]] = None) -> None:
        super().__init__()

        self._groupby_fields = groupby_fields
        if groupby_fields is None:
            self._n_groups = 1
        else:
            self._groupby_field_indices: List[int] = [
                i for (i, field) in enumerate(dataset.metadata_fields) if field in groupby_fields
            ]
            grouped_metadata = dataset.metadata_array[:, self._groupby_field_indices]
            for idx, field in enumerate(self._groupby_fields):
                min_value = grouped_metadata[:, idx].min()
                if min_value != 0:
                    raise ValueError(
                        f"Metadata for CombinatorialGrouper cannot have non-zero minimum value {min_value} for the "
                        + f"field of {field}."
                    )

            # We assume that the metadata fields are integers,
            # so we can measure the cardinality of each field by taking its max + 1, where 1 is for the first group of 0
            # Note that this might result in some empty groups.
            self._cardinality = 1 + torch.max(grouped_metadata, dim=0)[0]
            cumprod = torch.cumprod(self._cardinality, dim=0)
            self._n_groups = cumprod[-1].item()

            self._meta_to_group_index_adjustment_factor = np.concatenate(([1], cumprod[:-1]))
            self._metadata_map = copy.deepcopy(dataset.metadata_map)

    def metadata_to_group(
        self, metadata: torch.Tensor, return_counts: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._groupby_fields is None:
            groups = torch.zeros(metadata.shape[0], dtype=torch.long)
        else:
            groups = metadata[:, self._groupby_field_indices].long() @ torch.from_numpy(
                self._meta_to_group_index_adjustment_factor
            )

        if return_counts:
            return groups, _get_counts(groups, self._n_groups)
        else:
            return groups

    def group_field_str(self, group: int):
        return self.group_str(group).replace("=", ":").replace(",", "_").replace(" ", "")

    def group_str(self, group: int):
        if self._groupby_fields is None:
            return "all"

        # group is just an integer, not a Tensor
        n = len(self._meta_to_group_index_adjustment_factor)
        metadata = np.zeros(n)
        for i in range(n - 1):
            metadata[i] = (
                group % self._meta_to_group_index_adjustment_factor[i + 1]
            ) // self._meta_to_group_index_adjustment_factor[i]
        metadata[n - 1] = group // self._meta_to_group_index_adjustment_factor[n - 1]
        group_name = ""
        for i in reversed(range(n)):
            meta_val = int(metadata[i])
            if self._metadata_map is not None:
                if self._groupby_fields[i] in self._metadata_map:
                    meta_val = self._metadata_map[self._groupby_fields[i]][meta_val]
            group_name += f"{self._groupby_fields[i]} = {meta_val}, "
        group_name = group_name[:-2]
        return group_name


def _get_counts(group_indices: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Count the number of datapoints for each each group.

    This differs from split_into_groups in how it handles missing groups.
    get_counts always returns a count Tensor of length n_groups,
    whereas split_into_groups returns a unique_counts Tensor
    whose length is the number of unique groups present in g.

    Args:
        group_indices: A tensor of groups indices.

    Returns:
        A tensor of length n_groups, denoting the count of each group.
    """
    unique_groups, unique_counts = torch.unique(group_indices, sorted=False, return_counts=True)
    counts = torch.zeros(n_groups, device=group_indices.device)
    counts[unique_groups] = unique_counts.float()
    return counts
