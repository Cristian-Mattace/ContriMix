from abc import ABC
from abc import abstractmethod
import copy
import numpy as np
import torch
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from ip_drit.datasets import AbstractPublicDataset
from ip_drit.datasets import SubsetPublicDataset

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
    def metadata_to_group(self, metadata: torch.Tensor, return_counts: bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        Args:
            metadata: An n x d matrix containing d metadata fields for n different points.
            return_counts: If True, return group counts as well.
        Output:
            If return_counts == True:
                An n-length vector of groups.
                An n_group-length vector of integers containing the numbers of data points in each group.
            else:
                An n-length vector of groups
        """
        raise NotImplementedError


class CombinatorialGrouper(AbstractGrouper):
    """
    A grouper that form groups by taking all possible combinations of the groupby_fields in lexicographical order.

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
        groupby_fields (optional): A list of string that specifies what fiels to group. Defaults to None, in which case,
            all data points are assigned to group 0.
    """
    def __init__(self, dataset: AbstractPublicDataset, groupby_fields: Optional[List[str]] = None) -> None:
        super().__init__()

        datasets: List[AbstractPublicDataset] = [dataset,]
        metadata_fields: List[str] = datasets[0].metadata_fields
        self._groupby_fields = groupby_fields

        if groupby_fields is None:
            self._n_groups = 1
        else:
            self._groupby_field_indices = [i for (i, field) in enumerate(metadata_fields) if field in groupby_fields]
            metadata_array = torch.cat([dataset.metadata_array for dataset in datasets])
            grouped_metadata = metadata_array[:, self._groupby_field_indices]

            if not isinstance(grouped_metadata, torch.LongTensor):
                grouped_metadata = grouped_metadata.long()

            for idx, field in enumerate(self._groupby_fields):
                min_value = grouped_metadata[:, idx].min()
                if min_value < 0:
                    raise ValueError(f"Metadata for CombinatorialGrouper cannot have values less than 0: {field}, {min_value}")
                if min_value > 0:
                    logging.warn(f"Minimum metadata value for CombinatorialGrouper is not 0 ({field}, {min_value}). This will result in empty groups")

            # We assume that the metadata fields are integers,
            # so we can measure the cardinality of each field by taking its max + 1, where 1 is for the first group of 0
            # Note that this might result in some empty groups.
            assert grouped_metadata.min() >= 0, "Group numbers cannot be negative."
            self._cardinality = 1 + torch.max(grouped_metadata, dim=0)[0]
            cumprod = torch.cumprod(self._cardinality, dim=0)
            self._n_groups = cumprod[-1].item()
            self._factors_np = np.concatenate(([1], cumprod[:-1]))
            self._factors = torch.from_numpy(self._factors_np)
            self._metadata_map = self._build_largest_metadata_map(datasets=datasets)

    @staticmethod
    def _build_largest_metadata_map(datasets: List[AbstractPublicDataset]) -> Dict[str, Union[List, np.ndarray]]:
        # Build the largest metadata_map to see to check if all the metadata_maps are subsets of each other
        metadata_fields: List[str] = datasets[0].metadata_fields
        largest_metadata_map: Dict[str, Union[List, np.ndarray]] = copy.deepcopy(datasets[0].metadata_map)

        for i, dataset in enumerate(datasets):
            # The first dataset was used to get the metadata_fields and initial metadata_map
            if i == 0:
                continue

            if dataset.metadata_fields != metadata_fields:
                raise ValueError(
                    f"The datasets passed in have different metadata_fields: {dataset.metadata_fields}. "
                    f"Expected: {metadata_fields}"
                )

            if dataset.metadata_map is None:
                continue

            for field, values in dataset.metadata_map.items():
                n_overlap = min(len(values), len(largest_metadata_map[field]))
                if not (np.asarray(values[:n_overlap]) == np.asarray(largest_metadata_map[field][:n_overlap])).all():
                    raise ValueError("The metadata_maps of the datasets need to be ordered subsets of each other.")

                if len(values) > len(largest_metadata_map[field]):
                    largest_metadata_map[field] = values

        return largest_metadata_map

    def metadata_to_group(self, metadata: torch.Tensor, return_counts: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._groupby_fields is None:
            groups = torch.zeros(metadata.shape[0], dtype=torch.long)
        else:
            groups = metadata[:, self._groupby_field_indices].long() @ self._factors

        if return_counts:
            return groups, get_counts(groups, self._n_groups)
        else:
            return groups

def get_counts(g: torch.Tensor, n_groups: int) -> List[int]:
    """
    This differs from split_into_groups in how it handles missing groups.
    get_counts always returns a count Tensor of length n_groups,
    whereas split_into_groups returns a unique_counts Tensor
    whose length is the number of unique groups present in g.

    Args:
        g: Vector of groups
    Returns:
        A list of length n_groups, denoting the count of each group.
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    counts = torch.zeros(n_groups, device=g.device)
    counts[unique_groups] = unique_counts.float()
    return counts




