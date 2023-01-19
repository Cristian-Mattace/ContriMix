from abc import ABC
from abc import abstractmethod
import numpy as np
import torch
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
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

    def metadata_to_group(self, metadata: torch.Tensor, return_counts: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass





