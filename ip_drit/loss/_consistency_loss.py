"""A module that implements the consistency loss."""
from abc import abstractmethod
from enum import auto
from enum import Enum

import torch
import torch.nn as nn
from typing import Optional
from typing import List
from ._perceptual_loss import PerceptualLoss


class ImageConsistencyLossType(Enum):
    """An enum class that defines the type of the image consistency loss."""

    L1ImageConsistencyLoss = auto()
    LPipsImageConsistencyLoss = auto()


class AbstractImageConsistencyLoss(nn.Module):
    """A class that implement the image consistency loss."""

    def __init__(self, transforms: Optional[List[nn.Module]] = None):
        super().__init__()

    @abstractmethod
    def forward(self, im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
        # TODO: apply the transforms here!
        raise NotImplementedError


class L1ImageConsistencyLoss(AbstractImageConsistencyLoss):
    """An image consistency loss based on the L1 loss."""

    def __init__(self, transforms: Optional[List[nn.Module]]=None):
        super().__init__(transforms)
        self._loss = nn.L1Loss(reduction="sum")

    def forward(self, im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
        return self._loss(im1, im2) / (im1.size(1) * im1.size(2) * im1.size(3))


class LPipsImageConsistencyLoss(AbstractImageConsistencyLoss):
    """An image consistency loss based on the Perceptual loss."""

    def __init__(self, transforms: Optional[List[nn.Module]]=None):
        super().__init__(transforms)
        self._loss = PerceptualLoss()

    def forward(self, im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
        return self._loss(im1, im2) / im1.size(1)
