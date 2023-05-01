"""A module that defines the augmentation pipelines."""
from typing import Callable
from typing import List
from typing import Union

import numpy as np
import torch

from ._joint_tensor_transform import AbstractJointTensorTransform


class PostContrimixTransformPipeline:
    """A pipeline that defines transformations to be applied on the patches after Contrimix.

    This pipeline only applies on the x tensor, not the label y.
    """

    def __init__(self, transforms: List[Union[AbstractJointTensorTransform, Callable]]) -> None:
        self._transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self._transforms:
            x = transform(x)
        return x


class ZeroMeanUnitStdNormalizer:
    """Normalizatin transformer to transform an image so that it has zero mean, unit std."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("NormalizeToZeroMeanUnitStd can't be used for tensors without 4 dimensions [N, C, H, W]!")
        mean_val = x.mean(dim=(2, 3), keepdim=True).detach()
        std_val = x.mean(dim=(2, 3), keepdim=True).detach()
        std_val[std_val == 0.0] = 1.0
        return (x - mean_val) / std_val


class GaussianNoiseAdder:
    """Add Gaussian noise to the image.

    Args:
        noise_std: The deviation of te noise.
    """

    def __init__(self, noise_std: float) -> None:
        self._noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x
            + torch.normal(
                mean=torch.zeros_like(x, device=x.device), std=self._noise_std * torch.ones_like(x, device=x.device)
            ).detach()
        )
