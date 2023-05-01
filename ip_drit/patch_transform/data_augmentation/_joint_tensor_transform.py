"""This modules defines the joint (input, target) transform."""
from abc import ABC
from abc import abstractclassmethod
from typing import Tuple

import numpy as np
import torch


class AbstractJointTensorTransform(ABC):
    """A class that jointly transforms the input tensor and the target tensor."""

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transforms the input tensor and the target tensor.

        Returns:
            x: The transformed input tensor.
            y: The transformeed tensor of the class probalities.
        """
        if len(y.shape) != 2:
            raise ValueError(
                f"Can't transform the label y because it is only a 1 dimension tensor. "
                + "To fix this, set return_one_hot=True in the Dataset initialization!"
            )
        return self._transform(x, y)

    @abstractclassmethod
    def _transform(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class CutMixJointTensorTransform(AbstractJointTensorTransform):
    """A class that performs the CutMix operation.

    Args:
        alpha: The factor alpha to sample the mixing coefficient. Defaults to 1.0

    References:
        [1]. CutMix: Regularization strategy to train strong classifiers with localizable features.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self._alpha: float = alpha

    def _transform(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_ims, _, h, w = x.shape
        permuted_x_indices = torch.randperm(n_ims, device=x.device)
        lambd = np.random.beta(self._alpha, self._alpha)
        col = np.random.uniform(0, w)
        row = np.random.uniform(0, h)
        mix_h = (1 - lambd) ** 0.5 * h
        mix_w = (1 - lambd) ** 0.5 * w

        r1 = round(max(0, row - mix_h / 2))
        r2 = round(min(h, row + mix_h / 2))
        c1 = round(max(0, col - mix_w / 2))
        c2 = round(max(w, col + mix_w / 2))

        if r1 < r2 and c1 < c2:
            x_out = x.clone()
            y_out = y.clone()
            x_out[:, :, r1:r2, c1:c2] = x[permuted_x_indices, :, r1:r2, c1:c2]
            mix_cof = 1 - (r2 - r1) * (c2 - c1) / (h * w)
            y_out = y * mix_cof + y[permuted_x_indices] * (1 - mix_cof)
            return x_out, y_out
        return x, y
