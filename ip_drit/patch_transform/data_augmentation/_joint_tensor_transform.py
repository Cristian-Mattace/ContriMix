"""This modules defines the joint (input, target) transform."""
from abc import ABC
from abc import abstractclassmethod
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


class AbstractJointTensorTransform(ABC):
    """A class that jointly transforms the input tensor and the target tensor."""

    def __init__(self, num_classes: int) -> None:
        self._num_classes: int = num_classes

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transforms the input tensor and the target tensor.

        Returns:
            x: The transformed input tensor.
            y: The transformed tensor of the class probalities.
        """
        if len(y.shape) == 1:
            # Convert to 1-hot tensor
            y = F.one_hot(y, num_classes=self._num_classes).float()
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

    def __init__(self, num_classes: int, alpha: float = 1.0, exclude_num_first_samples: Optional[int] = None) -> None:
        super().__init__(num_classes=num_classes)
        self._alpha: float = alpha
        if exclude_num_first_samples is None:
            exclude_num_first_samples = 0
        self._exclude_num_first_samples = exclude_num_first_samples

    def _transform(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_ims, _, h, w = x.shape

        # Mixed up images
        x_used, y_used = x[self._exclude_num_first_samples :], y[self._exclude_num_first_samples :]
        num_mixed_up_ims = n_ims - self._exclude_num_first_samples
        permuted_x_indices = torch.randperm(num_mixed_up_ims, device=x.device)
        lambds = np.random.beta(self._alpha, self._alpha, size=num_mixed_up_ims)
        cols = np.random.uniform(0, w, size=num_mixed_up_ims)
        rows = np.random.uniform(0, h, size=num_mixed_up_ims)
        mix_hs = (1 - lambds) ** 0.5 * h
        mix_ws = (1 - lambds) ** 0.5 * w

        r1s = np.floor(np.maximum(0, rows - mix_hs / 2)).astype(int)
        r2s = np.floor(np.minimum(h, rows + mix_hs / 2)).astype(int)
        c1s = np.floor(np.maximum(0, cols - mix_ws / 2)).astype(int)
        c2s = np.floor(np.minimum(w, cols + mix_ws / 2)).astype(int)

        valid_indices = np.where(np.logical_and(r1s < r2s, c1s < c2s))[0]

        x_out = x_used.clone()
        y_out = y_used.clone()
        for idx in valid_indices:
            r1, r2, c1, c2 = r1s[idx], r2s[idx], c1s[idx], c2s[idx]
            perm_idx = permuted_x_indices[idx]
            x_out[idx, :, r1:r2, c1:c2] = x_out[perm_idx, :, r1:r2, c1:c2]
            mix_cof = 1 - (r2 - r1) * (c2 - c1) / (h * w)
            y_out[idx] = y_out[idx] * mix_cof + y_out[perm_idx] * (1 - mix_cof)
        x[self._exclude_num_first_samples :], y[self._exclude_num_first_samples :] = x_out, y_out
        return x, y
