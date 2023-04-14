"""A class that defines the transformation between absorbance and transmission."""
from enum import auto
from enum import Enum
from typing import Tuple

import torch


class SignalType(Enum):
    """An enum class that defines the type of the tensor."""

    TRANS = auto()
    ABS = auto()


_MAX_ABSORBANCE_VALUE = 6.0
_MIN_ABSORBANCE_VALUE = -0.2  # Using a little bit of negative for stability during training.

# This small neagive is to replace the value of 0.0. While 0.0 physically makes sense. This lower bound acts like a
# ReLU, causing dead gradients. This prevents backprop to correct aborbance pixel values that are negative. Without
# being updated to be correct, the self-reconstruction loss stay plateau because of negative pixels. A small negative
# value fixes it.
_MIN_ABSORBANCE_VALUE = -0.2


class AbsorbanceToTransmittance(object):
    """Converts an absorbance image to the transmittance image."""

    def __call__(self, im_abs: torch.Tensor) -> torch.Tensor:
        im_abs = torch.clip(im_abs, _MIN_ABSORBANCE_VALUE, _MAX_ABSORBANCE_VALUE)
        return 10 ** (-im_abs)


class TransmittanceToAbsorbance(object):
    """Converts a transmittance image to the absorbance image."""

    def __call__(self, im_trans: torch.Tensor, min_trans_signal=10 ** (-_MAX_ABSORBANCE_VALUE)) -> torch.Tensor:
        im_trans = torch.clip(im_trans, min_trans_signal, None)
        return -torch.log10(im_trans)
