"""A class that defines the transformation between absorbance and transmission."""
from enum import auto
from enum import Enum
from typing import Tuple

import torch


class SignalType(Enum):
    """An enum class that defines the type of the tensor."""

    TRANS = auto()
    ABS = auto()


class AbsorbanceToTransmittance(object):
    """Converts an absorbance image to the transmittance image."""

    def __call__(self, im_and_sig_type: Tuple[torch.Tensor, SignalType]) -> Tuple[torch.Tensor, SignalType]:
        im, sig_type = im_and_sig_type
        if sig_type != SignalType.ABS:
            raise RuntimeError(f"Can't convert a none absorbance signal. The current signal type is {sig_type}!")
        # im = torch.clip(im, 0.0, None)
        return 10 ** (-im), SignalType.TRANS


class TransmittanceToAbsorbance(object):
    """Converts a transmittance image to the absorbance image."""

    def __call__(
        self, im_and_sig_type: Tuple[torch.Tensor, SignalType], min_trans_signal=1e-6
    ) -> Tuple[torch.Tensor, SignalType]:
        im, sig_type = im_and_sig_type
        if sig_type != SignalType.TRANS:
            raise RuntimeError(f"Can't convert a non-transmission signal. The current signal type is {sig_type}")
        im = torch.clip(im, min_trans_signal, None)
        return -torch.log10(im), SignalType.ABS
