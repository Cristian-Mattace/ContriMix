"""A module that defines the transformations on patches."""
import numpy as np
import torch
from skimage.transform import rotate


class RandomFlip(object):
    """Flips the images up/down, left/right randomly with probability of 0.5."""

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        im = sample
        if np.random.uniform(low=0.0, high=1.0) > 0.5:
            im = np.fliplr(im)
        return im


class RandomRotate(object):
    """Randomly rotates the image."""

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        rotation_angle_degs = float(np.random.randint(4) * 90.0)
        im = rotate(sample.astype(np.float32), rotation_angle_degs)
        return im


class RGBToTransmittance(object):
    """Converts an RGB image to a transmittance image."""

    def __init__(self, min_transmittance: float = 1e-4, clip_max_transmittance_to_one: bool = True):
        self._min_transmittance: float = min_transmittance
        self._max_transmittance = 1.0 if clip_max_transmittance_to_one else None

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        trans_im = sample / np.percentile(sample, q=99.0, axis=(0, 1))
        trans_im = np.maximum(trans_im, self._min_transmittance)
        trans_im = np.minimum(trans_im, self._max_transmittance)
        return trans_im


class TransmittanceToRGB(object):
    """Converts the transmittance RGB assuming the 0 transmittance is 1.0 in RGB."""

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        return 10 ** (-sample)


class TransmittanceToAbsorbance(object):
    """Converts a transmittance image to the absorbance image."""

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        return -np.log10(sample)


class AbsorbanceToTransmittance(object):
    """Converts an absorbance image to the transmittance image."""

    def __call__(self, im: np.ndarray) -> np.ndarray:
        im = np.clip(im, 0.0, None)
        return 10 ** (-im)


class ToTensor(object):
    """Converts to tensor, changes the order from (H, W, C) to (C, H, W)."""

    def __call__(self, sample: np.ndarray) -> torch.Tensor:
        return torch.tensor(sample.astype(np.float32).copy().transpose(2, 0, 1))
