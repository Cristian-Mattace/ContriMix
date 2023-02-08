from abc import abstractmethod
from enum import auto
from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn

from ._contrimix_core import Downsampling2xkWithSkipConnection
from ._contrimix_core import LeakyReLUConv2d
from ._contrimix_core import ReLUInstNorm2dConv2d
from ._contrimix_core import ResInstNorm2dConv2d
from ._initializer import Initializer
from ._trans_abs_converters import AbsorbanceToTransmittance
from ._trans_abs_converters import TransmittanceToAbsorbance


class AttributeEncoder(nn.Module):
    """A class that estimate the stain matrix.

    The networks takes an input image returns a stain vector matrix of size N x (3k^2) x num_stain_vectors.
    To generate an output absorbance image, we need to resphae the stain vector into size
    (3k^2) * num_stain_vectors and multiply it to a content matrix of num_stain_vectors x H x W to obtain an output
    image of size 3k^2 * H x W. Then, using the pixel shuffling, we can reduce it to 3 x (kW) x (kH).

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels for the attribute vector.
        downsampling_factor (optional): A factor that describe how much of the image is downsampled. Defaults to 4.
    Reference:
        https://github.com/HsinYingLee/MDMM/blob/master/networks.py#L64
    """

    def __init__(self, in_channels: int, num_stain_vectors: int) -> None:
        super().__init__()
        self.needs_y_input: bool = False
        self._num_stain_vectors = num_stain_vectors
        self._three_times_k_sqr = 3 * 4**2
        self._model = Initializer()(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=16,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    padding_mode="reflect",
                    bias=True,
                ),
                Downsampling2xkWithSkipConnection(in_channels=16, out_channels=32),
                Downsampling2xkWithSkipConnection(in_channels=32, out_channels=64),
                Downsampling2xkWithSkipConnection(in_channels=64, out_channels=128),
                Downsampling2xkWithSkipConnection(in_channels=128, out_channels=256),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # Condense all the X, Y dimensions to 1 pixels.
                nn.Conv2d(
                    in_channels=256,
                    out_channels=self._num_stain_vectors * self._three_times_k_sqr,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.LeakyReLU(inplace=False),
            )
        )

    def forward(self, ims: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the mean and log of the variance for the conditional distribution z^a|ims."""
        x1 = self._model(ims)
        x2 = x1.view(x1.size(0), self._three_times_k_sqr, self._num_stain_vectors)
        return x2


class ContentEncoder(nn.Module):
    """A class that defines the architecture for the content encoder.

    This architecture of this model is:
        [LeakyReLUConv2d] -> 3 x [ReLUInstanceNormConv2d] -> 3 x [ResInstNorm2dConv2d]

    Args:
        in_channels: The number of input channels.

    Returns:
        For each minibatch with N samples, it returns a tensor of size N x out_channels x H x W in which
        H and W is a down-sampled version of the input image.
    """

    def __init__(self, in_channels: int, num_stain_vectors: int = 32) -> None:
        super().__init__()
        self.needs_y_input: bool = False
        self._model = nn.Sequential(
            LeakyReLUConv2d(in_channels=in_channels, out_channels=16, kernel_size=7, stride=1, padding=3),
            ReLUInstNorm2dConv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLUInstNorm2dConv2d(in_channels=32, out_channels=num_stain_vectors, kernel_size=3, stride=2, padding=1),
            ResInstNorm2dConv2d(in_channels=num_stain_vectors),
            ResInstNorm2dConv2d(in_channels=num_stain_vectors),
            ResInstNorm2dConv2d(in_channels=num_stain_vectors),
            ResInstNorm2dConv2d(in_channels=num_stain_vectors),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class AbsorbanceImGenerator(nn.Module):
    """A class that computes the product between the content and the attribute.

    This class generates a synthetic image G(z_c, z_a) from the content tensor z_c, attribute tensor z_a.

    Args:
        out_channels: The number of output channels for the generator.
        downsampling_factor (optional): A factor that describe how much of the image is downsampled. Defaults to 2.
    """

    def __init__(self) -> None:
        super().__init__()
        self.needs_y_input: bool = True
        self._shuffle_layer = torch.nn.PixelShuffle(upscale_factor=4)

    def forward(self, z_c: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """Generates an image based by the content and the attribute tensor.

        Args:
            z_c: The content image tensor, which contains abundance information of the stain. The size should be of size
                (N, num_stain_vectors, H, W).
            z_a: The attribute image tensor. The size should be of size (N, (3k^2), num_stain_vectors)

        Returns:
            The generated image tensor of size (N, 3, k*H, k*W)
        """
        if z_c.size(1) != z_a.size(2):
            raise ValueError(
                f"The number of elements in first dimension of z_c must match the number of elements in the 2nd "
                + "dimension of z_a"
            )
        num_rows, num_cols = z_c.size(2), z_c.size(3)
        x = torch.bmm(z_a, z_c.view(z_c.size(0), z_c.size(1), -1))
        x = x.view(x.size(0), x.size(1), num_rows, num_cols)
        return self._shuffle_layer(x)
