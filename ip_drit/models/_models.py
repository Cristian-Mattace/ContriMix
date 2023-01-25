from abc import abstractmethod
from enum import auto
from enum import Enum
from typing import List
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from ._core import Downsampling2xkWithSkipConnection
from ._core import LeakyReLUConv2d
from ._core import ReLUInstNorm2dConv2d
from ._core import ReLUUpsample2xWithConvTranspose2d
from ._core import ReLUUpsample2xWithInterpolation
from ._core import ResInstNorm2dConv2d
from ._initializer import Initializer


class AttributeEncoder(nn.Module):
    """A class that estimate the stain matrix i.e. attribute encoder.

    The networks takes an input image returns a stain vector matrix of size N x (3k^2) x num_stain_vectors.
    To generate an output absorbance image, we need to reshape the stain vector into size
    (3k^2) * num_stain_vectors and multiply it to a content matrix of num_stain_vectors x H x W to obtain
    an output image of size 3k^2 * H x W. Then, using the pixel shuffling, we can reduce it to 3 x (kW) x (kH)

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        num_stain_vectors (optional): The number of stain vectors. Defaults to 32.
        down-sampling_factor (optional): A factor that describe how much of the image is down-sampled. Defaults to 4.
    Reference:
        https://github.com/HsinYingLee/MDMM/blob/master/networks.py#L64
    """

    def __init__(self, in_channels: int, out_channels: int, num_stain_vectors: int = 32) -> None:
        super().__init__()
        self._num_stain_vectors = num_stain_vectors
        self._out_channels = out_channels
        self._model = Initializer()(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=64,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    padding_mode="reflect",
                    bias=True,
                ),
                Downsampling2xkWithSkipConnection(in_channels=64, out_channels=128),
                Downsampling2xkWithSkipConnection(in_channels=128, out_channels=256),
                Downsampling2xkWithSkipConnection(in_channels=256, out_channels=512),
                Downsampling2xkWithSkipConnection(in_channels=512, out_channels=1024),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # Condense all the X, Y dimensions to 1 pixels.
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=out_channels * num_stain_vectors,
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
        x2 = x1.view(x1.size(0), self._out_channels, self._num_stain_vectors)
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


class RealFakeDiscriminator(nn.Module):
    """A discriminator that aims to classify if an image is real or generated image.

    Args:
        in_channels: The number of channels for the input image.

    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self._downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self._encoder = self._make_network(in_channels=in_channels)
        self._real_fake_predictor = Initializer()(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        )

    @staticmethod
    def _make_network(in_channels: int) -> nn.Module:
        """Make a single scale discriminator."""
        return nn.Sequential(
            LeakyReLUConv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=2,
                enable_spectral_normalization=True,
                enable_instance_norm=False,
            ),
            LeakyReLUConv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=2,
                enable_spectral_normalization=True,
                enable_instance_norm=False,
            ),
            LeakyReLUConv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=2,
                enable_spectral_normalization=True,
                enable_instance_norm=False,
            ),
            LeakyReLUConv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1,
                stride=2,
                enable_spectral_normalization=True,
                enable_instance_norm=False,
            ),
            LeakyReLUConv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                padding=1,
                stride=2,
                enable_spectral_normalization=True,
                enable_instance_norm=False,
            ),
            LeakyReLUConv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                padding=1,
                stride=2,
                enable_spectral_normalization=True,
                enable_instance_norm=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._encoder(x)
        return self._real_fake_predictor(x)


class GeneratorType(Enum):
    """An Enum class that defines what type of Generators to use."""

    WithPixelShuffle = auto()
    WithInterpolation = auto()
    WithConvTranspose2d = auto()


class AbstractImGenerator(nn.Module):
    """A abstract class that computes the product between the content and the attribute.

    This class generates a synthetic image G(z_c, z_a) from the content tensor z_c, attribute tensor z_a.
    """

    _NUM_FEATURES_PER_FRACTION = 256

    def __init__(self) -> None:
        super().__init__()

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
                f"#elements in first dimension of z_c doesn't match the #elements in the 2nd dimension of z_a!"
            )
        num_rows, num_cols = z_c.size(2), z_c.size(3)
        x = torch.bmm(z_a, z_c.view(z_c.size(0), z_c.size(1), -1))
        x = x.view(x.size(0), x.size(1), num_rows, num_cols)
        return self._up_sample_tensor(x)

    @abstractmethod
    def _up_sample_tensor(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AbsorbanceImGenerator(AbstractImGenerator):
    """A class that computes the product between the content and the attribute.

    This class generates a synthetic image G(z_c, z_a) from the content tensor z_c, attribute tensor z_a.
    This class uses PixelShuffle for upsampling to generate the output image.

    Args:
        downsampling_factor (optional): A factor that describe how much of the image is downsampled. Defaults to 2.
    """

    def __init__(self, downsampling_factor: int = 4) -> None:
        super().__init__(downsampling_factor=downsampling_factor)
        self._shuffle_layer = torch.nn.PixelShuffle(upscale_factor=downsampling_factor)

    def _up_sample_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self._shuffle_layer(x)


class AbsorbanceImGeneratorWithInterpolation(AbstractImGenerator):
    """A class that computes the product between the content and the attribute.

    This class generates a synthetic image G(z_c, z_a) from the content tensor z_c, attribute tensor z_a.
    Different from AbsorbanceImGenerator that uses PixelShuffle to upsample, this class uses Upsampling followed by
    1x convolution to reduce the number of channels.

    Args:
        in_channels: The number of input channels for the generator.
        downsampling_factor (optional): A factor that describe how much of the image is downsampled. Defaults to 2.
    """

    def __init__(self, in_channels: int, downsampling_factor: int = 4) -> None:
        super().__init__(downsampling_factor=downsampling_factor)
        self._check_downsampling_factor_be_a_power_of_2(downsampling_factor)
        num_upsampling_blocks = int(np.log2(downsampling_factor))
        upsampling_blocks: List[nn.Module] = []
        for block_idx in range(num_upsampling_blocks - 1):
            upsampling_blocks.append(
                ReLUUpsample2xWithInterpolation(
                    in_channels=in_channels // 2**block_idx, out_channels=in_channels // 2 ** (block_idx + 1)
                )
            )

        upsampling_blocks.append(
            ReLUUpsample2xWithInterpolation(in_channels=in_channels // 2 ** (num_upsampling_blocks - 1), out_channels=3)
        )
        self._upsample_block = nn.Sequential(*upsampling_blocks)

    @staticmethod
    def _check_downsampling_factor_be_a_power_of_2(fact: int) -> None:
        log2 = np.log2(fact)
        if np.abs(log2 - float(int(log2))) > 1e-3:
            raise ValueError(f"Downsampling factor {fact} is not a power of 2!")

    def _up_sample_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self._upsample_block(x)


class AbsorbanceImGeneratorWithConvTranspose(AbstractImGenerator):
    """A class that computes the product between the content and the attribute.

    This class generates a synthetic image G(z_c, z_a) from the content tensor z_c, attribute tensor z_a.
    This class uses ConvTranspose2D to do the upsampling.

    Args:
        in_channels: The number of input channels for the generator.
        downsampling_factor (optional): A factor that describe how much of the image is downsampled. Defaults to 2.
    """

    def __init__(self, in_channels: int, downsampling_factor: int = 4) -> None:
        super().__init__(downsampling_factor=downsampling_factor)
        self._check_downsampling_factor_be_a_power_of_2(downsampling_factor)
        num_upsampling_blocks = int(np.log2(downsampling_factor))
        upsampling_blocks: List[nn.Module] = []
        for block_idx in range(num_upsampling_blocks - 1):
            upsampling_blocks.append(
                ReLUUpsample2xWithConvTranspose2d(
                    in_channels=in_channels // 2**block_idx, out_channels=in_channels // 2 ** (block_idx + 1)
                )
            )
        upsampling_blocks.append(
            ReLUUpsample2xWithConvTranspose2d(
                in_channels=in_channels // 2 ** (num_upsampling_blocks - 1), out_channels=3
            )
        )
        self._upsample_block = nn.Sequential(*upsampling_blocks)

    @staticmethod
    def _check_downsampling_factor_be_a_power_of_2(fact: int) -> None:
        log2 = np.log2(fact)
        if np.abs(log2 - float(int(log2))) > 1e-3:
            raise ValueError(f"Downsampling factor {fact} is not a power of 2!")

    def _up_sample_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self._upsample_block(x)
