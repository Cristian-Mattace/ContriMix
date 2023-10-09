import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from ._initializer import InitializationType
from ._initializer import Initializer


class LeakyReLUConv2d(nn.Module):
    """A class that defines a Conv2d follows by Leaky ReLU.

        [Conv2d with reflection pad with optional spectral normalization] -> [LeakyReLU].

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kernel_size: The size of the 2D convolutional kernel.
        stride: The stride of the 2D convolutional kernel.
        padding: The size of the padding.
        enable_spectral_normalization (optional): If True, the spectral normalization will be performed.
            Defaults to False. See https://arxiv.org/pdf/1802.05957.pdf for relating details.
        enable_instance_norm (optional): If True, instance normalization will be used. Defaults to False.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        enable_spectral_normalization: bool = False,
        enable_instance_norm: bool = False,
    ) -> None:
        super().__init__()
        conv2d_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="reflect",
            bias=True,
        )

        if enable_spectral_normalization:
            conv2d_layer = spectral_norm(conv2d_layer)

        layers = [conv2d_layer]

        if enable_instance_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=False))
        layers.append(nn.LeakyReLU(inplace=True))

        self._model = nn.Sequential(*layers)
        self._model = Initializer(init_type=InitializationType.NORMAL)(self._model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class ReLUInstNorm2dConv2d(nn.Module):
    """A class that defines the following transformation.

        [Conv2d with reflection padding] -> [InstanceNorm2d] -> [ReLU]

    Args;
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kernel_size: The size of the 2D convolutional kernel.
        stride: The stride of the 2D convolutional kernel.
        padding: The size of the padding.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self._model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="reflect",
                bias=True,
            ),
            nn.InstanceNorm2d(num_features=out_channels, affine=False),
            nn.LeakyReLU(inplace=True),
        )
        self._model = Initializer(init_type=InitializationType.NORMAL)(self._model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class ResInstNorm2dConv2d(nn.Module):
    """A class that defines the residual condition block that has the following architecture.

       - -> [Conv2d(3x3) -> [InstanceNorm2d] -> [ReLU()] -> [Conv2d(3x3)] -> [InstanceNorm2d] -> + ->
        |                                                                                        ^
        |----------------------------------------------------------------------------------------|

    The numbered of input and output channels are the same.

    Args;
       in_channels: The number of input channels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self._model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=True,
            ),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=True,
            ),
            nn.InstanceNorm2d(in_channels),
        )
        self._model = Initializer(init_type=InitializationType.NORMAL)(self._model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._model(x)


class Downsampling2xkWithSkipConnection(nn.Module):
    """Downsampling by 2x with skip connection block.

    This block implements the following architecture
    x -> -> LeakyReLU -> Conv2d -> LeakyReLU -> Conv2d -> AvgPool2d -> + -> output
        |                                                              ^
        |-------------> AvgPool2d -> Conv2d ---------------------------|
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
    Reference:
        https://github.com/HsinYingLee/MDMM/blob/18360fe3fa37dde28c70c5a945ec783e44eb72ed/networks.py#L334
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self._forward_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=True,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=True,
            ),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self._skip_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_block(x) + self._skip_block(x)


class ReLUUpsample2xWithInterpolation(nn.Module):
    """Upsampling by 2x using the Upsample block, followed by a 2D convolution with a (1x1) kernel."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self._forward_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="reflect",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_block(x)


class ReLUUpsample2xWithConvTranspose2d(nn.Module):
    """Upsampling by 2x using the Convolutional Transpose 2d."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self._model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0, output_padding=0
            ),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
