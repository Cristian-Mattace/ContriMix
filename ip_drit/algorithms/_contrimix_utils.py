"""A utility module for ContriMix."""
from enum import auto
from enum import Enum

import torch
import torch.nn as nn


class _InitilizationType(Enum):
    """An enum class for different types of initializer."""

    KAIMING = auto()
    XAVIER = auto()
    NORMAL = auto()


NUM_STAIN_VECTORS = 8


class _Initializer:
    """An intializer class that initializes the model weights.

    Args:
        init_type (optional): The type of the initialization. Defaults to InitilizationType.NORMAL.

    """

    def __init__(self, init_type: _InitilizationType = _InitilizationType.NORMAL, init_gain: float = 0.02) -> None:
        self._init_type = init_type
        self._init_gain = init_gain

    def __call__(self, m: nn.Module):
        m.apply(self._initialize_module)
        return m

    def _initialize_module(self, m: object) -> None:
        class_name = m.__class__.__name__
        if self._has_weights(m):
            if self._is_conv_layer(class_name) or self._is_linear_layer(m):
                if self._init_type == _InitilizationType.KAIMING:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif self._init_type == _InitilizationType.XAVIER:
                    nn.init.xavier_normal_(m.weight.data, gain=self._init_gain)
                elif self._init_type == _InitilizationType.NORMAL:
                    nn.init.normal_(m.weight.data, mean=0, std=self._init_gain)
                else:
                    raise ValueError(f"Unknown initialization type!")

                if m.bias is not None:
                    nn.init.constant_(m.bias.data, val=0)
            if self._is_batchnorm2d_layer(class_name):
                # TODO: investigate why the mean of this is set to 1.0
                nn.init.normal_(m.weight.data, mean=1.0, std=self._init_gain)
                nn.init.constant_(m.bias.data, val=0)

    @staticmethod
    def _has_weights(m: object) -> bool:
        return hasattr(m, "weights")

    @staticmethod
    def _is_conv_layer(cls_name: str) -> bool:
        return cls_name.find("Conv") != -1

    @staticmethod
    def _is_linear_layer(cls_name: str) -> bool:
        return cls_name.find("Linear") != -1

    @staticmethod
    def _is_batchnorm2d_layer(cls_name: str) -> bool:
        return cls_name.find("BatchNorm2d") != -1


class LeakyReLUConv2d(nn.Module):
    """A class that defines a Conv2d follows by Leaky ReLU.

        [Conv2d with reflection pad with optional spectral normalization] -> [LeakyReLU].

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kernel_size: The size of the 2D convolutional kernel.
        stride: The stride of the 2D convolutional kernel.
        padding: The size of the padding.
        enable_instance_norm (optional): If True, instance normalization will be used. Defaults to False.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
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

        layers = [conv2d_layer]

        if enable_instance_norm:
            layers.append(nn.BatchNorm2d(out_channels, affine=False))
        layers.append(nn.LeakyReLU(inplace=True))

        self._model = nn.Sequential(*layers)
        self._model = _Initializer(init_type=_InitilizationType.NORMAL)(self._model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class ReLUInstNorm2dConv2d(nn.Module):
    """A class that defines the following transformation.

        [Conv2d with reflection padding] -> [BatchNorm2d] -> [ReLU]

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
            nn.BatchNorm2d(num_features=out_channels, affine=False),
            nn.LeakyReLU(inplace=True),
        )
        self._model = _Initializer(init_type=_InitilizationType.NORMAL)(self._model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class ResInstNorm2dConv2d(nn.Module):
    """A class that defines the residulal condition block that has the following architecture.

       - -> [Conv2d(3x3) -> [BatchNorm2d] -> [ReLU()] -> [Conv2d(3x3)] -> [BatchNorm2d] -> + ->
        |                                                                                        ^
        |----------------------------------------------------------------------------------------|

    The numberd of input and output channels are the same.

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
            nn.BatchNorm2d(in_channels),
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
            nn.BatchNorm2d(in_channels),
        )
        self._model = _Initializer(init_type=_InitilizationType.NORMAL)(self._model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._model(x)


class Downsampling2xkWithSkipConnection(nn.Module):
    """A class for downsampling block with skip connection.

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


class ContentEncoder(nn.Module):
    """A class that defines the architecture for the content encoder.

    This architecture of this model is:
        [LeakyReLUConv2d] -> 3 x [ReLUInstanceNormConv2d] -> 3 x [ResInstNorm2dConv2d]
    Args:
        in_channels: The number of input channels.
        out_channels (optional): The number of output channels. Defaults to NUM_STAIN_VECTORS.

    Returns:
        For each minibatch with N samples, it returns a tensor of size N x out_channels x H x W in which
        H and W is a downsampled version of the input image.
    """

    def __init__(self, in_channels: int, num_stain_vectors: int = NUM_STAIN_VECTORS) -> None:
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
