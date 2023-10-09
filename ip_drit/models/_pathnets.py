"""A module that stores architectures similar to PathNets."""
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Optional

import torch
import torch.nn as nn


class _PathNetConv2d(nn.Module):
    """The convolutional layer for 2D convolution.

    Args:
        in_channels: number of output channels in convolution
        out_channels: number of output channels in convolution
        kernel_size: width of filter kernels
        stride: stride of convolution
        use_activation (optional): If true, apply a ReLU activation. Defaults to True.
        use_bn (optional): If true, apply batch normalization. Defaults to True.
        bn_momentum (optional): batch norm momentum parameter. Defautls to 0.4.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        use_activation: bool = True,
        use_bn: bool = True,
        bn_momentum: float = 0.4,  # Note: momentum in PyTorch is (1- (momentum used in TF))
    ):
        super().__init__()
        self._use_activation = use_activation
        self._use_bn = use_bn
        self._conv: nn.Module = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=False
        )
        self._bn_layer: Optional[nn.Module] = (
            nn.BatchNorm2d(
                eps=1e-3, num_features=out_channels, momentum=bn_momentum, affine=True, track_running_stats=False
            )
            if use_bn
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        if self._use_bn:
            x = self._bn_layer(x)
        if self._use_activation:
            x = torch.nn.functional.relu(x, inplace=True)
        return x


class _PathNetBranchedConv2d(nn.Module):
    """Defines a branched-convolutional block used in PathNet models using the torch..nn API.

    Args:
        in_filters: Number of output channels in convolution
        out_filters: Number of output channels in convolution
        conv_kwargs: Additional arguments to pass to pathnet_conv_layer method.
    """

    def __init__(self, in_channels: int, out_channels: int, conv_kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self._pathnet_conv1 = _PathNetConv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, use_activation=True, **conv_kwargs
        )
        self._pathnet_conv2 = _PathNetConv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=2, use_activation=True, **conv_kwargs
        )
        self._pathnet_conv3 = _PathNetConv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=4, use_activation=False, **conv_kwargs
        )

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_up = self._pathnet_conv1.forward(x_in)
        x_up = self._pathnet_conv2.forward(x_up)
        x_down = self._pathnet_conv3.forward(x_in)
        return torch.nn.functional.relu(x_up + x_down, inplace=True)


class PathNetBX1(nn.Module):
    """The PathNetBX1 architecture.

    Args:
        in_channels: The number of input channels.
        use_bn (optional): If True, the batch normalization will be used. Defaults to True.
        bn_momentum (optional): The batch normalization momentum parameter. Defaults to 0.4.
    """

    def __init__(self, in_channels: int, num_outputs: int, use_bn: bool = True, bn_momentum: float = 0.4) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._batchnorm_params: Dict[str, Any] = {"use_bn": use_bn, "bn_momentum": bn_momentum}
        self._encoder = self._build_encoder()
        self._classifier = nn.Sequential(
            OrderedDict([("fc_1", nn.Conv2d(in_channels=512, out_channels=num_outputs, kernel_size=[9, 9]))])
        )
        self.needs_y_input = False

    def _build_encoder(self) -> nn.Module:
        return nn.Sequential(
            OrderedDict(
                [
                    # -- Block 1 -- #
                    (
                        "pathnet_conv_1",
                        _PathNetConv2d(
                            in_channels=self._in_channels,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            **self._batchnorm_params
                        ),
                    ),
                    # -- Block 2 -- #
                    (
                        "pathnet_branched_conv_1",
                        _PathNetBranchedConv2d(in_channels=32, out_channels=64, conv_kwargs=self._batchnorm_params),
                    ),
                    ("conv_1", nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
                    ("relu_1", nn.ReLU(inplace=True)),
                    # -- Block 3 -- #
                    (
                        "pathnet_branched_conv_2",
                        _PathNetBranchedConv2d(in_channels=64, out_channels=64, conv_kwargs=self._batchnorm_params),
                    ),
                    # -- Block 4 -- #
                    (
                        "pathnet_branched_conv_3",
                        _PathNetBranchedConv2d(in_channels=64, out_channels=128, conv_kwargs=self._batchnorm_params),
                    ),
                    ("conv_2", nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)),
                    ("relu_2", nn.ReLU(inplace=True)),
                    # -- Block 5 -- #
                    ("conv_3", nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2)),
                    ("relu_3", nn.ReLU(inplace=True)),
                    (
                        "pathnet_branched_conv_4",
                        _PathNetBranchedConv2d(in_channels=256, out_channels=256, conv_kwargs=self._batchnorm_params),
                    ),
                    # -- Block 6 -- #
                    (
                        "pathnet_branched_conv_5",
                        _PathNetBranchedConv2d(in_channels=256, out_channels=256, conv_kwargs=self._batchnorm_params),
                    ),
                    # -- Block 7 -- #
                    ("dropout2d_1", nn.Dropout2d(p=0.5)),
                    ("conv_4", nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=2)),
                    ("relu_4", nn.ReLU(inplace=True)),
                    ("dropout2d_2", nn.Dropout2d(p=0.5)),
                    ("conv_5", nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, stride=1)),
                    ("relu_5", nn.ReLU(inplace=True)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._encoder(x)
        x = self._classifier(x)
        return x.squeeze()
