"""A module that defines the intializer for training models."""
from enum import auto
from enum import Enum

import torch.nn as nn


class InitilizationType(Enum):
    """Initialization types."""

    KAIMING = auto()
    XAVIER = auto()
    NORMAL = auto()


class Initializer:
    """A class that initializes the model weights.

    Args:
        init_type (optional): The type of the initialization. Defaults to InitilizationType.NORMAL.
        init_gain (optional): The gain that we use for initialization. Defaults to 0.02.

    """

    def __init__(self, init_type: InitilizationType = InitilizationType.NORMAL, init_gain: float = 0.02) -> None:
        self._init_type = init_type
        self._init_gain = init_gain

    def __call__(self, m: nn.Module):
        m.apply(self._initialize_module)
        return m

    def _initialize_module(self, m: object) -> None:
        class_name = m.__class__.__name__
        if self._has_weights(m):
            if self._is_conv_layer(class_name) or self._is_linear_layer(m):
                if self._init_type == InitilizationType.KAIMING:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif self._init_type == InitilizationType.XAVIER:
                    nn.init.xavier_normal_(m.weight.data, gain=self._init_gain)
                elif self._init_type == InitilizationType.NORMAL:
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
