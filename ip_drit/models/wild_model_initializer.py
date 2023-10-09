"""A module that initializes different models for the WILDS's dataset."""
from enum import auto
from enum import Enum
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torchvision

from ip_drit.models import PathNetBX1


class WildModel(Enum):
    """An enum class that defines the model used for the WILDS dataset."""

    DENSENET121 = auto()
    WIDERESNET50 = auto()
    RESNET18 = auto()
    RESNET34 = auto()
    RESNET50 = auto()
    RESNET101 = auto()
    PathNetBX1Torch = auto()


def initialize_model_from_configuration(
    model_type: WildModel, d_out: int, output_classifier: bool = False, use_pretrained_backbone: bool = False
) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
    """Initializes the model based on the input configuration file.

    Pretrained weights are loaded according to config.pretrained_model_path using either transformers.from_pretrained
    (for bert-based models) or our own utils.load function (for torchvision models, resnet18-ms, and gin-virtual).
        There is currently no support for loading pretrained weights from disk for other models.
    Args:
        model_type: The type of the model to initialize.
        d_out: The dimensionality of the output.
        output_classifier: Creates a sequential architecture in which the classifier will be at the output.
        use_pretrained_backbone (optional): If True, a pretrained network will be used. Defaults to False.

    Returns:
        If output_classifier=True, returns a tuple of featurizer and classifier
        Here, featurizer is a model that outputs feature Tensors of shape (batch_size, ..., feature dimensionality).
        classifier is a model that takes in feature Tensors and outputs predictions. In most cases, this is a linear
        layer.

        If output_classifier=False, returns the featurizer module only.
    """
    if model_type in [WildModel.DENSENET121, WildModel.RESNET50]:
        featurizer = _initialize_torchvision_model(
            name=model_type, d_out=d_out, use_pretrained_backbone=use_pretrained_backbone
        )

        # The `needs_y` attribute specifies whether the model's forward function
        # needs to take in both (x, y).
        # If False, Algorithm.process_batch will call model(x).
        # If True, Algorithm.process_batch() will call model(x, y) during training,
        # and model(x, None) during eval.
        if output_classifier:
            out = featurizer
            if not hasattr(featurizer, "needs_y_input"):
                out.needs_y_input = False
            classifier = nn.Linear(featurizer.d_out, d_out)
            out = featurizer, classifier
        else:
            out = featurizer
            if not hasattr(out, "needs_y_input"):
                out.needs_y_input = False
        return out
    elif model_type == WildModel.PathNetBX1Torch:
        return PathNetBX1(in_channels=3, num_outputs=d_out, bn_momentum=0.01, use_bn=True)
    else:
        raise ValueError(f"Model type ({model_type}) is not supported!")


def _initialize_torchvision_model(name: WildModel, d_out: int, use_pretrained_backbone: bool):
    # get constructor and last layer names
    if name == WildModel.WIDERESNET50:
        constructor_name = "wide_resnet50_2"
        last_layer_name = "fc"
    elif name == WildModel.DENSENET121:
        constructor_name = "densenet121"
        last_layer_name = "classifier"
    elif name in (WildModel.RESNET18, WildModel.RESNET34, WildModel.RESNET50, WildModel.RESNET101):
        constructor_name = {
            WildModel.RESNET18: "resnet18",
            WildModel.RESNET34: "resnet34",
            WildModel.RESNET50: "resnet50",
            WildModel.RESNET101: "resnet101",
        }[name]
        last_layer_name = "fc"
    else:
        raise ValueError(f"Torchvision model {name} not recognized")
    # construct the default model, which has the default last layer
    constructor = getattr(torchvision.models, constructor_name)
    model = constructor(pretrained=use_pretrained_backbone)
    # adjust the last layer
    d_features = getattr(model, last_layer_name).in_features

    if d_out is None:  # want to initialize a featurizer model
        last_layer = Identity(d_features)
        model.d_out = d_features
    else:  # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out
    setattr(model, last_layer_name, last_layer)

    return model


class Identity(nn.Module):
    """An identity layer."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.in_features = d
        self.out_features = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
