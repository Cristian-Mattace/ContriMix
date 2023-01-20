"""A module that initializes different models for the WILDS's dataset."""
import torch.nn as nn
from enum import Enum
from enum import auto
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
import torchvision

class WildModel(Enum):
    DENSENET121 = auto()
    WIDERESNET50 = auto()
    RESNET18 = auto()
    RESNET34 = auto()
    RESNET50 = auto()
    RESNET101 = auto()

def initialize_model_from_configuration(
        config: Dict[str, Any],
        d_out: int,
        output_separate_featurizer_and_classifier: bool = False,
    ) -> nn.Module:
    """Initializes the model based on the input configuration file.

    Pretrained weights are loaded according to config.pretrained_model_path using either transformers.from_pretrained
    (for bert-based models) or our own utils.load function (for torchvision models, resnet18-ms, and gin-virtual).
        There is currently no support for loading pretrained weights from disk for other models.
    Args:
        config: A configuration dictionary for the model to be initialized.
        d_out: The dimensionality of the output.
        output_separate_featurizer_and_classifier: output_separate_featurizer_and_classifier

    Returns:
         If output_separate_featurizer_and_classifier=True:
            - featurizer: a model that outputs feature Tensors of shape (batch_size, ..., feature dimensionality)
            - classifier: a model that takes in feature Tensors and outputs predictions. In most cases, this is a linear layer.

        If output_separate_featurizer_and_classifier=False:
            - model: a model that is equivalent to nn.Sequential(featurizer, classifier)
    """
    model_type = config['model']
    if model_type == WildModel.DENSENET121:
        featurizer = _initialize_torchvision_model(
            name=model_type,
            d_out=d_out,
        )

        out = featurizer

        if output_separate_featurizer_and_classifier:
            classifier = nn.Linear(
                featurizer.d_out, d_out
            )
            out = nn.Sequential(*(featurizer, classifier))

        # The `needs_y` attribute specifies whether the model's forward function
        # needs to take in both (x, y).
        # If False, Algorithm.process_batch will call model(x).
        # If True, Algorithm.process_batch() will call model(x, y) during training,
        # and model(x, None) during eval.
        if not hasattr(out, 'needs_y'):
            # Sometimes model is a tuple of (featurizer, classifier)
            if output_separate_featurizer_and_classifier:
                for submodel in out:
                    submodel.needs_y = False
            else:
                out.needs_y = False
        return out

def _initialize_torchvision_model(name: WildModel, d_out: int, **kwargs):
    # get constructor and last layer names
    if name == WildModel.WIDERESNET50:
        constructor_name = 'wide_resnet50_2'
        last_layer_name = 'fc'
    elif name == WildModel.DENSENET121:
        constructor_name = 'densenet121'
        last_layer_name = 'classifier'
    elif name in (WildModel.RESNET18, WildModel.RESNET34, WildModel.RESNET50, WildModel.RESNET101):
        constructor_name = {
            WildModel.RESNET18: 'resnet18',
            WildModel.RESNET34: 'resnet34',
            WildModel.RESNET50: 'resnet50',
            WildModel.RESNET101: 'resnet101',
        }[name]
        last_layer_name = 'fc'
    else:
        raise ValueError(f'Torchvision model {name} not recognized')
    # construct the default model, which has the default last layer
    constructor = getattr(torchvision.models, constructor_name)
    model = constructor(**kwargs)
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self, d: int) -> None:
        super().__init__()
        self.in_features = d
        self.out_features = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
