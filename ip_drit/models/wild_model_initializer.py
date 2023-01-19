"""A module that initializes different models for the WILDS's dataset."""
import torch.nn as nn
from enum import Enum
from enum import auto
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union

class WildModel(Enum):
    DENSENET121 = auto()

def initialize_model_from_configuration(
        config: Dict[str, Any],
        d_out: int,
        output_separate_featurizer_and_classifier: bool = False,
    ) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
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
    pass