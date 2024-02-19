"""A module that initialize different algorithms."""
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

from torch.utils.data import DataLoader

from ._contrimix import ContriMix
from ._erm import ERM
from ._histaugan import HistauGAN
from .single_model_algorithm import ModelAlgorithm
from .single_model_algorithm import SingleModelAlgorithm
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.metrics import Accuracy
from ip_drit.common.metrics import binary_logits_to_pred
from ip_drit.common.metrics import F1
from ip_drit.common.metrics import MSE
from ip_drit.common.metrics import multiclass_logits_to_pred
from ip_drit.common.metrics import MultiTaskAccuracy
from ip_drit.common.metrics import MultiTaskAveragePrecision
from ip_drit.datasets import AbstractLabelledPublicDataset
from ip_drit.datasets import AbstractUnlabelPublicDataset
from ip_drit.datasets import SubsetUnlabeledPublicDataset
from ip_drit.loss import ContriMixLoss
from ip_drit.loss import HistauGANLoss
from ip_drit.loss.initializer import initialize_loss
from ip_drit.patch_transform import AbstractJointTensorTransform
from saving_utils import load


algo_log_metrics = {
    "accuracy": Accuracy(prediction_fn=multiclass_logits_to_pred),
    "mse": MSE(),
    "multitask_accuracy": MultiTaskAccuracy(prediction_fn=multiclass_logits_to_pred),
    "multitask_binary_accuracy": MultiTaskAccuracy(prediction_fn=binary_logits_to_pred),
    "multitask_avgprec": MultiTaskAveragePrecision(prediction_fn=None),
    "f1": F1(prediction_fn=multiclass_logits_to_pred),
    None: None,
}


def initialize_algorithm(
    train_dataset: AbstractLabelledPublicDataset,
    config: Dict[str, Any],
    num_train_steps: int,
    train_grouper: AbstractGrouper,
    loss_weights_by_name: Optional[Dict[str, float]] = None,
    batch_transform: Optional[AbstractJointTensorTransform] = None,
    algorithm_parameters: Optional[Dict[str, Any]] = None,
    loss_kwargs: Dict[str, Any] = {},
    **kw_args: Dict[str, Any],
) -> SingleModelAlgorithm:
    """Initializes an algorithm based on the provided config dictionary.

    Args:
        train_dataset: The training dataset to use.
        config: A dictionary that is used to configure hwo the model should be initialized.
        num_train_steps: The number of training steps.
        train_grouper: A grouper object that defines the groups for which we compute/log statistics for.
        loss_weights_by_name (optional): A dictionary of loss weigths, keyed by the name of the loss. Defaults to None.
        batch_transform (optional): A module perform batch processing. Defaults to None, in which case, no batch
            processing will be performed.
        algorithm_parameters (optional): The parameters of the algorithm.
        loss_kwargs (optional): A dictionary of parameters for the loss. Defaults to None.
        kw_args: Keyword arguments.

    Returns:
        The initialized algorithm.
    """
    print(f"Initializing the {config['algorithm'].name} algorithm!")

    output_dim = _infer_output_dimensions(train_dataset=train_dataset, config=config)

    algorithm_parameters = {} if algorithm_parameters is None else algorithm_parameters

    if config["algorithm"] == ModelAlgorithm.ERM:
        algorithm = ERM(
            config=config,
            d_out=output_dim,
            grouper=train_grouper,
            loss=initialize_loss(loss_type=config["loss_function"]),
            metric=algo_log_metrics[config["algo_log_metric"]],
            n_train_steps=num_train_steps,
            batch_transform=batch_transform,
            **algorithm_parameters,
        )
    elif config["algorithm"] == ModelAlgorithm.HISTAUGAN:
        print("Initializing the HistauGAN algorithm")
        algorithm = HistauGAN(
            config=config,
            d_out=output_dim,
            grouper=train_grouper,
            loss=HistauGANLoss(loss_params=loss_kwargs),
            metric=algo_log_metrics[config["algo_log_metric"]],
            n_train_steps=num_train_steps,
            batch_transforms=batch_transform,
            training_mode=loss_kwargs["training_mode"],
            algorithm_parameters=algorithm_parameters,
        )

    elif config["algorithm"] == ModelAlgorithm.CONTRIMIX:
        print(f"Initializing the ContriMix algorithm!")
        algorithm = ContriMix(
            config=config,
            d_out=output_dim,
            grouper=train_grouper,
            loss=ContriMixLoss(d_out=output_dim, loss_weights_by_name=loss_weights_by_name, loss_params=loss_kwargs),
            metric=algo_log_metrics[config["algo_log_metric"]],
            n_train_steps=num_train_steps,
            num_attr_vectors=config["num_attr_vectors"],
            batch_transforms=batch_transform,
            training_mode=loss_kwargs["training_mode"],
            **algorithm_parameters,
        )
    else:
        raise ValueError(f"The algorithm {config['algorithm']} is not supported!")

    if config["pretrained_model_path"] is not None:
        pretrain_model_path = config["pretrained_model_path"]
        print(f"Loading pretrain model from {pretrain_model_path}.")
        load(module=algorithm, path=pretrain_model_path, device=config["device"])
        print(f"Loaded model state from {pretrain_model_path}!")

    return algorithm


def _infer_output_dimensions(
    train_dataset: Union[AbstractLabelledPublicDataset, SubsetUnlabeledPublicDataset], config: Dict[str, Any]
) -> Optional[int]:
    """Calculate the dimension of the output."""
    if isinstance(train_dataset, SubsetUnlabeledPublicDataset):
        return None

    if train_dataset.is_classification:
        if train_dataset.y_size == 1:  # Single task classification.
            return train_dataset.n_classes
        else:
            raise ValueError("Only support single-task classification!")
    else:
        raise ValueError("Only classification dataset is supported!")


def calculate_number_of_training_steps(config: Dict[str, Any], train_loader: DataLoader) -> int:
    """Computes the number of training steps."""
    return math.ceil(len(train_loader) / config["gradient_accumulation_steps"]) * config["n_epochs"]
