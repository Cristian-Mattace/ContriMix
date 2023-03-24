"""This modules implements the ContriMix Augmentation algorithm."""
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from ._utils import move_to
from .multi_model_algorithm import MultimodelAlgorithm
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.metrics import Metric
from ip_drit.loss import ContriMixLoss
from ip_drit.loss import ElementwiseLoss
from ip_drit.models import AbsorbanceImGenerator
from ip_drit.models import AbsorbanceToTransmittance
from ip_drit.models import AttributeEncoder
from ip_drit.models import ContentEncoder
from ip_drit.models import SignalType
from ip_drit.models import TransmittanceToAbsorbance
from ip_drit.models.wild_model_initializer import initialize_model_from_configuration


class ContriMix(MultimodelAlgorithm):
    """A class that implements the ContriMix algorithm.

    Args:
        config: A dictionary that defines the configuration to load the model.
        d_out: The dimension of the model output.
        grouper: A grouper object that defines the groups for which we compute/log statistics for.
        loss: The loss module.
        metric: The metric to use.
        n_train_steps: The number of training steps.
        convert_to_absorbance_in_between (optional): If True (default), the input image will be converted to absorbance
            before decomposing into content and attribute.
        num_mxing_per_image (optional): The number of mixing images for each original image. Defaults to 5.
    """

    _NUM_INPUT_CHANNELS = 3
    _NUM_ATTR_VECTORS = 4
    _LOGGED_FIELDS: List[str] = [
        "objective",
        "self_recon_loss",
        "entropy_loss",
        "attribute_consistency_loss",
        "content_consistency_loss",
    ]

    def __init__(
        self,
        config: Dict[str, Any],
        d_out: int,
        grouper: AbstractGrouper,
        loss: ContriMixLoss,
        metric: Metric,
        n_train_steps: int,
        convert_to_absorbance_in_between: bool = True,
        num_mixing_per_image: int = 5,
    ) -> None:
        if not isinstance(loss, ContriMixLoss):
            raise ValueError(f"The specified loss module is of type {type(loss)}, not ContriMixLoss!")

        backbone_network = initialize_model_from_configuration(config["model"], d_out, output_classifier=False)

        if convert_to_absorbance_in_between:
            self._trans_to_abs_converter = TransmittanceToAbsorbance()
            self._abs_to_trans_converter = AbsorbanceToTransmittance()
        else:
            raise ValueError("ContriMix without converting to absorbance in between is not supported yet!")

        downsampling_factor: int = 1
        super().__init__(
            config=config,
            models_by_names={
                "backbone": backbone_network,
                "cont_enc": ContentEncoder(
                    in_channels=self._NUM_INPUT_CHANNELS,
                    num_stain_vectors=self._NUM_ATTR_VECTORS,
                    k=downsampling_factor,
                ),
                "attr_enc": AttributeEncoder(
                    in_channels=self._NUM_INPUT_CHANNELS,
                    num_stain_vectors=self._NUM_ATTR_VECTORS,
                    k=downsampling_factor,
                ),
                "im_gen": AbsorbanceImGenerator(k=downsampling_factor),
            },
            grouper=grouper,
            loss=loss,
            logged_fields=ContriMix._LOGGED_FIELDS,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self._use_unlabeled_y = config["use_unlabeled_y"]
        self._num_mixing_per_image = num_mixing_per_image
        self._convert_to_absorbance_in_between = convert_to_absorbance_in_between

    def _process_batch(
        self, labeled_batch: Tuple[torch.Tensor, ...], unlabeled_batch: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Dict[str, torch.Tensor]:
        x, y_true, metadata = labeled_batch
        x = move_to(x, self._device)
        self._validates_valid_input_signal_range(x)

        y_true = move_to(y_true, self._device)
        group_indices = move_to(self._grouper.metadata_to_group_indices(metadata), self._device)

        unlabeled_x = None
        if unlabeled_batch is not None:
            unlabeled_x = unlabeled_batch[0]
            unlabeled_x = move_to(unlabeled_x, self._device)

        out_dict = self._get_model_output(x, y_true, unlabeled_x)

        results = {"g": group_indices, "y_true": y_true, "metadata": metadata, **out_dict}

        return results

    @staticmethod
    def _validates_valid_input_signal_range(x: torch.Tensor) -> None:
        """Makes sure that the input signal has a valid range [0.0, 1.0]."""
        if x.min().item() < 0.0 or x.max().item() > 1.0:
            raise ValueError("The input tensor is not in a valid range of [0, 1]!")

    def _get_model_output(
        self, x: torch.Tensor, y_true: torch.Tensor, unlabeled_x: Optional[torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, SignalType]]:
        """Computes the model outputs.

        Args:
            x: The tensor of the input image.
            y_true: The groundtruth label of x.

        Returns:
            A dictionary of tensors, keyed by the name of the tensor.
        """
        cont_enc = self._models_by_names["cont_enc"]
        attr_enc = self._models_by_names["attr_enc"]
        all_target_image_indices = self._select_random_image_indices_by_image_index(batch_size=x.shape[0])
        all_target_image_indices = torch.stack(all_target_image_indices, dim=0)  # (Minibatch dim x #augmentations)

        return {
            "x_org": x,
            "unlabeled_x_org": unlabeled_x,
            "y_true": y_true,
            "cont_enc": cont_enc,
            "attr_enc": attr_enc,
            "im_gen": self._models_by_names["im_gen"],
            "abs_to_trans_cvt": self._abs_to_trans_converter,
            "trans_to_abs_cvt": self._trans_to_abs_converter,
            "backbone": self._models_by_names["backbone"],
            "all_target_image_indices": all_target_image_indices,
        }

    def _select_random_image_indices_by_image_index(self, batch_size: int) -> List[torch.Tensor]:
        """Returns a list of tensors that contains target image indices to sample from.

        Args:
            batch_size: The size of the training batch.

        Returns:
            A list of tensors in which each is the index of the images in the minibatch that we can use for ContriMix.
        """
        return [torch.randint(low=0, high=batch_size, size=(self._num_mixing_per_image,)) for _ in range(batch_size)]

    def objective(self, in_dict: Dict[str, Union[torch.Tensor, SignalType]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns a tuple of objective that can be backpropagate and a dictionary of all loss term."""
        loss_dict = self._loss.compute(in_dict=in_dict, return_dict=True)
        objective_loss_name = self._loss.agg_metric_field
        labeled_loss = loss_dict[objective_loss_name]
        non_objective_loss_by_name = {k: v for k, v in loss_dict.items() if k != objective_loss_name}

        if self._use_unlabeled_y and "unlabeled_y_true" in in_dict:
            unlabeled_loss = self._loss.compute(
                in_dict["unlabeled_y_pred"], in_dict["unlabeled_y_true"], return_dict=False
            )
            lab_size = len(in_dict["y_pred"])
            unl_size = len(in_dict["unlabeled_y_pred"])
            return (lab_size * labeled_loss + unl_size * unlabeled_loss) / (
                lab_size + unl_size
            ), non_objective_loss_by_name
        else:
            return labeled_loss, non_objective_loss_by_name
