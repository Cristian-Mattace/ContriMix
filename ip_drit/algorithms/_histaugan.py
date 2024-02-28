"""This modules implements the ContriMix Augmentation algorithm."""
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._utils import move_to
from .multi_model_algorithm import MultimodelAlgorithm
from ip_drit.algorithms._contrimix_utils import ContrimixTrainingMode
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.metrics import Metric
from ip_drit.loss import HistauGANLoss
from ip_drit.models import InitializationType
from ip_drit.models import Initializer
from ip_drit.models import MD_E_content
from ip_drit.models import MD_G_multi_concat
from ip_drit.models import SignalType
from ip_drit.models.wild_model_initializer import initialize_model_from_configuration
from ip_drit.patch_transform import PostContrimixTransformPipeline


class HistauGAN(MultimodelAlgorithm):
    """A class that implements the HistauGAN algorithm.

    Args:
        config: A dictionary that defines the configuration to load the model.
        d_out: The dimension of the model output.
        grouper: A grouper object that defines the groups for which we compute/log statistics for.
        loss: The loss module.
        metric: The metric to use.
        n_train_steps: The number of training steps.
        convert_to_absorbance_in_between (optional): If True (default), the input image will be converted to absorbance
            before decomposing into content and attribute.
        batch_transform (optional): A module perform batch processing. Defaults to None, in which case, no batch
            processing will be performed.
        num_attr_vectors (optional): The number of stain vectors. Defaults to 4
        local_rank (optional): The local rank used for training with DDP. Defaults to None.
        input_zc_logits (bool): If True, the content encoder will output logits, not activation output.
            The label associated contrimix should have this one set to True. Defaults to False.
        training_mode (optional): If True, only the Contrimix encoder will be trained. Defaults to
            ContrimixTrainingMode.JOINTLY, meaning both the backbone and the encoders will be trained at the same time.
        logged_fields (optional): A list of fields to log. Defaults to None, in which case, the default field names will
            be used.

    Reference:
        https://github.com/sophiajw/HistAuGAN
    """

    _NUM_INPUT_CHANNELS = 3

    def __init__(
        self,
        config: Dict[str, Any],
        d_out: int,
        grouper: AbstractGrouper,
        loss: HistauGANLoss,
        metric: Metric,
        n_train_steps: int,
        algorithm_parameters: Dict[str, Any],
        batch_transforms: Optional[PostContrimixTransformPipeline] = None,
        training_mode: ContrimixTrainingMode = ContrimixTrainingMode.JOINTLY,
        logged_fields: Optional[List[str]] = None,
    ) -> None:
        if not isinstance(loss, HistauGANLoss):
            raise ValueError(f"The specified loss module is of type {type(loss)}, not HistauGANLoss!")
        self._gan_model_path = algorithm_parameters["gan_model_path"]
        self._num_domains = algorithm_parameters["num_domains"]
        self._aug_with_all_domains: bool = algorithm_parameters["aug_with_all_domains"]
        self._model_type = config["model"]
        self._d_out = d_out
        self._nz = algorithm_parameters["nz"]
        self._concat = True
        self._saving_freq_iters = 20

        backbone_network = initialize_model_from_configuration(
            self._model_type,
            self._d_out,
            output_classifier=False,
            use_pretrained_backbone=config["model_kwargs"]["pretrained"],
        )

        if logged_fields is None:
            logged_fields = self._update_log_fields_based_on_training_mode(training_mode=training_mode)

        super().__init__(
            config=config,
            models_by_names=self._initialize_histaugan_networks(
                backbone_network=backbone_network, algorithm_options=algorithm_parameters
            ),
            grouper=grouper,
            loss=loss,
            logged_fields=logged_fields,
            metric=metric,
            n_train_steps=n_train_steps,
            batch_transform=batch_transforms,
            training_mode=training_mode,
        )

        self._use_amp = getattr(config, "use_amp", False)

    @staticmethod
    def _update_log_fields_based_on_training_mode(training_mode: ContrimixTrainingMode):
        log_fields = []
        if training_mode in (ContrimixTrainingMode.BACKBONE, ContrimixTrainingMode.JOINTLY):
            log_fields.append("entropy_loss")
        elif training_mode in (ContrimixTrainingMode.ENCODERS, ContrimixTrainingMode.JOINTLY):
            raise ValueError("To train HistAuGan, please see https://github.com/sophiajw/HistAuGAN.")
        return log_fields

    def _initialize_histaugan_networks(
        self, backbone_network: nn.Module, algorithm_options: Dict[str, Any]
    ) -> Dict[str, nn.Module]:
        """Initialize the HistauGAN networks.

        Reference:
            https://github.com/sophiajw/HistAuGAN/blob/main/histaugan/model.py#L14
        """
        state_dict = torch.load(self._gan_model_path)
        gen = MD_G_multi_concat(
            output_dim=algorithm_options["input_dim"], c_dim=algorithm_options["num_domains"], nz=self._nz
        )
        gen.load_state_dict(state_dict["gen"])

        enc_c = MD_E_content(input_dim=algorithm_options["input_dim"])
        enc_c.load_state_dict(state_dict["enc_c"])

        return {
            "backbone": Initializer(init_type=InitializationType.NORMAL)(backbone_network),
            "enc_c": enc_c,
            "gen": gen,
        }

    def _parse_inputs(self, labeled_batch: Optional[Tuple[torch.Tensor, ...]], split: str) -> Dict[str, torch.Tensor]:
        x, y_true = None, None
        if labeled_batch is not None:
            x, y_true, metadata = labeled_batch
            x = move_to(x, self._device)
            y_true = move_to(y_true, self._device)

        return {
            "x": x,
            "g": move_to(
                F.one_hot(self._grouper.metadata_to_group(metadata), num_classes=self._num_domains).float(),
                self._device,
            ),
            "y_true": y_true,
            "metadata": metadata,
            "target_domain_indices": torch.arange(self._num_domains)
            if self._aug_with_all_domains
            else self._grouper.group_indices_by_split_name[split],
        }

    def objective(
        self, in_dict: Dict[str, Union[torch.Tensor, SignalType]], return_loss_components: bool
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns a tuple of objective that can be backpropagate and a dictionary of all loss term."""
        if self._use_amp:
            with torch.cuda.amp.autocast():
                loss_dict = self._loss.compute(
                    in_dict=in_dict, return_dict=True, return_loss_components=return_loss_components
                )
        else:
            loss_dict = self._loss.compute(
                in_dict=in_dict, return_dict=True, return_loss_components=return_loss_components
            )

        objective_loss_name = self._loss.agg_metric_field
        return loss_dict[objective_loss_name], {k: v for k, v in loss_dict.items() if k != objective_loss_name}

    def _process_batch(
        self,
        labeled_batch: Optional[Tuple[torch.Tensor, ...]],
        split: str,
        unlabeled_batch: Optional[Tuple[torch.Tensor, ...]] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        out_dict = self._parse_inputs(labeled_batch, split=split)
        out_dict.update(
            {
                "is_training": self._is_training,
                "backbone": self._models_by_names["backbone"],
                "gen": self._models_by_names["gen"],
                "enc_c": self._models_by_names["enc_c"],
            }
        )
        return out_dict
