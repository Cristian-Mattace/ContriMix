"""A module that implements the Noisy student algorithm."""
from typing import Any
from typing import Dict
from typing import Tuple

import torch
import torch.nn as nn

from ._utils import move_to
from .single_model_algorithm import SingleModelAlgorithm
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.metrics import Metric
from ip_drit.loss import ElementwiseLoss
from ip_drit.models.wild_model_initializer import initialize_model_from_configuration


class NoisyStudent(SingleModelAlgorithm):
    r"""A class that implements the Noisy Student algorithm, which is a semi-supervised learning algorithm.

    One run of this algorithm gives us one iteration (load a teacher, train student). To run another iteration, re-run
    the previous command.

    To warm start the student model, point config.pretrained_model_path to config.teacher_model_path

    Based on the original paper, loss is of the form
        \ell_s + \ell_u
    where
        \ell_s = cross-entropy with true labels; student predicts with noise
        \ell_u = cross-entropy with pseudolabel generated without noise; student predicts with noise

    The teacher is trained on labeled data using strong augmentation.

    The student is noised using:
        - Input images are augmented using RandAugment
        - Single dropout layer before final classifier (fc) layer
    We do not use stochastic depth.

    Pseudolabels are generated in on unlabeled images that have only been randomly cropped and flipped ("weak"
    transform).
    By default, we use hard pseudolabels; use the --soft_pseudolabels flag to add soft pseudolabels.

    This code only supports a teacher that is the same class as the student (e.g. both densenet121s)

    Args:
        config: A dictionary that defines the configuration to load the model.
        d_out: The dimension of the model output.
        grouper: A grouper object that defines the groups for which we compute/log statistics for.
        loss: The loss module, calculated on the labeled data.
        unlabeled_loss: The loss module, calculated on the unlabeled data.
        metric: The metric to use.

    References:
        1. @inproceedings{xie2020self,
            title={Self-training with noisy student improves imagenet classification},
            author={Xie, Qizhe and Luong, Minh-Thang and Hovy, Eduard and Le, Quoc V},
            booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
            pages={10687--10698},
            year={2020}
            }
        2. @inproceedings{xie2020self,
            title={Extending the WILDS Benchmark for Unsupervised Adaptation},
            author={Gao, Irena and Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori and Liang, Percy},
            booktitle={},
            pages={1-14},
            year={2021}
            }
    """

    def __init__(
        self,
        config: Dict[str, Any],
        d_out: int,
        grouper: AbstractGrouper,
        loss: ElementwiseLoss,
        unlabeled_loss: ElementwiseLoss,
        metric: Metric,
        n_train_steps: int,
    ) -> None:
        super().__init__(
            config=config,
            model=self._initialize_student_model(
                noisystudent_add_dropout=config["noisystudent_add_dropout"], config=config, d_out=d_out
            ),
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self._unlabeled_loss = unlabeled_loss
        self._logged_fields.extend(["classification_loss", "consistency_loss"])

    @staticmethod
    def _initialize_student_model(noisystudent_add_dropout: bool, d_out: int, config: Dict[str, Any]) -> nn.Module:
        if noisystudent_add_dropout:
            featurizer, classifier = initialize_model_from_configuration(
                model_type=config["model"], d_out=d_out, output_classifier=True
            )
            dropout = nn.Dropout(p=config["noisystudent_dropout_rate"])
            out_module = nn.Sequential(featurizer, dropout, classifier)
            out_module.needs_y_input = featurizer.needs_y_input
            return out_module
        else:
            return initialize_model_from_configuration(model_type=config["model"], dout=d_out, output_classifier=False)

    def _process_unlabeled_batch(self, unlabeled_batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        x, pseudo_y, metadata = unlabeled_batch
        x = move_to(x, self._device)
        pseudo_y = move_to(pseudo_y, self._device)
        g = move_to(self._grouper.metadata_to_group_indices(metadata), self._device)
        return {
            "unlabeled_metadata": metadata,
            "unlabeled_g": g,
            "unlabeled_y_pred": self._get_model_output(x, None),
            "unlabeled_y_pseudo": pseudo_y,
        }

    def objective(self, results: Dict[str, Any]) -> float:
        labeled_loss = self._loss.compute(results["y_pred"], results["y_true"], return_dict=False)
        if "unlabeled_y_pseudo" in results:
            consistency_loss = self._unlabeled_loss.compute(
                results["unlabeled_y_pred"], results["unlabeled_y_pseudo"], return_dict=False
            )
        else:
            consistency_loss = 0

        self.save_metric_for_logging(results, "consistency_loss", consistency_loss)
        self.save_metric_for_logging(results, "classification_loss", labeled_loss)
        return labeled_loss + consistency_loss
