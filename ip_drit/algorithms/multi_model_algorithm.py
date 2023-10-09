"""A module that defines the algorithm with multiple underlying models."""
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from ._contrimix_utils import ContrimixTrainingMode
from ._group_algorithm import GroupAlgorithm
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.metrics import Metric
from ip_drit.optimizer import get_parameters_from_models
from ip_drit.optimizer import initialize_optimizer
from ip_drit.patch_transform import AbstractJointTensorTransform
from ip_drit.scheduler import initialize_scheduler


class MultimodelAlgorithm(GroupAlgorithm):
    """An algorithm that contains many underlying models.

    Args:
        config: A configuration dictionary.
        models_by_names: A dictionary models used in the algorithm, keyed by the name of the model.
        grouper: A grouper object that defines the groups for which we compute/log statistics for.
        loss: The loss object.
        logged_fields: The list of strings that specifies the fields to tract. These strings are the subset of fields
            returns by the loss.
        metric: The metric to use.
        n_train_steps: The number of training steps.
        batch_transform (optional): A module perform batch processing. Defaults to None, in which case, no batch
            processing will be performed.
        train_networks (optional): If True, all networks will be trained. Otherwise, only the backbone network will be
            trained. Defaults to False.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        models_by_names: Dict[str, torch.nn.Module],
        grouper: AbstractGrouper,
        loss,
        logged_fields: List[str],
        metric: Metric,
        n_train_steps: int,
        batch_transform: Optional[AbstractJointTensorTransform] = None,
        training_mode: ContrimixTrainingMode = ContrimixTrainingMode.JOINTLY,
    ):
        self._training_mode: bool = training_mode
        self._loss = loss
        self._batch_transform = batch_transform
        logged_metrics = [self._loss]
        if metric is not None:
            self._metric = metric
            logged_metrics.append(self._metric)
        else:
            self._metric = None

        if config["use_ddp_over_dp"]:
            models_by_names = {
                k: nn.SyncBatchNorm.convert_sync_batchnorm(m.cuda(device=config["ddp_params"]["local_rank"]))
                for k, m in models_by_names.items()
            }

            # 'im_gen' does not have trainable params and excluded.
            parallelized_models_by_names = nn.ModuleDict(
                {
                    k: DistributedDataParallel(m, device_ids=[config["ddp_params"]["local_rank"]])
                    if k != "im_gen"
                    else m
                    for k, m in models_by_names.items()
                }
            )
        else:
            parallelized_models_by_names = nn.ModuleDict({k: DataParallel(m) for k, m in models_by_names.items()})

        self._use_amp = config.get("use_amp", False)
        if self._use_amp:
            self._fp16_scaler = torch.cuda.amp.GradScaler(enabled=True)

        if not hasattr(self, "optimizer") or self.optimizer is None:
            # TODO: move this piece of code to Contrimix, not here!
            if self._training_mode == ContrimixTrainingMode.ENCODERS:
                # Freeze the backbne
                parallelized_models_by_names["backbone"].train(False)
                for p in parallelized_models_by_names["backbone"].parameters():
                    p.requires_grad = False

            elif self._training_mode == ContrimixTrainingMode.BACKBONE:
                for name, m in parallelized_models_by_names.items():
                    if name != "backbone":
                        m.train(False)
                        for p in m.parameters():
                            p.requires_grad = False

            # Pass all the models to the optimizer to make sure that they are all registered.
            self._optimizer = initialize_optimizer(config, models=nn.ModuleList(parallelized_models_by_names.values()))
        self._max_grad_norm = config["max_grad_norm"]

        print(f"Using device {config['device']} for training.")
        for m in parallelized_models_by_names.values():
            m.to(config["device"])

        self._batch_idx = 0
        self._gradient_accumulation_steps = config["gradient_accumulation_steps"]

        super().__init__(
            device=config["device"],
            grouper=grouper,
            logged_metrics=logged_metrics,
            logged_fields=logged_fields,
            schedulers=[initialize_scheduler(config, self._optimizer, n_train_steps)],
            scheduler_metric_names=[config["scheduler_metric_name"]],
        )

        for k, m in parallelized_models_by_names.items():
            m.needs_y_input = models_by_names[k].needs_y_input

        self._models_by_names: Dict[str, nn.Module] = parallelized_models_by_names

    def update(
        self,
        labeled_batch: Tuple[torch.Tensor, ...],
        unlabeled_batch: Optional[Tuple[torch.Tensor, ...]] = None,
        is_epoch_end: bool = False,
        epoch: Optional[int] = None,
        return_loss_components: bool = True,
    ):
        """Process the batch, update the log, and update the model.

        Args:
            labeled_batch: A batch of data yielded by data loaders.
            unlabeled_batch (optional): A batch of data yielded by unlabeled data loader or None.
            is_epoch_end (optional): Whether this batch is the last batch of the epoch. If so, force optimizer to step,
                regardless of whether this batch idx divides self.gradient_accumulation_steps evenly. Defaults to False.
            epoch (optional): The index of the epoch.
            return_loss_components (optional): If True, the component of the loss will be return.

        Returns:
            A dictionary of the results, keyed by the field names. There are following fields.
                g: Groups indices for samples in the for batch.
                y_true: Ground truth labels for batch.
                metadata: Metadata for batch.
                y_pred: model output for batch.
                outputs: A tensor for the output.
                objective: The value of the objective.
        """
        if not self._is_training:
            raise RuntimeError("Can't upddate the model parameters because the algorithm is not in the training mode!")

        batch_results = self._process_batch(labeled_batch, unlabeled_batch, epoch=epoch)

        # update running statistics and update model if we've reached end of effective batch
        self._update(
            batch_results,
            should_step=((self._batch_idx + 1) % self._gradient_accumulation_steps == 0) or is_epoch_end,
            return_loss_components=return_loss_components,
        )

        if return_loss_components:
            self.update_log(batch_results)

        if is_epoch_end:
            self._batch_idx = 0
        else:
            self._batch_idx += 1

        return self._sanitize_dict(batch_results)

    def _update(self, results: Dict[str, Any], should_step: bool = False, return_loss_components: bool = True) -> None:
        """Computes the objective and updates the model.

        Also updates the results dictionary yielded by process_batch().
        Should be overridden to change algorithm update beyond modifying the objective.
        """
        objective, non_objective_loss_by_name = self.objective(results, return_loss_components=return_loss_components)

        if return_loss_components:
            # Only update the objective during log time.
            results["objective"] = objective.item()
            results.update(non_objective_loss_by_name)

        objective = objective / self._gradient_accumulation_steps

        if self._use_amp:
            # See more details about AMP at https://theaisummer.com/distributed-training-pytorch/
            self._fp16_scaler.scale(objective).backward()  # Scale the loss (32 bits) before the backward in 16 bits.
        else:
            objective.backward()

        if should_step:
            if self._max_grad_norm:
                all_params = get_parameters_from_models(self._models_by_names.values())
                clip_grad_norm_(all_params, self._max_grad_norm)

            if self._use_amp:
                self._fp16_scaler.step(self._optimizer)
                self._fp16_scaler.update()
            else:
                self._optimizer.step()

            self.step_schedulers(is_epoch=False, metrics=self._log_dict, log_access=False)

            # Prepare for the next update.
            for m in self._models_by_names.values():
                m.zero_grad(set_to_none=True)

    def evaluate(
        self,
        labeld_batch: Tuple[torch.Tensor, ...],
        ddp_params: Optional[Dict[str, Any]] = None,
        return_loss_components: bool = True,
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """Process the batch and update the log, without updating the model.

        Args:
            batch: A batch of data yielded by data loaders.
            ddp_params (optional): A dictionary that contains the parameters for DDP.
            return_loss_components (optional): If True, returns different components of the loss dictionary.. Defaults
                to True.

        Returns:
            A dictionary of the results, keyed by the field names. There are following fields.
                g: Groups for batch
                y_true: Ground truth labels for batch
                metadata: Metadata for batch
                y_pred: model output for batch
                outputs: A tensor for the output
                objective: The value of the objective.
        """
        assert not self._is_training
        results = self._process_batch(labeld_batch)

        objective, non_objective_loss_by_name = self.objective(results, return_loss_components=return_loss_components)
        results["objective"] = objective
        results.update(non_objective_loss_by_name)

        self.update_log(results)
        if ddp_params is not None:
            results = self._gather_results_from_all_gpus_for_ddp(results=results, ddp_params=ddp_params)

        return self._sanitize_dict(results)

    def _gather_results_from_all_gpus_for_ddp(
        self, results: Dict[str, Any], ddp_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        keys_to_gather = {"metadata_gpu", "y_true", "y_pred", "g"}

        out_res = {}
        for key, val in results.items():
            if key in keys_to_gather:
                out_res[key] = self._gather_items_from_all_gpus(val, num_replicas=ddp_params["num_replicas"])
            else:
                out_res[key] = val
        return out_res

    @staticmethod
    def _gather_items_from_all_gpus(input_tensor: torch.Tensor, num_replicas: int) -> torch.Tensor:
        gather_t_tensor = [torch.zeros_like(input_tensor) for _ in range(num_replicas)]
        dist.all_gather(gather_t_tensor, input_tensor)
        return torch.cat(gather_t_tensor, dim=0)

    def update_loss_weight_based_on_epoch(self, epoch: int) -> None:
        """Update the weights of the loss based on epoch index."""
        self._loss.update_epoch_index(epoch=epoch)
