from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import torch

from ._algorithm import Algorithm
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.metrics import Metric
from ip_drit.common.utils import numel
from ip_drit.scheduler._scheduler import step_scheduler


class GroupAlgorithm(Algorithm):
    """A class defined for algorithms with group-wise logging.

    This group also handles schedulers.

    Args:
        device: The device to run the algorithm on.
        grouper: A grouper object that defines the groups for which we compute/log statistics for.
        logged_metrics: The list of metrics.
        logged_fields: The list of fields to log.
    """

    def __init__(
        self,
        device: str,
        grouper: AbstractGrouper,
        logged_metrics: List[Metric],
        logged_fields: List[str],
        schedulers,
        scheduler_metric_names,
        no_group_logging,
        **kwargs,
    ) -> None:
        super().__init__(device)
        self._grouper = grouper
        self._group_prefix = "group_"
        self._count_field = "count"
        self._group_count_field = f"{self._group_prefix}{self._count_field}"

        self._logged_metrics = logged_metrics
        self._logged_fields = logged_fields

        self._schedulers = schedulers
        self._scheduler_metric_names = scheduler_metric_names
        self._no_group_logging = no_group_logging

    def update_log(self, results: Dict[str, Any]) -> None:
        """Updates the internal log, Algorithm.log_dict.

        Args:
            results: A dictionary result.
        """
        results = self.sanitize_dict(results, to_out_device=False)
        # check all the fields exist
        for field in self._logged_fields:
            assert field in results, f"field {field} missing"
        # compute statistics for the current batch
        batch_log = {}
        with torch.no_grad():
            for m in self._logged_metrics:
                if not self._no_group_logging:
                    group_metrics, group_counts, worst_group_metric = m.compute_group_wise(
                        results["y_pred"], results["y_true"], results["g"], self.grouper.n_groups, return_dict=False
                    )
                    batch_log[f"{self._group_prefix}{m.name}"] = group_metrics
                batch_log[m.agg_metric_field] = m.compute(
                    results["y_pred"], results["y_true"], return_dict=False
                ).item()
            count = numel(results["y_true"])

        # transfer other statistics in the results dictionary
        for field in self._logged_fields:
            if field.startswith(self._group_prefix) and self._no_group_logging:
                continue
            v = results[field]
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                batch_log[field] = v.item()
            else:
                if isinstance(v, torch.Tensor):
                    assert (
                        v.numel() == self._grouper.n_groups
                    ), "Current implementation deals only with group-wise statistics or a single-number statistic"
                    assert field.startswith(self._group_prefix)
                batch_log[field] = v

        # update the log dict with the current batch
        if not self._has_log:  # since it is the first log entry, just save the current log
            self.log_dict = batch_log
            if not self._no_group_logging:
                self.log_dict[self._group_count_field] = group_counts
            self.log_dict[self._count_field] = count
        else:  # take a running average across batches otherwise
            for k, v in batch_log.items():
                if k.startswith(self._group_prefix):
                    if self._no_group_logging:
                        continue
                    self.log_dict[k] = _update_average(
                        self.log_dict[k], self.log_dict[self._group_count_field], v, group_counts
                    )
                else:
                    self.log_dict[k] = _update_average(self.log_dict[k], self.log_dict[self._count_field], v, count)
            if not self._no_group_logging:
                self.log_dict[self._group_count_field] += group_counts
            self.log_dict[self._count_field] += count
        self._has_log = True

    def get_log(self) -> Dict[str, Any]:
        """Sanitizes the internal log (Algorithm.log_dict) and outputs it."""
        sanitized_log = {}
        for k, v in self.log_dict.items():
            if k.startswith(self._group_prefix):
                field = k[len(self._group_prefix) :]
                for g in range(self._grouper.n_groups):
                    # set relevant values to NaN depending on the group count
                    count = self.log_dict[self._group_count_field][g].item()
                    if count == 0 and k != self._group_count_field:
                        outval = np.nan
                    else:
                        outval = v[g].item()
                    # add to dictionary with an appropriate name
                    # in practice, it is saving each value as {field}_group:{g}
                    added = False
                    for m in self._logged_metrics:
                        if field == m.name:
                            sanitized_log[m.group_metric_field(g)] = outval
                            added = True
                    if k == self._group_count_field:
                        sanitized_log[self.loss.group_count_field(g)] = outval
                        added = True
                    elif not added:
                        sanitized_log[f"{field}_group:{g}"] = outval
            else:
                assert not isinstance(v, torch.Tensor)
                sanitized_log[k] = v
        return sanitized_log

    def step_schedulers(self, is_epoch: bool, metrics: Dict[str, Any] = {}, log_access: bool = False) -> None:
        """Updates the scheduler after an epoch.

        If a scheduler is updated based on a metric (SingleModelAlgorithm.scheduler_metric),
        then it first looks for an entry in metrics_dict and then in its internal log
        (SingleModelAlgorithm.log_dict) if log_access is True.

        Args:
            is_epoch: If True, this is the current epoch.
            metrics: A dictionary of metrics.
            log_access: whether the scheduler_metric can be fetched from internal log (self.log_dict). Defaults to
                False.
        """
        for scheduler, metric_name in zip(self._schedulers, self._scheduler_metric_names):
            if scheduler is None:
                continue
            if is_epoch and scheduler.step_every_batch:
                continue
            if (not is_epoch) and (not scheduler.step_every_batch):
                continue
            self._step_specific_scheduler(
                scheduler=scheduler, metric_name=metric_name, metrics=metrics, log_access=log_access
            )

    def _step_specific_scheduler(self, scheduler, metric_name: str, metrics: Dict[str, Any], log_access: bool) -> None:
        """Helper function for updating scheduler.

        Args:
            scheduler: scheduler to update
            metric_name: name of the metric (key in metrics or log dictionary) to use for updates
            metrics: a dictionary of metrics that can be used for scheduler updates
            log_access: whether metrics from self.get_log() can be used to update schedulers
        """
        if not scheduler.use_metric or metric_name is None:
            metric = None
        elif metric_name in metrics:
            metric = metrics[metric_name]
        elif log_access:
            sanitized_log_dict = self.get_log()
            if metric_name in sanitized_log_dict:
                metric = sanitized_log_dict[metric_name]
            else:
                raise ValueError("scheduler metric not recognized")
        else:
            raise ValueError("scheduler metric not recognized")
        step_scheduler(scheduler, metric)

    def get_pretty_log_str(self) -> str:
        """Returns a pretty log string."""
        results_str = ""

        # Get sanitized log dict
        log = self.get_log()

        # Process aggregate logged fields
        for field in self._logged_fields:
            if field.startswith(self._group_prefix):
                continue
            results_str += f"{field}: {log[field]:.3f}\n"

        # Process aggregate logged metrics
        for metric in self._logged_metrics:
            results_str += f"{metric.agg_metric_field}: {log[metric.agg_metric_field]:.3f}\n"

        # Process logs for each group
        if not self._no_group_logging:
            for g in range(self._grouper.n_groups):
                group_count = log[f"count_group:{g}"]
                if group_count <= 0:
                    continue

                results_str += f"  {self._grouper.group_str(g)}  " f"[n = {group_count:6.0f}]:\t"

                # Process grouped logged fields
                for field in self._logged_fields:
                    if field.startswith(self._group_prefix):
                        field_suffix = field[len(self._group_prefix) :]
                        log_key = f"{field_suffix}_group:{g}"
                        results_str += f"{field_suffix}: " f"{log[log_key]:5.3f}\t"

                # Process grouped metric fields
                for metric in self._logged_metrics:
                    results_str += f"{metric.name}: " f"{log[metric.group_metric_field(g)]:5.3f}\t"
                results_str += "\n"
        else:
            results_str += "\n"

        return results_str


def _update_average(
    prev_avg: float, prev_counts: Union[int, torch.Tensor], curr_avg: float, curr_counts: Union[int, torch.Tensor]
) -> float:
    denominator = prev_counts + curr_counts
    if isinstance(curr_counts, torch.Tensor):
        denominator += (denominator == 0).float()
    elif isinstance(curr_counts, int) or isinstance(curr_counts, float):
        if denominator == 0:
            return 0.0
    else:
        raise ValueError("Type of curr_counts not recognized")
    prev_weight = prev_counts / denominator
    curr_weight = curr_counts / denominator
    return prev_weight * prev_avg + curr_weight * curr_avg
