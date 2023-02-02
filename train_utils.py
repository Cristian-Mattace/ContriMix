"""A module for training the model."""
import logging
import math
from typing import Any
from typing import Dict
from typing import Tuple

import torch
from tqdm import tqdm

from ip_drit.algorithms.single_model_algorithm import SingleModelAlgorithm
from ip_drit.common.metrics import binary_logits_to_pred
from ip_drit.logger import Logger
from saving_utils import save_model_if_needed
from saving_utils import save_pred_if_needed
from script_utils import detach_and_clone
from script_utils import log_results


def train(
    algorithm: SingleModelAlgorithm,
    split_dict_by_name: Dict[str, Dict],
    general_logger: Logger,
    config_dict: Dict[str, Any],
    epoch_offset: int,
) -> None:
    """Trains the model.

    For each epoch:
        - Steps an algorithm on the datasets['train'] split and the unlabeled split.
        - Evaluates the algorithm on the datasets['ood_val'] split.
        - Saves models / preds with frequency according to the configs
        - Evaluates on any other specified splits in the configs
    Assumes that the split_dict_by_name dict contains labeled data.

    Args:
        algorithm: The algorithm to use.
        split_dict_by_name: A dictionary that defines different things (dataset, loaders etc) for different split name.
        general_logger: The logger that is used to write the training logs.
        epoch: The index of the current epoch.
        config_dict: The configuration dictionary.
        epoch_offset: The initial epoch offset.
    """
    best_val_metric = None
    for epoch in range(epoch_offset, config_dict["n_epochs"]):
        general_logger.write(f"Epoch {epoch}: \n")

        _run_train_epoch(
            algorithm=algorithm,
            split_dict=split_dict_by_name["train"],
            general_logger=general_logger,
            epoch=epoch,
            config_dict=config_dict,
        )

        test_results, y_pred = _run_eval_epoch(
            algorithm=algorithm,
            split_dict=split_dict_by_name["test"],
            general_logger=general_logger,
            epoch=epoch,
            config_dict=config_dict,
        )

        metrics_to_evaluate = config_dict["metric"]
        general_logger.write(f" => Test {metrics_to_evaluate}: {test_results[metrics_to_evaluate]:.3f}\n")

        val_results, y_pred = _run_eval_epoch(
            algorithm=algorithm,
            split_dict=split_dict_by_name["ood_val"],
            general_logger=general_logger,
            epoch=epoch,
            config_dict=config_dict,
        )

        curr_val_metric = val_results[metrics_to_evaluate]
        general_logger.write(f" => OOD Validation {metrics_to_evaluate}: {curr_val_metric:.3f}\n")

        if best_val_metric is None:
            is_best = True
        else:
            if config_dict["val_metric_decreasing"]:
                is_best = curr_val_metric < best_val_metric
                best_val_metric = curr_val_metric
            else:
                is_best = curr_val_metric > best_val_metric
                best_val_metric = curr_val_metric

        if is_best:
            general_logger.write(f"Epoch {epoch} has the best validation performance so far.\n")

        save_model_if_needed(algorithm, split_dict_by_name["ood_val"], epoch, config_dict, is_best, best_val_metric)
        save_pred_if_needed(y_pred, split_dict_by_name["ood_val"], epoch, config_dict, is_best)
        general_logger.write("======================================================= \n\n")


def _run_train_epoch(
    algorithm: SingleModelAlgorithm,
    split_dict: Dict[str, Any],
    general_logger: Logger,
    config_dict: Dict[str, Any],
    epoch: int,
) -> Tuple[Dict[str, Any], str]:
    """Run 1 training epoch.

    Args:
        algorithm: The algorithm to use.
        split_dict: A dictionary for 1 split.
        general_logger: The logger that is used to write the training logs.
        config_dict: The configuration dictionary.
        epoch: The index of the current epoch.

    Returns:
        A dictionary of results
        A pretty print version of the results
    """
    algorithm.train()
    torch.set_grad_enabled(True)

    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []

    batches = split_dict["loader"]
    batches = tqdm(batches)
    last_batch_idx = len(batches) - 1

    for batch_idx, labeled_batch in enumerate(batches):
        batch_results = algorithm.update(labeled_batch, is_epoch_end=(batch_idx == last_batch_idx))
        epoch_y_true.append(detach_and_clone(batch_results["y_true"]))
        epoch_y_pred.append(detach_and_clone(batch_results["y_pred"]))
        epoch_metadata.append(detach_and_clone(batch_results["metadata"]))

        effective_batch_idx = (batch_idx + 1) / config_dict["gradient_accumulation_steps"]
        if effective_batch_idx % config_dict["log_every_n_batches"] == 0:
            if config_dict["verbose"]:
                general_logger.write(f"  Batch {batch_idx} \n")
            log_results(algorithm, split_dict, general_logger, epoch, math.ceil(effective_batch_idx))

    epoch_y_true = torch.cat(epoch_y_true, dim=0)
    epoch_y_pred = torch.cat(epoch_y_pred, dim=0)
    epoch_metadata = torch.cat(epoch_metadata, dim=0)

    # Running the evaluation on all the training slides.
    results, results_str = split_dict["dataset"].eval(
        epoch_y_pred, epoch_y_true, epoch_metadata, prediction_fn=binary_logits_to_pred
    )

    if config_dict["scheduler_metric_split"] == split_dict["split"]:
        algorithm.step_schedulers(is_epoch=True, metrics=results, log_access=False)

    # log after updating the scheduler in case it needs to access the internal logs
    log_results(algorithm, split_dict, general_logger, epoch, math.ceil(effective_batch_idx))

    results["epoch"] = epoch
    split_dict["eval_logger"].log(results)
    if split_dict["verbose"]:
        general_logger.write("  -> Epoch evaluation on all traning slides:\n" + results_str)

    return results, epoch_y_pred


def _run_eval_epoch(
    algorithm: SingleModelAlgorithm,
    split_dict: Dict[str, Any],
    general_logger: Logger,
    config_dict: Dict[str, Any],
    epoch: int,
) -> Tuple[Dict, str]:
    """Run 1 evaluation epoch.

    Args:
        algorithm: The algorithm to use.
        split_dict: A dictionary for 1 split.
        general_logger: The logger that is used to write the training logs.
        config_dict: The configuration dictionary.
        epoch: The index of the current epoch.

    Returns:
        A dictionary of results.
        A pretty print version of the results
    """
    algorithm.eval()
    torch.set_grad_enabled(False)

    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []
    effective_batch_idx = 0
    for batch_idx, labeled_batch in enumerate(tqdm(split_dict["loader"])):
        batch_results = algorithm.evaluate(labeled_batch)
        epoch_y_true.append(detach_and_clone(batch_results["y_true"]))
        epoch_y_pred.append(detach_and_clone(batch_results["y_pred"]))
        epoch_metadata.append(detach_and_clone(batch_results["metadata"]))

        effective_batch_idx = batch_idx + 1
        if effective_batch_idx % config_dict["log_every_n_batches"] == 0:
            log_results(algorithm, split_dict, general_logger, epoch, effective_batch_idx)

    epoch_y_true = torch.cat(epoch_y_true, dim=0)
    epoch_y_pred = torch.cat(epoch_y_pred, dim=0)
    epoch_metadata = torch.cat(epoch_metadata, dim=0)

    results, results_str = split_dict["dataset"].eval(
        epoch_y_pred, epoch_y_true, epoch_metadata, prediction_fn=binary_logits_to_pred
    )

    if config_dict["scheduler_metric_split"] == split_dict["split"]:
        algorithm.step_schedulers(is_epoch=True, metrics=results, log_access=True)

    # log after updating the scheduler in case it needs to access the internal logs
    log_results(algorithm, split_dict, general_logger, epoch, math.ceil(effective_batch_idx))

    results["epoch"] = epoch
    split_dict["eval_logger"].log(results)
    if split_dict["verbose"]:
        general_logger.write("  -> Epoch evaluation on all validation slides:\n" + results_str)

    return results, epoch_y_pred
