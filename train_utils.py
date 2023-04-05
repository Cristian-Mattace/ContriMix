"""A module for training the model."""
import logging
import math
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from ip_drit.algorithms.single_model_algorithm import ModelAlgorithm
from ip_drit.algorithms.single_model_algorithm import SingleModelAlgorithm
from ip_drit.common.data_loaders import InfiniteDataIterator
from ip_drit.common.metrics import binary_logits_to_pred
from ip_drit.logger import Logger
from saving_utils import save_model_if_needed
from saving_utils import save_pred_if_needed
from script_utils import detach_and_clone
from script_utils import log_results


def train(
    algorithm: SingleModelAlgorithm,
    labeled_split_dict_by_name: Dict[str, Dict],
    general_logger: Logger,
    config_dict: Dict[str, Any],
    epoch_offset: int,
    unlabeled_split_dict_by_name: Optional[Dict[str, Dict]] = None,
) -> None:
    """Trains the model.

    For each epoch:
        - Steps an algorithm on the datasets['train'] split and the unlabeled split.
        - Evaluates the algorithm on the datasets['val'] split (OOD val split).
        - Saves models / preds with frequency according to the configs
        - Evaluates on any other specified splits in the configs
    Assumes that the split_dict_by_name dict contains labeled data.

    Args:
        algorithm: The algorithm to use.
        labeled_split_dict_by_name: A dictionary that defines different things (dataset, loaders etc) for different
            splits of the labeled data.
        general_logger: The logger that is used to write the training logs.
        epoch: The index of the current epoch.
        config_dict: The configuration dictionary.
        epoch_offset: The initial epoch offset.
        unlabeled_split_dict_by_name (optional): A dictionary that defines different things (dataset, loaders etc) for
            different splits of the unlabeled data. Defaults to None.
    """
    best_val_metric = None
    for epoch in range(epoch_offset, config_dict["n_epochs"]):
        general_logger.write(f"Epoch {epoch}: \n")

        _run_train_epoch(
            algorithm=algorithm,
            labeled_split_dict=labeled_split_dict_by_name["train"],
            general_logger=general_logger,
            epoch=epoch,
            config_dict=config_dict,
            unlabeled_split_dict=None
            if unlabeled_split_dict_by_name is None
            else unlabeled_split_dict_by_name["train_unlabeled"],
        )

        metrics_to_evaluate = config_dict["metric"]
        test_results, y_pred = _run_eval_epoch(
            algorithm=algorithm,
            labeled_split_dict=labeled_split_dict_by_name["test"],
            general_logger=general_logger,
            epoch=epoch,
            config_dict=config_dict,
        )
        general_logger.write(f" => Test {metrics_to_evaluate}: {test_results[metrics_to_evaluate]:.3f}\n\n")

        val_results, y_pred = _run_eval_epoch(
            algorithm=algorithm,
            labeled_split_dict=labeled_split_dict_by_name["val"],
            general_logger=general_logger,
            epoch=epoch,
            config_dict=config_dict,
        )

        curr_val_metric = val_results[metrics_to_evaluate]
        general_logger.write(f" => OOD Validation {metrics_to_evaluate}: {curr_val_metric:.3f}\n\n")

        if best_val_metric is None:
            is_best = True
            best_val_metric = curr_val_metric

        if config_dict["val_metric_decreasing"]:
            is_best = curr_val_metric <= best_val_metric
        else:
            is_best = curr_val_metric >= best_val_metric

        if is_best:
            best_val_metric = curr_val_metric
            general_logger.write(f"Epoch {epoch} has the best validation performance so far.\n")
        save_model_if_needed(algorithm, labeled_split_dict_by_name["val"], epoch, config_dict, is_best, best_val_metric)
        save_pred_if_needed(y_pred, labeled_split_dict_by_name["val"], epoch, config_dict, is_best)
        general_logger.write("======================================================= \n\n")


def _run_train_epoch(
    algorithm: SingleModelAlgorithm,
    labeled_split_dict: Dict[str, Any],
    general_logger: Logger,
    config_dict: Dict[str, Any],
    epoch: int,
    unlabeled_split_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Run 1 training epoch.

    Args:
        algorithm: The algorithm to use.
        labeled_split_dict: A dictionary for 1 split of the labeled data.
        general_logger: The logger that is used to write the training logs.
        config_dict: The configuration dictionary.
        epoch: The index of the current epoch.
        unlabeled_split_dict (optional): A dictionary for 1 split of the unlabeled data. Defaults to None.

    Returns:
        A dictionary of results
        A pretty print version of the results
    """
    if config_dict["reset_random_generator_after_every_epoch"]:
        # This is needed to make sure that each epoch starts with the same base random generator.
        _reset_random_number_generator(seed=config_dict["seed"])

    use_unlabeled_data = unlabeled_split_dict is not None
    algorithm.train()
    torch.set_grad_enabled(True)

    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []

    batches = labeled_split_dict["loader"]
    batches = tqdm(batches)
    last_batch_idx = len(batches) - 1

    if use_unlabeled_data:
        unlabeled_data_iterator = InfiniteDataIterator(unlabeled_split_dict["loader"])

    algorithm.update_loss_weight_based_on_epoch(epoch=epoch)
    for batch_idx, labeled_batch in enumerate(batches):
        logging.debug(f" -> batch_index: {batch_idx}, data = {labeled_batch[0][0,0,0,0]}")
        if use_unlabeled_data:
            batch_results = algorithm.update(
                labeled_batch=labeled_batch,
                unlabeled_batch=next(unlabeled_data_iterator),
                is_epoch_end=(batch_idx == last_batch_idx),
                epoch=epoch,
            )
        else:
            batch_results = algorithm.update(
                labeled_batch=labeled_batch, is_epoch_end=(batch_idx == last_batch_idx), epoch=epoch
            )

        epoch_y_true.append(detach_and_clone(batch_results["y_true"]))
        epoch_y_pred.append(detach_and_clone(batch_results["y_pred"]))
        epoch_metadata.append(detach_and_clone(batch_results["metadata"]))

        effective_batch_idx = (batch_idx + 1) / config_dict["gradient_accumulation_steps"]
        if effective_batch_idx % config_dict["log_every_n_batches"] == 0:
            if config_dict["verbose"]:
                general_logger.write(f"  Batch {batch_idx} \n")
            log_results(algorithm, labeled_split_dict, general_logger, epoch, math.ceil(effective_batch_idx))

    epoch_y_true = torch.cat(epoch_y_true, dim=0)
    epoch_y_pred = torch.cat(epoch_y_pred, dim=0)
    epoch_metadata = torch.cat(epoch_metadata, dim=0)

    # Running the evaluation on all the training slides.
    results, results_str = labeled_split_dict["dataset"].eval(
        epoch_y_pred, epoch_y_true, epoch_metadata, prediction_fn=binary_logits_to_pred
    )

    if config_dict["scheduler_metric_split"] == labeled_split_dict["split"]:
        algorithm.step_schedulers(is_epoch=True, metrics=results, log_access=False)

    # log after updating the scheduler in case it needs to access the internal logs
    log_results(algorithm, labeled_split_dict, general_logger, epoch, math.ceil(effective_batch_idx))

    results["epoch"] = epoch
    labeled_split_dict["eval_logger"].log(results)
    if labeled_split_dict["verbose"]:
        general_logger.write("  -> Epoch evaluation on all training slides:\n" + results_str)

    return results, epoch_y_pred


def _reset_random_number_generator(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_eval_epoch(
    algorithm: SingleModelAlgorithm,
    labeled_split_dict: Dict[str, Any],
    general_logger: Logger,
    config_dict: Dict[str, Any],
    epoch: int,
    save_results: bool = False,
    is_best: bool = False,
    split: Optional[str] = "train",
) -> Tuple[Dict, str]:
    """Run 1 evaluation epoch.

    Args:
        algorithm: The algorithm to use.
        labeled_split_dict: A dictionary for 1 split of the labeled data.
        general_logger: The logger that is used to write the training logs.
        config_dict: The configuration dictionary.
        epoch: The index of the current epoch.
        save_results: Save results in a csv. Needed for eval
        split: The type of the split for the labeled data, which can be one of `train`, `id_val`, `val`, `test`.

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
    for batch_idx, labeled_batch in enumerate(tqdm(labeled_split_dict["loader"])):
        logging.debug(f" -> batch_index: {batch_idx}, data = {labeled_batch[0][0,0,0,0]}")
        batch_results = algorithm.evaluate(labeled_batch)
        epoch_y_true.append(detach_and_clone(batch_results["y_true"]))
        epoch_y_pred.append(detach_and_clone(batch_results["y_pred"]))
        epoch_metadata.append(detach_and_clone(batch_results["metadata"]))

        effective_batch_idx = batch_idx + 1
        if effective_batch_idx % config_dict["log_every_n_batches"] == 0:
            log_results(algorithm, labeled_split_dict, general_logger, epoch, effective_batch_idx)

    epoch_y_true = torch.cat(epoch_y_true, dim=0)
    epoch_y_pred = torch.cat(epoch_y_pred, dim=0)
    epoch_metadata = torch.cat(epoch_metadata, dim=0)

    results, results_str = labeled_split_dict["dataset"].eval(
        epoch_y_pred, epoch_y_true, epoch_metadata, prediction_fn=binary_logits_to_pred
    )

    if config_dict["scheduler_metric_split"] == labeled_split_dict["split"]:
        algorithm.step_schedulers(is_epoch=True, metrics=results, log_access=True)

    # log after updating the scheduler in case it needs to access the internal logs
    log_results(algorithm, labeled_split_dict, general_logger, epoch, math.ceil(effective_batch_idx))

    results["epoch"] = epoch
    labeled_split_dict["eval_logger"].log(results)
    if labeled_split_dict["verbose"]:
        general_logger.write(results_str)

    # Skip saving train preds, since the train loader generally shuffles the data
    if save_results and split != "train":
        save_pred_if_needed(epoch_y_pred, labeled_split_dict, epoch, config_dict, is_best, force_save=True)

    return results, epoch_y_pred


def evaluate_over_splits(algorithm, datasets, epoch, general_logger, config_dict, is_best, save_results) -> None:
    """Evaluate the algorithm over multiple data splits."""
    for split, dataset in datasets.items():
        _run_eval_epoch(
            algorithm=algorithm,
            labeled_split_dict=dataset,
            epoch=epoch,
            general_logger=general_logger,
            config_dict=config_dict,
            is_best=is_best,
            split=split,
            save_results=save_results,
        )
