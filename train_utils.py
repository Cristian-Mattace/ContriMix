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

from ip_drit.algorithms.single_model_algorithm import SingleModelAlgorithm
from ip_drit.common.metrics import binary_logits_to_pred
from ip_drit.common.metrics import multiclass_logits_to_pred
from saving_utils import save_model_if_needed
from saving_utils import save_pred_if_needed
from script_utils import detach_and_clone
from script_utils import is_master_process
from script_utils import log_results

process_outputs_functions = {
    "binary_logits_to_pred": binary_logits_to_pred,
    "multiclass_logits_to_pred": multiclass_logits_to_pred,
    None: None,
}


def train(
    algorithm: SingleModelAlgorithm,
    labeled_split_dict_by_name: Optional[Dict[str, Dict]],
    config_dict: Dict[str, Any],
    epoch_offset: int,
    unlabeled_split_dict_by_name: Optional[Dict[str, Dict]] = None,
    num_training_epochs_per_evaluation: int = 1,
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
        num_training_epochs_per_evaluation (optional): The number of training epochs before running the evaluation.
            Defaults to 1.
    """
    best_val_metric = None
    for epoch in range(epoch_offset, config_dict["n_epochs"]):
        print(f"Epoch {epoch}: \n")
        do_training = True
        if do_training:
            _run_train_epoch(
                algorithm=algorithm,
                labeled_split_dict=None if labeled_split_dict_by_name is None else labeled_split_dict_by_name["train"],
                epoch=epoch,
                config_dict=config_dict,
                unlabeled_split_dict=None
                if unlabeled_split_dict_by_name is None
                else unlabeled_split_dict_by_name["train"],
            )

        if epoch % num_training_epochs_per_evaluation == 0:
            metrics_to_evaluate = config_dict["metric"]
            if (labeled_split_dict_by_name is not None and "test" in labeled_split_dict_by_name) or (
                unlabeled_split_dict_by_name is not None and "test" in unlabeled_split_dict_by_name
            ):
                test_res = _run_eval_epoch(
                    algorithm=algorithm,
                    labeled_split_dict=None
                    if labeled_split_dict_by_name is None
                    else labeled_split_dict_by_name["test"],
                    unlabeled_split_dict=None
                    if unlabeled_split_dict_by_name is None
                    else unlabeled_split_dict_by_name["test"],
                    epoch=epoch,
                    config_dict=config_dict,
                )
                if test_res is not None:
                    test_results, y_pred = test_res
                    print(f" => Test {metrics_to_evaluate}: {test_results[metrics_to_evaluate]:.3f}\n\n")

                if is_master_process(config_dict):
                    if labeled_split_dict_by_name is not None:
                        dataset_name = labeled_split_dict_by_name["test"]["dataset"].dataset_name
                    else:
                        dataset_name = unlabeled_split_dict_by_name["test"]["dataset"].dataset_name

                    save_model_if_needed(
                        algorithm, dataset_name, epoch, config_dict, is_best=False, best_val_metric=None
                    )

            if (labeled_split_dict_by_name is not None and "val" in labeled_split_dict_by_name) or (
                unlabeled_split_dict_by_name is not None and "val" in unlabeled_split_dict_by_name
            ):
                val_res = _run_eval_epoch(
                    algorithm=algorithm,
                    labeled_split_dict=None
                    if labeled_split_dict_by_name is None
                    else labeled_split_dict_by_name["val"],
                    unlabeled_split_dict=None
                    if unlabeled_split_dict_by_name is None
                    else unlabeled_split_dict_by_name["val"],
                    epoch=epoch,
                    config_dict=config_dict,
                )

                if val_res is not None:
                    val_results, y_pred = val_res
                    curr_val_metric = val_results[metrics_to_evaluate]
                    print(f" => OOD Validation {metrics_to_evaluate}: {curr_val_metric:.3f}\n\n")

                    if best_val_metric is None:
                        is_best = True
                        best_val_metric = curr_val_metric

                    if config_dict["val_metric_decreasing"]:
                        is_best = curr_val_metric <= best_val_metric
                    else:
                        is_best = curr_val_metric >= best_val_metric

                    if is_best:
                        best_val_metric = curr_val_metric
                        print(f"Epoch {epoch} has the best validation performance so far.\n")

                    if is_master_process(config_dict):
                        save_model_if_needed(
                            algorithm,
                            labeled_split_dict_by_name["val"]["dataset"].dataset_name,
                            epoch,
                            config_dict,
                            is_best,
                            best_val_metric,
                        )
                        save_pred_if_needed(y_pred, labeled_split_dict_by_name["val"], epoch, config_dict, is_best)

        print("======================================================= \n\n")


def _run_train_epoch(
    algorithm: SingleModelAlgorithm,
    labeled_split_dict: Optional[Dict[str, Any]],
    config_dict: Dict[str, Any],
    epoch: int,
    unlabeled_split_dict: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Run 1 training epoch.

    Args:
        algorithm: The algorithm to use.
        labeled_split_dict: A dictionary for 1 split of the labeled data.
        general_logger: The logger that is used to write the training logs.
        config_dict: The configuration dictionary.
        epoch: The index of the current epoch.
        unlabeled_split_dict (optional): A dictionary for 1 split of the unlabeled data. Defaults to None.

    Returns:
        For labeled dataset:
            A dictionary of results
            A pretty print version of the results
        For unlabeld dataset:
    """
    if config_dict["reset_random_generator_after_every_epoch"]:
        # This is needed to make sure that each epoch starts with the same base random generator.
        _reset_random_number_generator(seed=config_dict["seed"])

    algorithm.train()
    torch.set_grad_enabled(True)

    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []

    if labeled_split_dict is not None:
        batches = labeled_split_dict["loader"]
        batches = tqdm(batches, disable=not is_master_process(config_dict=config_dict))
        last_batch_idx = len(batches) - 1
        for batch_idx, labeled_batch in enumerate(batches):
            logging.debug(f" -> batch_index: {batch_idx}, data = {labeled_batch[0][0,0,0,0]}")
            effective_batch_idx = (batch_idx + 1) / config_dict["gradient_accumulation_steps"]
            update_log_dict = effective_batch_idx % config_dict["log_every_n_batches"] == 0
            algorithm.update(
                labeled_batch=labeled_batch,
                is_epoch_end=(batch_idx == last_batch_idx),
                return_loss_components=update_log_dict,
                batch_idx=batch_idx,
            )
            # y_pred = detach_and_clone(batch_results["y_pred"])
            # if config_dict["process_outputs_function"] is not None:
            #     y_pred = process_outputs_functions[config_dict["process_outputs_function"]](y_pred)
            # epoch_y_pred.append(y_pred)

            # epoch_y_true.append(detach_and_clone(batch_results["y_true"]))
            # epoch_metadata.append(detach_and_clone(batch_results["metadata"]))

            if update_log_dict:
                if config_dict["verbose"]:
                    print(f"  Batch {batch_idx} \n")
                log_results(algorithm, labeled_split_dict, epoch, math.ceil(effective_batch_idx))

        epoch_y_true = torch.cat(epoch_y_true, dim=0)
        epoch_y_pred = torch.cat(epoch_y_pred, dim=0)
        epoch_metadata = torch.cat(epoch_metadata, dim=0)

        # Running the evaluation on all the training slides.
        results, results_str = labeled_split_dict["dataset"].eval(epoch_y_pred, epoch_y_true, epoch_metadata)

        if config_dict["scheduler_metric_split"] == labeled_split_dict["split"]:
            algorithm.step_schedulers(is_epoch=True, metrics=results, log_access=False)

        # log after updating the scheduler in case it needs to access the internal logs
        log_results(algorithm, labeled_split_dict, epoch, math.ceil(effective_batch_idx))

        labeled_split_dict["eval_logger"].log(results)
        if labeled_split_dict["verbose"]:
            print("  -> Epoch evaluation on all training slides:\n" + results_str)

        results["epoch"] = epoch
        return results, epoch_y_pred

    else:
        batches = unlabeled_split_dict["loader"]
        batches = tqdm(batches, disable=not is_master_process(config_dict=config_dict))
        last_batch_idx = len(batches) - 1

        for batch_idx, un_labeled_batch in enumerate(batches):
            logging.debug(f" -> batch_index: {batch_idx}, data = {un_labeled_batch[0][0,0,0,0]}")
            effective_batch_idx = (batch_idx + 1) / config_dict["gradient_accumulation_steps"]
            update_log_dict = effective_batch_idx % config_dict["log_every_n_batches"] == 0
            algorithm.update(
                labeled_batch=None,
                unlabeled_batch=un_labeled_batch,
                is_epoch_end=(batch_idx == last_batch_idx),
                epoch=epoch,
                return_loss_components=update_log_dict,
            )

            if update_log_dict:
                if config_dict["verbose"]:
                    print(f"  Batch {batch_idx} \n")
                log_results(algorithm, unlabeled_split_dict, epoch, math.ceil(effective_batch_idx))

        log_results(algorithm, unlabeled_split_dict, epoch, math.ceil(effective_batch_idx))
        return None


def _reset_random_number_generator(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_eval_epoch(
    algorithm: SingleModelAlgorithm,
    labeled_split_dict: Optional[Dict[str, Any]],
    config_dict: Dict[str, Any],
    epoch: int,
    unlabeled_split_dict: Optional[Dict[str, Any]] = None,
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
        unlabeled_split_dict: The split dict for the unlabeled dataset. Defaults to None.
        save_results: Save results in a csv. Needed for eval
        split: The type of the split for the labeled data, which can be one of `train`, `id_val`, `val`, `test`.
        normalization_split_dict: The split dict that we use for loading the normalization data for inference.

    Returns:
        For unlabeled data only:
            A dictionary of results.
            A pretty print version of the results
    """
    algorithm.eval()
    torch.set_grad_enabled(False)
    algorithm._is_training = False
    effective_batch_idx = 0

    if labeled_split_dict is not None:
        epoch_y_true = []
        epoch_y_pred = []
        epoch_metadata = []
        for batch_idx, labeled_batch in enumerate(tqdm(labeled_split_dict["loader"])):
            logging.debug(f" -> batch_index: {batch_idx}, data = {labeled_batch[0][0,0,0,0]}")

            with torch.no_grad():  # The BatchNorm2d is known to have unstable performance if using model.eval()
                batch_results = algorithm.evaluate(
                    labeled_batch,
                    # ddp_params=config_dict.get("ddp_params", None),
                )

            y_pred = detach_and_clone(batch_results["y_pred"])
            if config_dict["process_outputs_function"] is not None:
                y_pred = process_outputs_functions[config_dict["process_outputs_function"]](y_pred)
            epoch_y_pred.append(y_pred)

            epoch_y_true.append(detach_and_clone(batch_results["y_true"]))

            # TODO: handle this faster in DDP
            if "metadata_gpu" in batch_results:
                epoch_metadata.append(detach_and_clone(batch_results["metadata_gpu"]))
            else:
                epoch_metadata.append(detach_and_clone(batch_results["metadata"]))

            effective_batch_idx = batch_idx + 1
            if effective_batch_idx % config_dict["log_every_n_batches"] == 0:
                log_results(algorithm, labeled_split_dict, epoch, effective_batch_idx)

        epoch_y_true = torch.cat(epoch_y_true, dim=0)
        epoch_y_pred = torch.cat(epoch_y_pred, dim=0)
        epoch_metadata = torch.cat(epoch_metadata, dim=0)
        results, results_str = labeled_split_dict["dataset"].eval(epoch_y_pred, epoch_y_true, epoch_metadata)

        if config_dict["scheduler_metric_split"] == labeled_split_dict["split"]:
            algorithm.step_schedulers(is_epoch=True, metrics=results, log_access=True)

        # log after updating the scheduler in case it needs to access the internal logs
        log_results(algorithm, labeled_split_dict, epoch, math.ceil(effective_batch_idx))

        results["epoch"] = epoch
        labeled_split_dict["eval_logger"].log(results)
        if labeled_split_dict["verbose"]:
            print(results_str)

        # Skip saving train preds, since the train loader generally shuffles the data
        if save_results and split != "train":
            save_pred_if_needed(epoch_y_pred, labeled_split_dict, epoch, config_dict, is_best, force_save=True)
        return results, epoch_y_pred
    else:

        for batch_idx, unlabeled_batch in enumerate(tqdm(unlabeled_split_dict["loader"])):
            logging.debug(f" -> batch_index: {batch_idx}, data = {unlabeled_batch[0][0,0,0,0]}")

            effective_batch_idx = batch_idx + 1
            update_log_dict = effective_batch_idx % config_dict["log_every_n_batches"] == 0
            with torch.no_grad():  # The BatchNorm2d is known to have unstable performance if using model.eval()
                batch_results = algorithm.evaluate(
                    labeled_batch=None,
                    unlabeled_batch=unlabeled_batch,
                    # ddp_params=config_dict.get("ddp_params", None),
                    return_loss_components=update_log_dict,
                )

            if update_log_dict:
                log_results(algorithm, unlabeled_split_dict, epoch, effective_batch_idx)

        # log after updating the scheduler in case it needs to access the internal logs
        log_results(algorithm, unlabeled_split_dict, epoch, math.ceil(effective_batch_idx))
        return None


def evaluate_over_splits(algorithm, datasets, epoch, general_logger, config_dict, is_best, save_results) -> None:
    """Evaluate the algorithm over multiple data splits."""
    for split, dataset in datasets.items():
        _run_eval_epoch(
            algorithm=algorithm,
            labeled_split_dict=dataset,
            epoch=epoch,
            config_dict=config_dict,
            is_best=is_best,
            split=split,
            save_results=save_results,
        )
