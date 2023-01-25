import math
import os
import pandas as pd
from typing import Any
from typing import Dict
from typing import Hashable
from typing import List
from typing import Tuple
from typing import Union
import torch
from tqdm import tqdm
from ip_drit.algorithms.single_model_algorithm import SingleModelAlgorithm
from ip_drit.logger import Logger
from ip_drit.common.metrics import binary_logits_to_pred

def train(
        algorithm: SingleModelAlgorithm,
        split_dict_by_name: Dict[str, Dict],
        general_logger: Logger,
        config_dict: Dict[str, Any],
        epoch_offset: int,
) -> None:
    """Trains the model.

    Train loop that, each epoch:
        - Steps an algorithm on the datasets['train'] split and the unlabeled split
        - Evaluates the algorithm on the datasets['val'] split
        - Saves models / preds with frequency according to the configs
        - Evaluates on any other specified splits in the configs
    Assumes that the datasets dict contains labeled data.

    Args:
        algorithm: The algorithm to use.
        split_dict_by_name: A dictionary that defines different things (dataset, loaders etc) for different split name.
        general_logger: The logger that is used to write the training logs.
        epoch: The index of the current epoch.
        config_dict: The configuration dictionary.
        epoch_offset: The initial epoch offset.
    """
    best_val_metric = None
    for epoch in range(epoch_offset, config_dict['n_epochs']):
        general_logger.write(f"Epoch {epoch}: \n")

        # First run training
        _run_train_epoch(
            algorithm=algorithm,
            split_dict=split_dict_by_name['train'],
            general_logger=general_logger,
            epoch=epoch,
            config_dict=config_dict,
        )

        val_results, y_pred = _run_eval_epoch(
            algorithm=algorithm,
            split_dict=split_dict_by_name['ood_val'],
            general_logger=general_logger,
            epoch=epoch,
            config_dict=config_dict,
        )

        curr_val_metric = val_results[config_dict['val_metric']]
        general_logger.write(f"Validation {config_dict['val_metric']}: {curr_val_metric:.3f}\n")

        if best_val_metric is None:
            is_best = True
        else:
            if config_dict['val_metric_decreasing']:
                is_best = curr_val_metric < best_val_metric
                best_val_metric = curr_val_metric
            else:
                is_best = curr_val_metric > best_val_metric
                best_val_metric = curr_val_metric

        if is_best:
            general_logger.write(f'Epoch {epoch} has the best validation performance so far.\n')

        _save_model_if_needed(algorithm, split_dict_by_name['ood_val'], epoch, config_dict, is_best, best_val_metric)
        _save_pred_if_needed(y_pred, split_dict_by_name['ood_val'], epoch, config_dict, is_best)
        general_logger.write('\n')


def _run_train_epoch(
        algorithm: SingleModelAlgorithm,
        split_dict: Dict[str, Any],
        general_logger: Logger,
        config_dict: Dict[str, Any],
        epoch: int,
    ) -> Tuple[Dict, str]:
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

    # Not preallocating memory is slower
    # but makes it easier to handle different types of data loaders
    # (which might not return exactly the same number of examples per epoch)
    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []

    if 'loader' not in split_dict:
        raise RuntimeError("The loader is not in split dict!")

    batches = split_dict['loader']
    batches = tqdm(batches)
    last_batch_idx = len(batches) - 1

    # so we manually increment batch_idx
    for batch_idx, labeled_batch in enumerate(batches):
        batch_results = algorithm.update(labeled_batch, is_epoch_end=(batch_idx==last_batch_idx))

        # These tensors are already detached, but we need to clone them again
        # Otherwise they don't get garbage collected properly in some versions
        # The extra detach is just for safety
        # (they should already be detached in batch_results)
        epoch_y_true.append(_detach_and_clone(batch_results['y_true']))
        epoch_y_pred.append(_detach_and_clone(batch_results['y_pred']))
        epoch_metadata.append(_detach_and_clone(batch_results['metadata']))

        effective_batch_idx = (batch_idx + 1) / config_dict['gradient_accumulation_steps']
        if effective_batch_idx % config_dict['log_every'] == 0:
            _log_results(
                algorithm=algorithm,
                split_dict=split_dict,
                general_logger=general_logger,
                epoch=epoch,
                effective_batch_idx=math.ceil(effective_batch_idx),
            )

    epoch_y_true = _collate_list(epoch_y_true)
    epoch_y_pred = _collate_list(epoch_y_pred)
    epoch_metadata = _collate_list(epoch_metadata)

    results, results_str = split_dict['dataset'].eval(
        epoch_y_pred,
        epoch_y_true,
        epoch_metadata,
        prediction_fn=binary_logits_to_pred,
    )

    if config_dict['scheduler_metric_split'] == split_dict['split']:
        algorithm.step_schedulers(
            is_epoch=True,
            metrics=results,
            log_access=False)

    # log after updating the scheduler in case it needs to access the internal logs
    _log_results(algorithm, split_dict, general_logger, epoch, math.ceil(effective_batch_idx))

    results['epoch'] = epoch
    split_dict['eval_logger'].log(results)
    if split_dict['verbose']:
        general_logger.write('Epoch eval:\n')
        general_logger.write(results_str)

    return results, epoch_y_pred

def _detach_and_clone(obj: Union[torch.Tensor, Dict[Hashable, Any], List[Any], float, int]):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: _detach_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_detach_and_clone(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        raise TypeError("Invalid type for detach_and_clone")

def _collate_list(vec: List[torch.Tensor]) -> torch.Tensor:
    """
    If vec is a list of Tensors, it concatenates them all along the first dimension.

    If vec is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.

    If vec is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    return torch.cat(vec, dim=0)

def _log_results(
        algorithm: SingleModelAlgorithm,
        split_dict: Dict[str, Any],
        general_logger: Logger,
        epoch:int,
        effective_batch_idx: int) -> None:
    if algorithm.has_log:
        log = algorithm.get_log()
        log['epoch'] = epoch
        log['batch'] = effective_batch_idx
        split_dict['algo_logger'].log(log)
        if split_dict['verbose']:
            general_logger.write(algorithm.get_pretty_log_str())
        algorithm.reset_log()


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

    # Not preallocating memory is slower
    # but makes it easier to handle different types of data loaders
    # (which might not return exactly the same number of examples per epoch)
    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []

    if 'loader' not in split_dict:
        raise RuntimeError("The loader is not in split dict!")

    batches = split_dict['loader']
    batches = tqdm(batches)

    # so we manually increment batch_idx
    for batch_idx, labeled_batch in enumerate(batches):
        batch_results = algorithm.evaluate(labeled_batch)

        epoch_y_true.append(_detach_and_clone(batch_results['y_true']))
        epoch_y_pred.append(_detach_and_clone(batch_results['y_pred']))
        epoch_metadata.append(_detach_and_clone(batch_results['metadata']))

        effective_batch_idx = batch_idx + 1
        if effective_batch_idx % config_dict['log_every'] == 0:
            _log_results(
                algorithm=algorithm,
                split_dict=split_dict,
                general_logger=general_logger,
                epoch=epoch,
                effective_batch_idx=effective_batch_idx,
            )

    epoch_y_true = _collate_list(epoch_y_true)
    epoch_y_pred = _collate_list(epoch_y_pred)
    epoch_metadata = _collate_list(epoch_metadata)

    results, results_str = split_dict['dataset'].eval(
        epoch_y_pred,
        epoch_y_true,
        epoch_metadata,
        prediction_fn=binary_logits_to_pred,
    )

    if config_dict['scheduler_metric_split'] == split_dict['split']:
        algorithm.step_schedulers(
            is_epoch=True,
            metrics=results,
            log_access=True)

    # log after updating the scheduler in case it needs to access the internal logs
    _log_results(algorithm, split_dict, general_logger, epoch, math.ceil(effective_batch_idx))

    results['epoch'] = epoch
    split_dict['eval_logger'].log(results)
    if split_dict['verbose']:
        general_logger.write('Epoch eval:\n')
        general_logger.write(results_str)

    return results, epoch_y_pred

def _save_model_if_needed(
        algorithm: SingleModelAlgorithm,
        split_dict: Dict[str, Any],
        epoch: int,
        config_dict: Dict[str, Any],
        is_best: bool,
        best_val_metric: float,
    ) -> None:
    prefix = _get_model_prefix(split_dict['dataset'].dataset_name, config_dict)
    if config_dict['save_step'] is not None and (epoch + 1) % config_dict['save_step'] == 0:
        _save_model(algorithm, epoch, best_val_metric, prefix + f'epoch:{epoch}_model.pth')
    if config_dict['save_last']:
        _save_model(algorithm, epoch, best_val_metric, prefix + 'epoch:last_model.pth')
    if config_dict['save_best'] and is_best:
        _save_model(algorithm, epoch, best_val_metric, prefix + 'epoch:best_model.pth')

def _get_model_prefix(dataset_name: str, config_dict: Dict[str, Any]):
    return os.path.join(
        config_dict['log_dir'],
        f"{dataset_name}_seed:{config_dict['seed']}_")

def _save_model(algorithm: SingleModelAlgorithm, epoch: int, best_val_metric: float, path: str) -> None:
    """Save the model checkpoint.

    Args:
        algorithm: The algorithm used to train the model.
        epoch: The epoch number.
        best_val_metric: The best validation metric.
        path: The path to the saving folder.
    """
    state = {}
    state['algorithm'] = algorithm.state_dict()
    state['epoch'] = epoch
    state['best_val_metric'] = best_val_metric
    torch.save(state, path)

def _save_pred_if_needed(y_pred, dataset, epoch: int, config_dict: Dict[str, Any], is_best, force_save=False) -> None:
    if config_dict['save_pred']:
        prefix: str = _get_pred_prefix(dataset, config_dict)
        if force_save or (config_dict['save_step'] is not None and (epoch + 1) % config_dict['save_step'] == 0):
            _save_pred(y_pred, prefix + f'epoch:{epoch}_pred')
        if (not force_save) and config_dict['save_last']:
            _save_pred(y_pred, prefix + f'epoch:last_pred')
        if config_dict['save_best'] and is_best:
            _save_pred(y_pred, prefix + f'epoch:best_pred')

def _get_pred_prefix(split_dict: Dict[str, Any], config_dict: Dict[str, Any]):
    dataset_name = split_dict['dataset'].dataset_name
    split = split_dict['split']
    replicate_str = f"seed:{config_dict['seed']}"
    prefix = os.path.join(
        config_dict['log_dir'],
        f"{dataset_name}_split:{split}_{replicate_str}_")
    return prefix

def _save_pred(y_pred: torch.Tensor, path_prefix: str):
    # Single tensor
    if torch.is_tensor(y_pred):
        df = pd.DataFrame(y_pred.numpy())
        df.to_csv(path_prefix + '.csv', index=False, header=False)
    # Dictionary
    elif isinstance(y_pred, dict) or isinstance(y_pred, list):
        torch.save(y_pred, path_prefix + '.pth')
    else:
        raise TypeError("Invalid type for save_pred")
