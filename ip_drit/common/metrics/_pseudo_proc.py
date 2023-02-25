"""A module that provides processing function for pseudo labels."""
import copy
from enum import auto
from enum import Enum
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F

from ._all_metrics import multiclass_logits_to_pred


class PseudoLabelProcessingFuncType(Enum):
    """An enum class that defines different types of processing functions for pseudo labels."""

    BINARY_LOGITS = auto()
    MULTICLASS_LOGITS = auto()
    IDENTITY = auto()


def pseudolabel_binary_logits(
    logits: torch.Tensor, confidence_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Generates pseudo label for the binary logits.

    Input:
        logits: Binary logits of size (batch_size, n_tasks). If an entry is >0, it means the prediction for that
            (example, task) is positive.
        confidence_threshold: A floating point value in the range of [0,1].

    Returns:
        unlabeled_y_pred: Filtered version of logits, discarding any rows (examples) that have no predictions with
            confidence above confidence_threshold.
        unlabeled_y_pseudo : Corresponding hard-pseudo-labeled version of logits. All entries with confidence below
            confidence_threshold are set to nan. All rows with no confident entries are discarded.
        pseudolabels_kept_frac : Fraction of (examples, tasks) not set to nan or discarded.
        mask: A mask used to discard predictions with confidence under the confidence threshold.
    """
    if len(logits.shape) != 2:
        raise ValueError("Logits must be 2-dimensional.")
    probs = 1 / (1 + torch.exp(-logits))
    mask = torch.max(probs, 1 - probs) >= confidence_threshold
    unlabeled_y_pseudo = (logits > 0).float()
    unlabeled_y_pseudo[~mask] = float("nan")
    pseudolabels_kept_frac = mask.sum() / mask.numel()  # mask is bool, so no .mean()
    example_mask = torch.any(~torch.isnan(unlabeled_y_pseudo), dim=1)
    unlabeled_y_pseudo = unlabeled_y_pseudo[example_mask]
    unlabeled_y_pred = logits[example_mask]

    return unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac, example_mask


def pseudolabel_multiclass_logits(
    logits: torch.Tensor, confidence_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Returns the pseudolabel of multi-class logits.

    Args:
        logits: Multi-class logits of size (batch_size, ..., n_classes).
        confidence_threshold: In [0,1]

    Returns:
        unlabeled_y_pred: Filtered version of logits, discarding any rows (examples) that have no predictions with
            confidence above confidence_threshold.
        unlabeled_y_pseudo: Corresponding hard-pseudo-labeled version of logits. All examples with confidence below
            confidence_threshold are discarded.
        pseudolabels_kept_frac : Fraction of examples not discarded.
        mask: Mask used to discard predictions with confidence under the confidence threshold.
    """
    mask = torch.max(F.softmax(logits, -1), -1)[0] >= confidence_threshold
    unlabeled_y_pseudo = multiclass_logits_to_pred(logits)
    unlabeled_y_pseudo = unlabeled_y_pseudo[mask]
    unlabeled_y_pred = logits[mask]
    pseudolabels_kept_frac = mask.sum() / mask.numel()  # mask is bool, so no .mean()
    return unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac, mask


def pseudolabel_identity(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]:
    """Returns the identity pseudolabels of the logits."""
    return logits, logits, 1, None


def pseudolabel_detection(
    preds: torch.Tensor, confidence_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Detecting the pseudo labels.

    Input:
        preds (List): List of len batch_size. Each entry is a dict containing
                      the keys 'boxes', 'labels', 'scores', and 'losses'
                      ('losses' is empty)
        confidence_threshold (float): In [0,1]
    """
    preds, pseudolabels_kept_frac = _mask_pseudolabels_detection(preds, confidence_threshold)
    unlabeled_y_pred = [
        {"boxes": pred["boxes"], "labels": pred["labels"], "scores": pred["scores"], "losses": pred["losses"]}
        for pred in preds
    ]
    unlabeled_y_pseudo = [{"boxes": pred["boxes"], "labels": pred["labels"]} for pred in preds]

    # Keep all examples even if they don't have any confident-enough predictions
    # They will be treated as empty images
    example_mask = torch.ones(len(preds), dtype=torch.bool)
    return unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac, example_mask


def pseudolabel_detection_discard_empty(preds: torch.Tensor, confidence_threshold: float):
    """Detecting pseudo labels of the predictions while discarding empty ones.

    Input:
        preds (List): List of len batch_size. Each entry is a dict containing
                      the keys 'boxes', 'labels', 'scores', and 'losses'
                      ('losses' is empty)
        confidence_threshold (float): In [0,1]
    """
    preds, pseudolabels_kept_frac = _mask_pseudolabels_detection(preds, confidence_threshold)
    unlabeled_y_pred = [
        {"boxes": pred["boxes"], "labels": pred["labels"], "scores": pred["scores"], "losses": pred["losses"]}
        for pred in preds
        if len(pred["labels"]) > 0
    ]
    unlabeled_y_pseudo = [
        {"boxes": pred["boxes"], "labels": pred["labels"]} for pred in preds if len(pred["labels"]) > 0
    ]
    example_mask = torch.tensor([len(pred["labels"]) > 0 for pred in preds])
    return unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac, example_mask


def _mask_pseudolabels_detection(preds: torch.Tensor, confidence_threshold: float) -> Tuple[torch.Tensor, float]:
    total_boxes = 0.0
    kept_boxes = 0.0

    preds = copy.deepcopy(preds)
    for pred in preds:
        mask = pred["scores"] >= confidence_threshold
        pred["boxes"] = pred["boxes"][mask]
        pred["labels"] = pred["labels"][mask]
        pred["scores"] = pred["scores"][mask]
        total_boxes += len(mask)
        kept_boxes += mask.sum()

    pseudolabels_kept_frac = kept_boxes / total_boxes
    return preds, pseudolabels_kept_frac


PSEUDO_LABEL_PROCESS_FUNC_BY_TYPE: Dict[PseudoLabelProcessingFuncType, Callable] = {
    PseudoLabelProcessingFuncType.BINARY_LOGITS: pseudolabel_binary_logits,
    PseudoLabelProcessingFuncType.MULTICLASS_LOGITS: pseudolabel_multiclass_logits,
    PseudoLabelProcessingFuncType.IDENTITY: pseudolabel_identity,
}
