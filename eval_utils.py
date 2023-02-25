"""A helper module for evaluation."""
import logging
from typing import Any
from typing import Dict
from typing import List

import torch
import torch.nn as nn
from torch.nn import DataParallel
from tqdm import tqdm

from ip_drit.common.data_loaders import DataLoader
from ip_drit.common.metrics import binary_logits_to_pred
from ip_drit.common.metrics import PSEUDO_LABEL_PROCESS_FUNC_BY_TYPE


def infer_predictions(
    model: nn.Module, loader: DataLoader, config: Dict[str, Any], process_pseudo_label: bool = True
) -> torch.Tensor:
    """Infers the labels from a model.

    Args:
        model: A torch module that contains the trained model.
        loader: A data loader.
        config: A configuration dictionary.
        process_pseudo_label (optional): If True, will process pseudo labels. Defaults to True.

    Returns:
        A tensor of predicted labels.
    """
    model.eval()

    y_pred: List[float] = []
    logging.info(f"Evaluating the model on {len(loader)} batches.")
    for batch in tqdm(loader):
        x = batch[0].to(config["device"])
        with torch.no_grad():
            out = model(x)
            if process_pseudo_label:
                if config["soft_pseudolabels"]:
                    _, y_out, _, _ = PSEUDO_LABEL_PROCESS_FUNC_BY_TYPE[config["pseudolabels_proc_func"]](
                        out, confidence_threshold=config["self_training_threshold"]
                    )
                else:
                    y_out = binary_logits_to_pred(out)
            else:
                raise RuntimeError(f"infer_predictions() does not support non-pseudo label processing")
            y_pred.extend(y_out.detach().clone())
    return torch.cat(y_pred, dim=0)
