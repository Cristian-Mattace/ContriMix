from typing import Any
from typing import Dict
from ip_drit.algorithms.single_model_algorithm import SingleModelAlgorithm
from ip_drit.logger import Logger
def train(
        algorithm: SingleModelAlgorithm,
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
        general_logger: The logger that is used to write the training logs.
        config_dict: The configuration dictionary.
        epoch_offset: The initial epoch offset.
    """
    for epoch in range(epoch_offset, config_dict['n_epochs']):
        general_logger.write(f"Epoch {epoch}: \n")