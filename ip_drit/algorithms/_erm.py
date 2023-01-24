import torch
from typing import Any
from typing import Dict
from .single_model_algorithm import SingleModelAlgorithm
from ip_drit.models.wild_model_initializer import initialize_model_from_configuration
from ._utils import move_to
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.metrics import ElementwiseLoss

class ERM(SingleModelAlgorithm):
    """A class that implements the ERM algorithm.

    Args:
        config: A dictionary that defines the configuration to load the model.
        d_out: The dimension of the model output.
        grouper: A grouper object that defines the groups for which we compute/log statistics for.
        loss: The loss module

    References:
        https://binhu7.github.io/courses/ECE598/Spring2019/files/Lecture4.pdf
    """
    def __init__(
            self,
            config: Dict[str, Any],
            d_out: int,
            grouper: AbstractGrouper,
            loss: ElementwiseLoss,
            metric,
            n_train_steps: int):
        model = initialize_model_from_configuration(config, d_out, output_classifier=True)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self._use_unlabeled_y = config['use_unlabeled_y'] # Expect x,y,m from unlabeled loaders and train on the unlabeled y

    def process_batch(self, batch, unlabeled_batch=None):
        """
        Overrides single_model_algorithm.process_batch().
        ERM defines its own process_batch to handle if self.use_unlabeled_y is true.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch
                - unlabeled_g (Tensor): groups for unlabeled batch
                - unlabeled_metadata (Tensor): metadata for unlabeled batch
                - unlabeled_y_pred (Tensor): predictions for unlabeled batch for fully-supervised ERM experiments
                - unlabeled_y_true (Tensor): true labels for unlabeled batch for fully-supervised ERM experiments
        """
        x, y_true, metadata = batch
        x = move_to(x, self._device)
        y_true = move_to(y_true, self._device)
        g = move_to(self._grouper.metadata_to_group(metadata), self._device)

        outputs = self.get_model_output(x, y_true)

        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
        }
        if unlabeled_batch is not None:
            if self.use_unlabeled_y: # expect loaders to return x,y,m
                x, y, metadata = unlabeled_batch
                y = move_to(y, self.device)
            else:
                x, metadata = unlabeled_batch
            x = move_to(x, self.device)
            results['unlabeled_metadata'] = metadata
            if self.use_unlabeled_y:
                results['unlabeled_y_pred'] = self.get_model_output(x, y)
                results['unlabeled_y_true'] = y
            results['unlabeled_g'] = self.grouper.metadata_to_group(metadata).to(self.device)
        return results

    def objective(self, results):
        labeled_loss = self._loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        if self._use_unlabeled_y and 'unlabeled_y_true' in results:
            unlabeled_loss = self._loss.compute(
                results['unlabeled_y_pred'],
                results['unlabeled_y_true'],
                return_dict=False
            )
            lab_size = len(results['y_pred'])
            unl_size = len(results['unlabeled_y_pred'])
            return (lab_size * labeled_loss + unl_size * unlabeled_loss) / (lab_size + unl_size)
        else:
            return labeled_loss