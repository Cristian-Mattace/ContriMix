"""A training script for training the teacher model for the noisy student.

Note that the teacher model is only trained with the labeled data with strong augmentation.
The algorithm will be the ERM algorithm.
"""
import logging
import sys
from typing import Any
from typing import Dict

import torch.cuda

from ip_drit.algorithms import calculate_number_of_training_steps
from ip_drit.algorithms.initializer import initialize_algorithm
from ip_drit.algorithms.single_model_algorithm import ModelAlgorithm
from ip_drit.common.data_loaders import LoaderType
from ip_drit.common.grouper import CombinatorialGrouper
from ip_drit.datasets.camelyon17 import CamelyonDataset
from ip_drit.logger import Logger
from ip_drit.models.wild_model_initializer import WildModel
from ip_drit.patch_transform import TransformationType
from saving_utils import load
from script_utils import calculate_batch_size
from script_utils import configure_parser
from script_utils import configure_split_dict_by_names
from script_utils import dataset_and_log_location
from script_utils import use_data_parallel
from train_utils import evaluate_over_splits
from train_utils import train

package_path = "/jupyter-users-home/tan-2enguyen/intraminibatch_permutation_drit/"
if package_path not in sys.path:
    sys.path.append(package_path)


logging.getLogger().setLevel(logging.INFO)


def main():
    """Demo scripts for training, evaluation with Camelyon."""
    logging.info("Running the unlabel Camelyon 17 dataset benchmark.")
    parser = configure_parser()
    FLAGS = parser.parse_args()

    all_dataset_dir, log_dir = dataset_and_log_location(
        FLAGS.run_on_cluster,
        FLAGS.log_dir_cluster,
        FLAGS.log_dir_local,
        FLAGS.dataset_dir_cluster,
        FLAGS.dataset_dir_local,
    )

    all_dataset_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    labeled_camelyon_dataset = CamelyonDataset(
        dataset_dir=all_dataset_dir / "camelyon17/", use_full_size=FLAGS.use_full_dataset
    )

    logger = Logger(fpath=str(log_dir / "log.txt"))

    algorithm = ModelAlgorithm.ERM
    batch_size = calculate_batch_size(algorithm=algorithm, run_on_cluster=FLAGS.run_on_cluster)
    config_dict: Dict[str, Any] = {
        "algo_log_metric": "accuracy",
        "algorithm": algorithm,
        "batch_size": batch_size,
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "distinct_groups": False,  # If True, enforce groups sampled per batch are distinct.
        "eval_only": FLAGS.eval_only,  # If True, only evaluation will be performed, no training.
        "eval_epoch": FLAGS.eval_epoch,  # If not none, use this epoch for eval, else use the best epoch by val perf.
        "gradient_accumulation_steps": 1,
        "log_dir": str(log_dir),
        "log_every_n_batches": FLAGS.log_every_n_batches,
        "loss_function": "multitask_bce",
        "lr": 1e-3,
        "max_grad_norm": 0.5,
        "metric": "acc_avg",
        "model": WildModel.DENSENET121,
        "n_epochs": FLAGS.n_epochs,
        "n_groups_per_batch": FLAGS.num_groups_per_training_batch,  # 4
        "optimizer": "AdamW",
        "optimizer_kwargs": {"SGD": {"momentum": 0.9}, "Adam": {}, "AdamW": {}},
        "pretrained_model_path": FLAGS.pretrained_model_path,
        "randaugment_n": 2,  # FLAGS.randaugment_n,
        "report_batch_metric": True,
        "reset_random_generator_after_every_epoch": FLAGS.reset_random_generator_after_every_epoch,
        "run_on_cluster": FLAGS.run_on_cluster,
        # Saving parameters
        "save_step": 5,
        "save_last": False,
        "save_best": True,
        "save_pred": True,
        "scheduler_metric_split": "val",
        "scheduler": "linear_schedule_with_warmup",
        "scheduler_kwargs": {"num_warmup_steps": 3},
        "scheduler_metric_name": "scheduler_metric_name",
        "seed": FLAGS.seed,
        "target_resolution": None,  # Keep the original dataset resolution
        "train_group_by_fields": ["hospital", "y"],
        "train_loader": LoaderType.GROUP,
        "transform": TransformationType.RANDAUGMENT,  # Teacher is always trained with strong augmentation.
        "uniform_over_groups": FLAGS.sample_uniform_over_groups,  #
        "use_data_parallel": use_data_parallel(),
        "soft_pseudolabels": FLAGS.soft_pseudolabels,
        "use_unlabeled_y": False,  # If true, unlabeled loaders will also the true labels for the unlabeled data.
        "val_metric_decreasing": False,
        "verbose": FLAGS.verbose,
        "weight_decay": 1e-2,
    }

    train_grouper = CombinatorialGrouper(
        dataset=labeled_camelyon_dataset, groupby_fields=config_dict["train_group_by_fields"]
    )

    labeled_split_dict_by_names = configure_split_dict_by_names(
        full_dataset=labeled_camelyon_dataset, grouper=train_grouper, config_dict=config_dict
    )

    algorithm = initialize_algorithm(
        config=config_dict,
        num_train_steps=calculate_number_of_training_steps(
            config=config_dict, train_loader=labeled_split_dict_by_names["train"]["loader"]
        ),
        train_grouper=train_grouper,
    )

    if not config_dict["eval_only"]:
        logging.info("Training the teacher model for noisy student!")
        train(
            algorithm=algorithm,
            labeled_split_dict_by_name=labeled_split_dict_by_names,
            unlabeled_split_dict_by_name=None,
            general_logger=logger,
            config_dict=config_dict,
            epoch_offset=0,
        )
    else:
        logging.info("Evaluation mode on the training data!")
        best_epoch, _ = load(algorithm, config_dict["pretrained_model_path"], device=config_dict["device"])
        epoch = best_epoch if FLAGS.eval_epoch is None else FLAGS.eval_epoch
        evaluate_over_splits(
            algorithm=algorithm,
            datasets=labeled_split_dict_by_names,
            epoch=epoch,
            general_logger=logger,
            config_dict=config_dict,
            is_best=epoch == best_epoch,
            save_results=True,
        )


if __name__ == "__main__":
    main()
