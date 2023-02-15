"""A scripts to run the benchmark for the Camelyon dataset."""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Tuple

package_path = "/jupyter-users-home/dinkar-2ejuyal/intraminibatch_permutation_drit/"
if package_path not in sys.path:
    sys.path.append(package_path)

from script_utils import num_of_available_devices
import torch.cuda
from ip_drit.algorithms.initializer import initialize_algorithm
from ip_drit.algorithms.single_model_algorithm import ModelAlgorithm
from ip_drit.common.grouper import CombinatorialGrouper
from ip_drit.datasets.camelyon17 import CamelyonDataset
from ip_drit.logger import Logger
from ip_drit.models.wild_model_initializer import WildModel
from ip_drit.common.data_loaders import LoaderType
from ip_drit.patch_transform import TransformationType
from script_utils import configure_split_dict_by_names
from script_utils import parse_bool
from script_utils import use_data_parallel
from train_utils import train, evaluate_over_splits, _run_eval_epoch
from saving_utils import load

logging.getLogger().setLevel(logging.INFO)


def main():
    """Demo scripts for training, evaluation with Camelyon."""
    logging.info("Running the Camelyon 17 dataset benchmark.")
    parser = _configure_parser()
    FLAGS = parser.parse_args()

    all_dataset_dir, log_dir = _dataset_and_log_location(
        FLAGS.run_on_cluster,
        FLAGS.log_dir_cluster,
        FLAGS.log_dir_local,
        FLAGS.dataset_dir_cluster,
        FLAGS.dataset_dir_local,
    )

    all_dataset_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    camelyon_dataset = CamelyonDataset(
        dataset_dir=all_dataset_dir / "camelyon17/", use_full_size=FLAGS.use_full_dataset
    )

    config_dict: Dict[str, Any] = {
        "algorithm": ModelAlgorithm.CONTRIMIX,
        "model": WildModel.DENSENET121,
        "transform": TransformationType.WEAK_NORMALIZE_TO_0_1,
        "target_resolution": None,  # Keep the original dataset resolution
        "scheduler_metric_split": "val",
        "train_group_by_fields": ["hospital"],
        "loss_function": "multitask_bce",
        "algo_log_metric": "accuracy",
        "log_dir": str(log_dir),
        "gradient_accumulation_steps": 1,
        "n_epochs": FLAGS.n_epochs,
        "log_every_n_batches": FLAGS.log_every_n_batches,
        "train_loader": LoaderType.GROUP,
        "reset_random_generator_after_every_epoch": FLAGS.reset_random_generator_after_every_epoch,
        "batch_size": calculate_batch_size(FLAGS.run_on_cluster),
        "uniform_over_groups": FLAGS.sample_uniform_over_groups,  #
        "distinct_groups": False,  # If True, enforce groups sampled per batch are distinct.
        "n_groups_per_batch": FLAGS.num_groups_per_training_batch,  # 4
        "scheduler": "linear_schedule_with_warmup",
        "scheduler_kwargs": {"num_warmup_steps": 3},
        "scheduler_metric_name": "scheduler_metric_name",
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "optimizer_kwargs": {"SGD": {"momentum": 0.9}, "Adam": {}, "AdamW": {}},
        "max_grad_norm": 0.5,
        "use_data_parallel": use_data_parallel(),
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "use_unlabeled_y": False,  # If true, unlabeled loaders will also the true labels for the unlabeled data.
        "verbose": FLAGS.verbose,
        "report_batch_metric": True,
        "metric": "acc_avg",
        "val_metric_decreasing": False,
        # Saving parameters
        "save_step": 5,
        "seed": FLAGS.seed,
        "save_last": False,
        "save_best": True,
        "save_pred": True,
        "eval_only": FLAGS.eval_only,  # If True, only evaluation will be performed, no training.
        "eval_epoch": FLAGS.eval_epoch,  # If not none, use this epoch for eval, else use the best epoch by val perf.
        "pretrained_model_path": FLAGS.pretrained_model_path,
    }

    logger = Logger(fpath=str(log_dir / "log.txt"))

    train_grouper = CombinatorialGrouper(dataset=camelyon_dataset, groupby_fields=config_dict["train_group_by_fields"])

    split_dict_by_names = configure_split_dict_by_names(
        full_dataset=camelyon_dataset, grouper=train_grouper, config_dict=config_dict
    )
    algorithm = initialize_algorithm(
        config=config_dict, split_dict_by_name=split_dict_by_names, train_grouper=train_grouper
    )

    if not config_dict["eval_only"]:
        logging.info("Training mode!")
        train(
            algorithm=algorithm,
            split_dict_by_name=split_dict_by_names,
            general_logger=logger,
            config_dict=config_dict,
            epoch_offset=0,
        )
    if config_dict["eval_only"] or FLAGS.run_eval_after_train:
        logging.info("Evaluation mode!")
        eval_model_path = _generate_eval_model_path(FLAGS.eval_epoch, FLAGS.model_prefix, FLAGS.seed)
        best_epoch, best_val_metric = load(algorithm, eval_model_path, device=config_dict["device"])
        epoch = best_epoch if FLAGS.eval_epoch is None else FLAGS.eval_epoch
        is_best = epoch == best_epoch
        evaluate_over_splits(
            algorithm=algorithm,
            datasets=split_dict_by_names,
            epoch=epoch,
            general_logger=logger,
            config_dict=config_dict,
            is_best=is_best,
            save_results=True,
        )


def _configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_on_cluster",
        type=parse_bool,
        default=True,
        const=True,
        nargs="?",
        help="A string to specify where to run the code. "
        + "Defaults to True, in which case the code will be run on the cluster.",
    )
    parser.add_argument(
        "--verbose",
        type=parse_bool,
        default=False,
        const=True,
        nargs="?",
        help="A string to specify if we want to display several images or not. Defaults to False.",
    )

    parser.add_argument(
        "--sample_uniform_over_groups",
        type=parse_bool,
        default=True,
        const=True,
        nargs="?",
        help="A string to specify how the sampling should be done. If True, sample examples such that batches are "
        + "uniform over groups. If False, sample using the group probabilities that is proportional to group size.",
    )
    parser.add_argument(
        "--num_groups_per_training_batch", type=int, default=3, help="The number of groups per training batch."
    )

    parser.add_argument(
        "--log_every_n_batches", type=int, default=3, help="The number of batches to log once. Defaults to 3."
    )

    parser.add_argument(
        "--use_full_dataset",
        type=parse_bool,
        default=False,
        help="If True, full dataset will be used. Defaults to False.",
    )

    parser.add_argument(
        "--log_dir_cluster",
        type=str,
        default="/jupyter-users-home/tan-2enguyen/all_log_dir",
        help="Directory for logging in cluster",
    )

    parser.add_argument(
        "--log_dir_local", type=str, default="/Users/tan.nguyen/", help="Directory for logging in local"
    )

    parser.add_argument(
        "--dataset_dir_cluster",
        type=str,
        default="/jupyter-users-home/tan-2enguyen/datasets",
        help="Directory for datasets in cluster",
    )

    parser.add_argument(
        "--dataset_dir_local", type=str, default="/Users/tan.nguyen/datasets", help="Directory for datasets in local"
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed, use values from 0 to 9.")

    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer")

    parser.add_argument("--run_eval_after_train", type=parse_bool, default=False, help="Optional: run eval after train")

    parser.add_argument(
        "--eval_only",
        type=parse_bool,
        default=False,
        help="If true, only evaluation is done. Defaults to False, in which case, training will be performed.",
    )

    parser.add_argument(
        "--eval_epoch",
        type=int,
        default=None,
        help="If specified, this epoch is used for evaluation, else the epoch with the best metric is used.",
    )

    parser.add_argument(
        "--model_prefix",
        type=str,
        default="/jupyter-users-home/dinkar-2ejuyal/all_log_dir/erm_camelyon",
        help="The prefix to the model path for evaluation mode. "
        "It will be appended by either best_model or a specific epoch number to generate evaluation model path.",
    )

    parser.add_argument("--n_epochs", type=int, default=30, help="Number of epochs to train for")
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="The path to a pretrained model.")

    parser.add_argument(
        "--reset_random_generator_after_every_epoch",
        type=parse_bool,
        default=False,
        help="If True, the random number generator will be restarted using the same seed after every epoches.",
    )

    return parser


def _dataset_and_log_location(
    run_on_cluster: bool, log_dir_cluster: str, log_dir_local: str, dataset_dir_cluster: str, dataset_dir_local: str
) -> Tuple[Path, Path]:
    if run_on_cluster:
        return Path(dataset_dir_cluster), Path(log_dir_cluster)
    else:
        return Path(dataset_dir_local), Path(log_dir_local)


def _generate_eval_model_path(eval_epoch: int, model_prefix: str, seed: int) -> str:
    if eval_epoch is None:
        eval_model_path: str = os.path.join(model_prefix, f"camelyon17_seed:{seed}_epoch:best_model.pth")
    else:
        eval_model_path: str = os.path.join(model_prefix, f"camelyon17_seed:{seed}_epoch:{eval_epoch}_model.pth")
    return eval_model_path


def calculate_batch_size(run_on_cluster: bool) -> int:
    """Returns the minibatchsize per GPU."""
    num_devices = num_of_available_devices()
    logging.info(f"Number of training devices = {num_devices}.")
    if run_on_cluster:
        batch_size_per_gpu = 210
    else:
        batch_size_per_gpu = 153
    batch_size = batch_size_per_gpu * num_devices
    logging.info(f"Using a batch size of {batch_size} for {batch_size_per_gpu}/device * {num_devices} device(s).")
    return batch_size


if __name__ == "__main__":
    main()
