"""A training script for training the student model for the noisy student algorithm.

Note that in this case, the labeled data is still strongly augmented.
Unlabeled data will be weakly augmeneted for the teacher but strongly augmented for the student to encourage
generalization.
"""
import argparse
import logging
import sys
from typing import Any
from typing import Dict

import torch.cuda
import torch.nn as nn

from eval_utils import infer_predictions
from ip_drit.algorithms import calculate_number_of_training_steps
from ip_drit.algorithms.initializer import initialize_algorithm
from ip_drit.algorithms.single_model_algorithm import ModelAlgorithm
from ip_drit.common.data_loaders import DataLoader
from ip_drit.common.data_loaders import get_eval_loader
from ip_drit.common.data_loaders import LoaderType
from ip_drit.common.grouper import MultiDatasetCombinatorialGrouper
from ip_drit.common.metrics import PseudoLabelProcessingFuncType
from ip_drit.datasets import AbstractUnlabelPublicDataset
from ip_drit.datasets.camelyon17 import CamelyonDataset
from ip_drit.datasets.unlabeled_camelyon import CamelyonUnlabeledDataset
from ip_drit.logger import Logger
from ip_drit.models.wild_model_initializer import initialize_model_from_configuration
from ip_drit.models.wild_model_initializer import WildModel
from ip_drit.patch_transform import initialize_transform
from ip_drit.patch_transform import TransformationType
from saving_utils import load
from saving_utils import load_model_state_dict_from_checkpoint
from script_utils import calculate_batch_size
from script_utils import configure_parser
from script_utils import configure_split_dict_by_names
from script_utils import dataset_and_log_location
from script_utils import generate_eval_model_path
from script_utils import use_data_parallel
from train_utils import train

package_path = "/jupyter-users-home/tan-2enguyen/intraminibatch_permutation_drit/"
if package_path not in sys.path:
    sys.path.append(package_path)


logging.getLogger().setLevel(logging.INFO)


def main():
    """Demo scripts for training, evaluation with Camelyon."""
    logging.info("Running the unlabel Camelyon 17 dataset benchmark.")
    parser = configure_parser()
    parser = _add_script_specific_flags(parser=parser)
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

    unlabeled_camelyon_dataset = CamelyonUnlabeledDataset(
        dataset_dir=all_dataset_dir / "unlabelled_camelyon17/", use_full_size=FLAGS.use_full_dataset
    )

    logger = Logger(fpath=str(log_dir / "log.txt"))
    target_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    algorithm = ModelAlgorithm.NOISY_STUDENT
    batch_size = calculate_batch_size(
        dataset_name=labeled_camelyon_dataset.dataset_name, algorithm=algorithm, run_on_cluster=FLAGS.run_on_cluster
    )

    config_dict: Dict[str, Any] = {
        "algo_log_metric": "accuracy",
        "algorithm": algorithm,
        "batch_size": batch_size,
        "device": target_device,
        "distinct_groups": False,  # If True, enforce groups sampled per batch are distinct.
        "eval_only": FLAGS.eval_only,  # If True, only evaluation will be performed, no training.
        "eval_epoch": FLAGS.eval_epoch,  # If not none, use this epoch for eval, else use the best epoch by val perf.
        "gradient_accumulation_steps": 1,
        "log_dir": str(log_dir),
        "log_every_n_batches": FLAGS.log_every_n_batches,
        "loss_function": "cross_entropy",
        "lr": 1e-3,
        "max_grad_norm": 0.5,
        "metric": "acc_avg",
        "model": WildModel.DENSENET121,
        "n_epochs": FLAGS.n_epochs,
        "n_groups_per_batch": FLAGS.num_groups_per_training_batch,  # 4
        "n_iters": 3,  # The number of iterations per training
        "noisystudent_add_dropout": True,  # If True, drop out will be added to the output of the student.
        "noisystudent_dropout_rate": 0.5,
        "optimizer": "AdamW",
        "optimizer_kwargs": {"SGD": {"momentum": 0.9}, "Adam": {}, "AdamW": {}},
        "pretrained_model_path": FLAGS.pretrained_model_path,
        "pseudolabels_proc_func": PseudoLabelProcessingFuncType.BINARY_LOGITS,
        "print_accuracy_on_labeled_data_from_teacher_model": False,
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
        "soft_pseudolabels": False,
        "target_resolution": None,  # Keep the original dataset resolution
        "train_group_by_fields": ["hospital"],
        "train_loader": LoaderType.GROUP,
        "transform": TransformationType.WEAK,
        "uniform_over_groups": FLAGS.sample_uniform_over_groups,  #
        "use_data_parallel": use_data_parallel(),
        "use_unlabeled_y": False,  # If true, unlabeled loaders will also the true labels for the unlabeled data.
        "val_metric_decreasing": False,
        "verbose": FLAGS.verbose,
        "weight_decay": 1e-2,
    }

    teacher_model = initialize_model_from_configuration(
        model_type=WildModel.DENSENET121, d_out=1, output_classifier=False
    )
    teacher_model.load_state_dict(
        state_dict=load_model_state_dict_from_checkpoint(
            model_path=FLAGS.teacher_model_path, start_str="_model.module."
        )
    )
    teacher_model.to(config_dict["device"])

    if config_dict["print_accuracy_on_labeled_data_from_teacher_model"]:
        _print_accuracy_on_trained_labeled_data_for_debugging(labeled_camelyon_dataset, teacher_model, config_dict)

    pseudo_label_eval_loader = get_eval_loader(
        loader_type=LoaderType.STANDARD,
        dataset=unlabeled_camelyon_dataset,
        batch_size=config_dict["batch_size"],
        run_on_cluster=config_dict["run_on_cluster"],
    )

    _update_pseudo_label(
        model=teacher_model,
        unlabeled_dataset=unlabeled_camelyon_dataset,
        config_dict=_construct_config_dict_pseudolabel_generation(config_dict),
        eval_loader=pseudo_label_eval_loader,
    )

    train_grouper = MultiDatasetCombinatorialGrouper(
        datasets=[labeled_camelyon_dataset, unlabeled_camelyon_dataset],
        groupby_fields=config_dict["train_group_by_fields"],
    )

    labeled_split_dict_by_names = configure_split_dict_by_names(
        full_dataset=labeled_camelyon_dataset, grouper=train_grouper, config_dict=config_dict
    )

    unlabeled_split_dict_by_names = configure_split_dict_by_names(
        full_dataset=unlabeled_camelyon_dataset, grouper=train_grouper, config_dict=config_dict
    )

    algorithm = initialize_algorithm(
        train_dataset=labeled_split_dict_by_names["train"]["dataset"],
        config=config_dict,
        num_train_steps=calculate_number_of_training_steps(
            config=config_dict, train_loader=labeled_split_dict_by_names["train"]["loader"]
        ),
        train_grouper=train_grouper,
    )

    if not config_dict["eval_only"]:
        logging.info("Training the student model!")
        for iter in range(config_dict["n_iters"]):
            logging.info(f"***************** Training for iter {iter} ***********************")
            train(
                algorithm=algorithm,
                labeled_split_dict_by_name=labeled_split_dict_by_names,
                unlabeled_split_dict_by_name=unlabeled_split_dict_by_names,
                general_logger=logger,
                config_dict=config_dict,
                epoch_offset=0,
            )

            logging.info("Updating the pseudo label ...")
            eval_model_path = generate_eval_model_path(
                eval_epoch=None, model_prefix=FLAGS.log_dir_cluster, seed=FLAGS.seed
            )
            load(algorithm, eval_model_path, device=config_dict["device"])
            _update_pseudo_label(
                model=algorithm.model,
                unlabeled_dataset=unlabeled_camelyon_dataset,
                config_dict=_construct_config_dict_pseudolabel_generation(config_dict),
                eval_loader=pseudo_label_eval_loader,
            )
    else:
        raise ValueError("Evaluation mode of the Noisy Student is not supported.")


def _add_script_specific_flags(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        default="/jupyter-users-home/tan-2enguyen/all_log_dir/noisy_student_teacher_training/"
        + "camelyon17_seed:0_epoch:best_model.pth",
        help="Path to the trained teacher model.",
    )
    return parser


def _print_accuracy_on_trained_labeled_data_for_debugging(
    labeled_camelyon_dataset: CamelyonDataset, teacher_model: nn.Module, config_dict: Dict[str, Any]
):
    labeled_camelyon_dataset.set_transform(
        val=initialize_transform(
            transform_name=TransformationType.RANDAUGMENT,
            full_dataset=labeled_camelyon_dataset,  # No need to supply the full dataset of we use Weak transform
            config_dict={"target_resolution": config_dict["target_resolution"], "randaugment_n": 2},
        )
    )
    infer_predictions(
        model=teacher_model,
        loader=get_eval_loader(
            loader_type=LoaderType.STANDARD,
            dataset=labeled_camelyon_dataset,
            batch_size=config_dict["batch_size"],
            run_on_cluster=config_dict["run_on_cluster"],
        ),
        config=config_dict,
        acc_cal=True,
    )


def _construct_config_dict_pseudolabel_generation(full_config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts a small configuration dictionary for model evaluation from the full dictionary."""
    return {
        "model": full_config_dict["model"],
        "target_resolution": full_config_dict["target_resolution"],
        "soft_pseudolabels": full_config_dict["soft_pseudolabels"],
        "pseudolabels_proc_func": PseudoLabelProcessingFuncType.BINARY_LOGITS,
        "self_training_threshold": 0.3,
        "run_on_cluster": full_config_dict["run_on_cluster"],
        "device": full_config_dict["device"],
    }


def _update_pseudo_label(
    model: nn.Module,
    unlabeled_dataset: AbstractUnlabelPublicDataset,
    config_dict: Dict[str, Any],
    eval_loader: DataLoader,
) -> None:
    """Generates the pseudo labels for unlabeled dataset.

    Args:
        model: The model used to generate the pseudo
        unlabeled_dataset: The full unlabeled dataset.
        target_device: The device to run the evaluation on.
        split: The split to extract the data from.
    """
    unlabeled_dataset.set_transform(
        val=initialize_transform(
            transform_name=TransformationType.WEAK,
            full_dataset=unlabeled_dataset,  # No need to supply the full dataset of we use Weak transform
            config_dict={"target_resolution": config_dict["target_resolution"]},
        )
    )

    unlabeled_dataset.pseudolabels = infer_predictions(model=model, loader=eval_loader, config=config_dict).to(
        torch.device("cpu")
    )

    # Prepare for the next training
    unlabeled_dataset.set_transform(
        val=initialize_transform(
            transform_name=TransformationType.RANDAUGMENT,
            full_dataset=unlabeled_dataset,
            config_dict={"target_resolution": config_dict["target_resolution"], "randaugment_n": 3},
        )
    )


if __name__ == "__main__":
    main()
