"""A scripts that adopts the RxRx1 dataset for Contrimix.

Reference:
    [1]. https://worksheets.codalab.org/bundles/0x7d33860545b64acca5047396d42c0ea0
    [2]. https://worksheets.codalab.org/worksheets/0x231465946de84c06976a0f6626597024
    [3]. https://github.com/huaxiuyao/LISA
"""
import logging
import sys
from typing import Any
from typing import Dict

import torch
import torch.nn as nn
import wilds

package_path = "/jupyter-users-home/dinkar-2ejuyal/intraminibatch_permutation_drit/"
if package_path not in sys.path:
    sys.path.append(package_path)

from wilds.common.grouper import CombinatorialGrouper
from script_utils import log_group_data

import torch.multiprocessing

from script_utils import dataset_and_log_location, generate_eval_model_path
from script_utils import configure_parser
from script_utils import set_seed
import torch.cuda
from ip_drit.algorithms.initializer import initialize_algorithm
from ip_drit.algorithms.single_model_algorithm import ModelAlgorithm
from ip_drit.algorithms._contrimix_utils import ContrimixTrainingMode

from ip_drit.logger import Logger
from ip_drit.models.wild_model_initializer import WildModel
from ip_drit.common.data_loaders import LoaderType
from ip_drit.patch_transform import TransformationType
from script_utils import configure_split_dict_by_names
from train_utils import train, evaluate_over_splits
from saving_utils import load
from ip_drit.algorithms import calculate_number_of_training_steps
from ip_drit.algorithms._contrimix_utils import ContriMixMixingType
from ip_drit.loss import ContriMixAggregationType

logging.getLogger().setLevel(logging.INFO)

from script_utils import set_visible_gpus

parser = configure_parser()
FLAGS = parser.parse_args()
set_visible_gpus(gpu_ids=FLAGS.gpu_ids)  # This one will need to be call outside the program.


def main():
    """Demo scripts for training, evaluation with the labeled RxRx1 data."""
    print("Running the RxRx1 dataset benchmark with the Contrimix algorithm for backbone training")

    all_dataset_dir, log_dir = dataset_and_log_location(
        FLAGS.run_on_cluster,
        FLAGS.log_dir_cluster,
        FLAGS.log_dir_local,
        FLAGS.dataset_dir_cluster,
        FLAGS.dataset_dir_local,
    )

    all_dataset_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    config_dict: Dict[str, Any] = {
        "algorithm": ModelAlgorithm.CONTRIMIX,
        "model": WildModel.RESNET50,
        "transform": TransformationType.WEAK_NORMALIZE_TO_0_1,
        "target_resolution": None,  # Keep the original dataset resolution
        "scheduler_metric_split": "val",
        "train_group_by_fields": ["experiment"],
        "loss_function": "cross_entropy",
        "algo_log_metric": "accuracy",
        "log_dir": str(log_dir),
        "gradient_accumulation_steps": 1,
        "n_epochs": 90,
        "log_every_n_batches": FLAGS.log_every_n_batches,
        "train_loader": LoaderType.STANDARD,
        "reset_random_generator_after_every_epoch": False,
        "model_kwargs": {"pretrained": True},  # RxRx1 always use the pre-trained model.
        "batch_size": 72,
        "run_on_cluster": FLAGS.run_on_cluster,
        "uniform_over_groups": False,  #
        "distinct_groups": True,  # If True, enforce groups sampled per batch are distinct.
        "n_groups_per_batch": 9,
        "scheduler": "cosine_schedule_with_warmup",
        "scheduler_kwargs": {"num_warmup_steps": 5415},
        "scheduler_metric_name": "scheduler_metric_name",
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "optimizer_kwargs": {"SGD": {"momentum": 0.9}, "Adam": {}, "AdamW": {}},
        "max_grad_norm": None,
        "use_ddp_over_dp": False,
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "use_unlabeled_y": False,  # If true, unlabeled loaders will also the true labels for the unlabeled data.
        "verbose": FLAGS.verbose,
        "report_batch_metric": True,
        "metric": "acc_avg",
        "val_metric_decreasing": False,
        # Saving parameters
        "save_step": 10,
        "seed": FLAGS.seed,
        "save_last": False,
        "save_best": True,
        "save_pred": True,
        "eval_only": FLAGS.eval_only,  # If True, only evaluation will be performed, no training.
        "eval_epoch": FLAGS.eval_epoch,  # If not none, use this epoch for eval, else use the best epoch by val perf.
        "pretrained_model_path": FLAGS.pretrained_model_path,
        "randaugment_n": 2,  # FLAGS.randaugment_n,
        "num_attr_vectors": 10,
        "weight_warm_up_steps": 0,  # No weights warm up in ERM
        "process_outputs_function": "multiclass_logits_to_pred",
        "progress_bar": True,
        "evaluate_all_splits": False,
    }

    logger = Logger(fpath=str(log_dir / "log.txt"))

    # Set random seed
    set_seed(FLAGS.seed)

    rxrx1_dataset = wilds.get_dataset(
        dataset="rxrx1", version=None, root_dir=all_dataset_dir, download=False, split_scheme="official"
    )

    train_grouper = CombinatorialGrouper(dataset=rxrx1_dataset, groupby_fields=config_dict["train_group_by_fields"])

    split_dict_by_names = configure_split_dict_by_names(
        full_dataset=rxrx1_dataset, grouper=train_grouper, config_dict=config_dict
    )

    log_group_data(split_dict_by_names, train_grouper, logger)

    # Initialize algorithm & load pretrained weights if provided
    algorithm = initialize_algorithm(
        train_dataset=split_dict_by_names["train"]["dataset"],
        num_train_steps=calculate_number_of_training_steps(
            config=config_dict, train_loader=split_dict_by_names["train"]["loader"]
        ),
        config=config_dict,
        datasets=split_dict_by_names,
        train_grouper=train_grouper,
        unlabeled_dataset=None,
        loss_weights_by_name={
            "attr_cons_weight": 0.1,
            "self_recon_weight": 0.4,
            "cont_cons_weight": 0.25,
            "attr_similarity_weight": 0.05,
            "entropy_weight": 0.2,
        },
        algorithm_parameters={
            "convert_to_absorbance_in_between": False,
            "num_mixing_per_image": 0,  # 2
            "contrimix_mixing_type": ContriMixMixingType.RANDOM,
        },
        batch_transform=None,
        loss_kwargs={
            "loss_fn": nn.CrossEntropyLoss(reduction="none"),
            "use_cut_mix": False,  # True
            "cut_mix_alpha": FLAGS.cut_mix_alpha,
            "normalize_signals_into_to_backbone": True,  # Important to match with the RxRx1 transform.
            "use_original_image_for_entropy_loss": False,  # True
            "aggregation": ContriMixAggregationType.MAX,  # MEAN
            "training_mode": ContrimixTrainingMode.BACKBONE,
        },
    )

    if not config_dict["eval_only"]:
        print("Training mode!")
        train(
            algorithm=algorithm,
            labeled_split_dict_by_name=split_dict_by_names,
            config_dict=config_dict,
            epoch_offset=0,
            num_training_epochs_per_evaluation=1,
        )

    if config_dict["eval_only"] or FLAGS.run_eval_after_train:
        print("Evaluation mode!")
        eval_model_path = generate_eval_model_path(FLAGS.eval_epoch, FLAGS.model_prefix, FLAGS.seed)
        best_epoch, _ = load(algorithm, eval_model_path, device=config_dict["device"])
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

    logger.close()
    for split in split_dict_by_names:
        split_dict_by_names[split]["eval_logger"].close()
        split_dict_by_names[split]["algo_logger"].close()


if __name__ == "__main__":
    main()
