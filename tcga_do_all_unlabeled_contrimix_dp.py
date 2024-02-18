"""A scripts to train a Contrimix model for TCGA dataset using Data Parallel (DP)."""
import logging
import sys
from pathlib import Path
from typing import Any
from typing import Dict

import torchvision.transforms as transforms

from script_utils import calculate_batch_size

package_path = "/jupyter-users-home/tan-2enguyen/intraminibatch_permutation_drit/"
if package_path not in sys.path:
    sys.path.append(package_path)

from script_utils import configure_parser, dataset_and_log_location, generate_eval_model_path
import torch.cuda
from ip_drit.algorithms.initializer import initialize_algorithm
from ip_drit.algorithms.single_model_algorithm import ModelAlgorithm
from ip_drit.datasets.tcga import TCGADataset
from ip_drit.models.wild_model_initializer import WildModel
from wilds.common.grouper import CombinatorialGrouper
from ip_drit.common.data_loaders import LoaderType
from ip_drit.patch_transform import TransformationType
from script_utils import configure_split_dict_by_names
from train_utils import train
from ip_drit.algorithms import calculate_number_of_training_steps
from ip_drit.patch_transform import RandomRotation
from ip_drit.patch_transform import GaussianNoiseAdder
from ip_drit.patch_transform import PostContrimixTransformPipeline
from ip_drit.loss import ContriMixAggregationType
from ip_drit.algorithms import ContrimixTrainingMode

logging.getLogger().setLevel(logging.INFO)


def main():
    """Training script of ContriMix encoders on the TCGA dataset."""
    print("Training Contrimix on TCGA dataset.")
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

    tcga_dataset = TCGADataset(dataset_dir=all_dataset_dir / "contrimix_demo/", use_full_size=False)
    config_dict: Dict[str, Any] = {
        "algorithm": ModelAlgorithm.CONTRIMIX,
        "model": WildModel.DENSENET121,
        "transform": TransformationType.WEAK_NORMALIZE_TO_0_1,
        "target_resolution": None,  # Keep the original dataset resolution
        "scheduler_metric_split": "val",
        "train_group_by_fields": ["slide_id"],
        "loss_function": "cross_entropy",
        "algo_log_metric": "accuracy",
        "log_dir": str(log_dir),
        "gradient_accumulation_steps": 1,
        "n_epochs": FLAGS.n_epochs,
        "log_every_n_batches": 20,
        "model_kwargs": {"pretrained": False},  # Train from scratch.
        "run_on_cluster": FLAGS.run_on_cluster,
        "train_loader": LoaderType.GROUP,
        "reset_random_generator_after_every_epoch": FLAGS.reset_random_generator_after_every_epoch,
        "batch_size": calculate_batch_size(
            dataset_name=tcga_dataset.dataset_name,
            algorithm=ModelAlgorithm.CONTRIMIX,
            run_on_cluster=FLAGS.run_on_cluster,
        ),
        "uniform_over_groups": FLAGS.sample_uniform_over_groups,  #
        "distinct_groups": True,  # If True, enforce groups sampled per batch are distinct.
        "n_groups_per_batch": 4,
        "scheduler": "linear_schedule_with_warmup",
        "scheduler_kwargs": {"num_warmup_steps": 3},
        "scheduler_metric_name": "scheduler_metric_name",
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "optimizer_kwargs": {"SGD": {"momentum": 0.9}, "Adam": {}, "AdamW": {}},
        "max_grad_norm": 0.5,
        "use_ddp_over_dp": False,
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "use_unlabeled_y": False,  # If true, unlabeled loaders will also the true labels for the unlabeled data.
        "verbose": True,
        "report_batch_metric": True,
        "metric": "acc_avg",
        "val_metric_decreasing": False,
        # Saving parameters
        "save_step": 1,
        "seed": FLAGS.seed,
        "save_last": False,
        "save_best": True,
        "save_pred": True,
        "eval_only": FLAGS.eval_only,  # If True, only evaluation will be performed, no training.
        "eval_epoch": FLAGS.eval_epoch,  # If not none, use this epoch for eval, else use the best epoch by val perf.
        "pretrained_model_path": FLAGS.pretrained_model_path,
        "randaugment_n": 2,
        "num_attr_vectors": 16,
    }

    train_grouper = CombinatorialGrouper(dataset=tcga_dataset, groupby_fields=config_dict["train_group_by_fields"])

    unlabeled_split_dict_by_names = configure_split_dict_by_names(
        full_dataset=tcga_dataset, grouper=train_grouper, config_dict=config_dict
    )

    algorithm = initialize_algorithm(
        train_dataset=unlabeled_split_dict_by_names["train"]["dataset"],
        config=config_dict,
        train_grouper=train_grouper,
        num_train_steps=calculate_number_of_training_steps(
            config=config_dict, train_loader=unlabeled_split_dict_by_names["train"]["loader"]
        ),
        convert_to_absorbance_in_between=True,
        loss_weights_by_name={
            "attr_cons_weight": 0.1,
            "self_recon_weight": 0.6,
            "cont_cons_weight": 0.2,
            "entropy_weight": 0.0,
            "cont_corr_weight": 0.05,
            "attr_similarity_weight": 0.05,
        },
        batch_transform=PostContrimixTransformPipeline(
            transforms=[
                RandomRotation(),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                GaussianNoiseAdder(noise_std=0.02),
            ]
        ),
        loss_kwargs={
            "loss_fn": None,
            "normalize_signals_into_to_backbone": False,
            "aggregation": ContriMixAggregationType.MEAN,
            "training_mode": ContrimixTrainingMode.ENCODERS,
        },
        algorithm_parameters={"num_mixing_per_image": 3},
    )

    if not config_dict["eval_only"]:
        print("Training mode!")
        train(
            algorithm=algorithm,
            labeled_split_dict_by_name=None,
            unlabeled_split_dict_by_name=unlabeled_split_dict_by_names,
            config_dict=config_dict,
            epoch_offset=0,
        )


if __name__ == "__main__":
    main()
