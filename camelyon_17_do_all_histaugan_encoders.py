"""A scripts to run the benchmark for the Camelyon dataset."""
import logging
import sys
from typing import Any
from typing import Dict

import torch.nn as nn

package_path = "/jupyter-users-home/dinkar-2ejuyal/intraminibatch_permutation_drit/"
if package_path not in sys.path:
    sys.path.append(package_path)

from script_utils import calculate_batch_size
from script_utils import set_visible_gpus
from script_utils import configure_parser, dataset_and_log_location, generate_eval_model_path
import torch.cuda
from ip_drit.algorithms.initializer import initialize_algorithm
from ip_drit.algorithms.single_model_algorithm import ModelAlgorithm
from ip_drit.loss import ContriMixAggregationType
from ip_drit.common.grouper import CombinatorialGrouper
from ip_drit.datasets.camelyon17 import CamelyonDataset
from ip_drit.logger import Logger
from ip_drit.models.wild_model_initializer import WildModel
from ip_drit.common.data_loaders import LoaderType
from ip_drit.patch_transform import TransformationType
from ip_drit.algorithms import calculate_number_of_training_steps
from ip_drit.algorithms import ContrimixTrainingMode
from script_utils import configure_split_dict_by_names
from train_utils import train, evaluate_over_splits
from saving_utils import load
import torchvision.transforms as transforms

logging.getLogger().setLevel(logging.INFO)


def main():
    """Demo scripts for training, evaluation with Camelyon."""
    print("Running the Camelyon 17 dataset benchmark.")
    parser = configure_parser()
    FLAGS = parser.parse_args()

    set_visible_gpus(gpu_ids=FLAGS.gpu_ids)

    all_dataset_dir, log_dir = dataset_and_log_location(
        FLAGS.run_on_cluster,
        FLAGS.log_dir_cluster,
        FLAGS.log_dir_local,
        FLAGS.dataset_dir_cluster,
        FLAGS.dataset_dir_local,
    )

    all_dataset_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    camelyon_dataset = CamelyonDataset(
        dataset_dir=all_dataset_dir / "camelyon17/",
        use_full_size=True,
        drop_centers=FLAGS.drop_centers,
        return_one_hot=False,
    )

    config_dict: Dict[str, Any] = {
        "algorithm": ModelAlgorithm.HISTAUGAN,
        "model": WildModel.DENSENET121,
        "transform": TransformationType.HISTAUGAN_ENCODERS,
        "target_resolution": None,  # Keep the original dataset resolution
        "scheduler_metric_split": "val",
        "train_group_by_fields": ["hospital"],
        "loss_function": "cross_entropy",
        "algo_log_metric": "accuracy",
        "log_dir": str(log_dir),
        "gradient_accumulation_steps": 1,
        "n_epochs": 50,
        "log_every_n_batches": 3,
        "model_kwargs": {"pretrained": False},  # Train from scratch.
        "run_on_cluster": FLAGS.run_on_cluster,
        "train_loader": LoaderType.GROUP,
        "reset_random_generator_after_every_epoch": FLAGS.reset_random_generator_after_every_epoch,
        "batch_size": calculate_batch_size(
            dataset_name=camelyon_dataset.dataset_name,
            algorithm=ModelAlgorithm.HISTAUGAN,
            run_on_cluster=FLAGS.run_on_cluster,
            batch_size_per_gpu=FLAGS.batch_size_per_gpu,
        ),
        "uniform_over_groups": True,  # FLAGS.sample_uniform_over_groups,  #
        "distinct_groups": False,  # True,  # If True, enforce groups sampled per batch are distinct.
        "n_groups_per_batch": 3,  # 4
        "weight_warm_up_steps": 5,
        "scheduler": "linear_schedule_with_warmup",
        "scheduler_kwargs": {"num_warmup_steps": 3},
        "scheduler_metric_name": "scheduler_metric_name",
        "optimizer": "Adam",
        "lr": None,  # The lr is hard-coded in the code. Not here!
        "weight_decay": None,
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
        "save_step": 5,
        "seed": FLAGS.seed,
        "save_last": False,
        "save_best": True,
        "save_pred": True,
        "eval_only": FLAGS.eval_only,  # If True, only evaluation will be performed, no training.
        "eval_epoch": FLAGS.eval_epoch,  # If not none, use this epoch for eval, else use the best epoch by val perf.
        "pretrained_model_path": FLAGS.pretrained_model_path,
        "randaugment_n": 2,  # FLAGS.randaugment_n,
        "num_attr_vectors": 5,
        "noise_std": FLAGS.noise_std,
        "process_outputs_function": "multiclass_logits_to_pred",
    }

    logger = Logger(fpath=str(log_dir / "log.txt"))

    grouper = CombinatorialGrouper(dataset=camelyon_dataset, groupby_fields=config_dict["train_group_by_fields"])

    split_dict_by_names = configure_split_dict_by_names(
        full_dataset=camelyon_dataset, grouper=grouper, config_dict=config_dict
    )

    nz = 8
    algorithm = initialize_algorithm(
        train_dataset=split_dict_by_names["train"]["dataset"],
        config=config_dict,
        train_grouper=grouper,
        num_train_steps=calculate_number_of_training_steps(
            config=config_dict, train_loader=split_dict_by_names["train"]["loader"]
        ),
        convert_to_absorbance_in_between=True,
        loss_weights_by_name={},
        loss_kwargs={
            "training_mode": ContrimixTrainingMode.ENCODERS,
            "loss_fn": nn.CrossEntropyLoss(reduction="none"),
            "aggregation": ContriMixAggregationType.MEAN,
            "nz": nz,
            "original_resolution": camelyon_dataset.original_resolution,
        },
        algorithm_parameters={
            "concat": 1,  # concatenate attribute features for translation
            "input_dim": 3,  # Number of input channels.
            "dis_norm": "None",  # normalization layer in discriminator
            "dis_spectral_norm": False,  # use spectral normalization in discriminator
            "num_domains": 3,  # The number of domains for the training dataset.
            "crop_size": 216,  # cropped image size for training
            "nz": nz,
            "d_iter": 3,
            "lambda_cls": 1.0,
            "lambda_cls_G": 5.0,
            "lambda_rec": 10.0,
        },
    )
    if not config_dict["eval_only"]:
        print("Training mode!")
        train(
            algorithm=algorithm, labeled_split_dict_by_name=split_dict_by_names, config_dict=config_dict, epoch_offset=0
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


if __name__ == "__main__":
    main()
