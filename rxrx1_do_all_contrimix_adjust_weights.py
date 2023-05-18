"""A scripts to obtain the baseline for the RxRx1 dataset.

Reference:
    The related physics of the flourescence and the Lambert Beer's law can be found in
        https://application.wiley-vch.de/books/sample/3527316698_c01.pdf.
"""
import logging
import sys
from typing import Any
from typing import Dict

package_path = "/jupyter-users-home/dinkar-2ejuyal/intraminibatch_permutation_drit/"
if package_path not in sys.path:
    sys.path.append(package_path)

from script_utils import dataset_and_log_location, generate_eval_model_path
from script_utils import configure_parser
import torch.cuda
from ip_drit.algorithms.initializer import initialize_algorithm
from ip_drit.algorithms.single_model_algorithm import ModelAlgorithm
from ip_drit.common.grouper import CombinatorialGrouper
from ip_drit.datasets.rxrx1 import RxRx1Dataset
from ip_drit.logger import Logger
from ip_drit.models.wild_model_initializer import WildModel
from ip_drit.common.data_loaders import LoaderType
from ip_drit.patch_transform import TransformationType
from script_utils import configure_split_dict_by_names
from script_utils import use_data_parallel
from train_utils import train, evaluate_over_splits
from saving_utils import load
from script_utils import num_of_available_devices
from ip_drit.algorithms import calculate_number_of_training_steps
from ip_drit.datasets import SplitSchemeType
from ip_drit.algorithms import ContriMixMixingType
from ip_drit.patch_transform import PostContrimixTransformPipeline, GaussianNoiseAdder, RandomRotation

import torchvision.transforms as transforms

logging.getLogger().setLevel(logging.INFO)

from script_utils import set_visible_gpus


def main():
    """Demo scripts for training, evaluation with the labeled RxRx1 data."""
    logging.info("Running the RxRx1 dataset benchmark with the ContriMix algorithm.")
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

    rxrx1_dataset = RxRx1Dataset(
        dataset_dir=all_dataset_dir / "rxrx1/",
        use_full_size=FLAGS.use_full_dataset,
        split_scheme=SplitSchemeType.OFFICIAL,
        cache_inputs=False,
        downsampling_factor=1,
        return_one_hot=True,
    )

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
        "n_epochs": FLAGS.n_epochs,
        "log_every_n_batches": FLAGS.log_every_n_batches,
        "train_loader": LoaderType.GROUP,
        "reset_random_generator_after_every_epoch": False,
        "batch_size": FLAGS.batch_size_per_gpu * num_of_available_devices(),
        "run_on_cluster": FLAGS.run_on_cluster,
        "uniform_over_groups": FLAGS.sample_uniform_over_groups,  #
        "distinct_groups": True,  # If True, enforce groups sampled per batch are distinct.
        "n_groups_per_batch": FLAGS.num_groups_per_training_batch,
        "scheduler": "linear_schedule_with_warmup",
        "scheduler_kwargs": {"num_warmup_steps": 3},
        "scheduler_metric_name": "scheduler_metric_name",
        "optimizer": "AdamW",
        "lr": 0.5e-3,
        "weight_decay": 1e-3,
        "optimizer_kwargs": {"SGD": {"momentum": 0.9}, "Adam": {}, "AdamW": {}},
        "max_grad_norm": 0.8,
        "use_data_parallel": use_data_parallel(),
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
        "num_attr_vectors": FLAGS.contrimix_num_attr_vectors,
    }

    logger = Logger(fpath=str(log_dir / "log.txt"))

    train_grouper = CombinatorialGrouper(dataset=rxrx1_dataset, groupby_fields=config_dict["train_group_by_fields"])

    split_dict_by_names = configure_split_dict_by_names(
        full_dataset=rxrx1_dataset, grouper=train_grouper, config_dict=config_dict
    )
    algorithm = initialize_algorithm(
        train_dataset=split_dict_by_names["train"]["dataset"],
        config=config_dict,
        train_grouper=train_grouper,
        num_train_steps=calculate_number_of_training_steps(
            config=config_dict, train_loader=split_dict_by_names["train"]["loader"]
        ),
        loss_weights_by_name={
            "attr_cons_weight": 0.001,
            "self_recon_weight": 0.7,
            "cont_cons_weight": 0.099,
            "entropy_weight": 0.2,
        },
        algorithm_parameters={
            "convert_to_absorbance_in_between": False,
            "num_mixing_per_image": FLAGS.num_mixing_per_image,
            "contrimix_mixing_type": ContriMixMixingType.WITHIN_CHUNK,
        },
        batch_transform=PostContrimixTransformPipeline(
            transforms=[
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomRotation(),
                GaussianNoiseAdder(noise_std=0.01),
            ]
        ),
        loss_kwargs={
            "use_original_image_for_entropy_loss": True,
            "use_cut_mix": FLAGS.use_cut_mix,
            "cut_mix_alpha": FLAGS.cut_mix_alpha,
            "normalize_signals_into_to_backbone": True,
            "use_original_image_for_entropy_loss": True,
        },
    )
    #
    if not config_dict["eval_only"]:
        logging.info("Training mode!")
        train(
            algorithm=algorithm,
            labeled_split_dict_by_name=split_dict_by_names,
            general_logger=logger,
            config_dict=config_dict,
            epoch_offset=0,
        )
    if config_dict["eval_only"] or FLAGS.run_eval_after_train:
        logging.info("Evaluation mode!")
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
