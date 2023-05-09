"""A scripts to run the benchmark for the Camelyon dataset."""
import logging
import sys
from typing import Any
from typing import Dict

package_path = "/jupyter-users-home/dinkar-2ejuyal/intraminibatch_permutation_drit/"
if package_path not in sys.path:
    sys.path.append(package_path)

from script_utils import calculate_batch_size
from script_utils import set_visible_gpus
from script_utils import configure_parser, dataset_and_log_location, generate_eval_model_path
import torch.cuda
from ip_drit.algorithms.initializer import initialize_algorithm
from ip_drit.algorithms.single_model_algorithm import ModelAlgorithm
from ip_drit.common.grouper import CombinatorialGrouper
from ip_drit.datasets.camelyon17 import CamelyonDataset
from ip_drit.logger import Logger
from ip_drit.models.wild_model_initializer import WildModel
from ip_drit.common.data_loaders import LoaderType
from ip_drit.patch_transform import TransformationType
from ip_drit.algorithms import calculate_number_of_training_steps
from ip_drit.patch_transform import RandomRotation
from ip_drit.patch_transform import GaussianNoiseAdder
from ip_drit.patch_transform import PostContrimixTransformPipeline
from script_utils import configure_split_dict_by_names
from script_utils import use_data_parallel
from train_utils import train, evaluate_over_splits
from saving_utils import load
import torchvision.transforms as transforms

logging.getLogger().setLevel(logging.INFO)


def main():
    """Demo scripts for training, evaluation with Camelyon."""
    logging.info("Running the Camelyon 17 dataset benchmark.")
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
        use_full_size=FLAGS.use_full_dataset,
        drop_centers=FLAGS.drop_centers,
        return_one_hot=True,
    )

    config_dict: Dict[str, Any] = {
        "algorithm": ModelAlgorithm.CONTRIMIX,
        "model": WildModel.DENSENET121,
        "transform": TransformationType.WEAK_NORMALIZE_TO_0_1,  # RANDAUGMENT_TO_0_1,WEAK_NORMALIZE_TO_0_1
        "target_resolution": None,  # Keep the original dataset resolution
        "scheduler_metric_split": "val",
        "train_group_by_fields": ["hospital"],
        "loss_function": "cross_entropy",
        "algo_log_metric": "accuracy",
        "log_dir": str(log_dir),
        "gradient_accumulation_steps": 1,
        "n_epochs": FLAGS.n_epochs,
        "log_every_n_batches": FLAGS.log_every_n_batches,
        "run_on_cluster": FLAGS.run_on_cluster,
        "train_loader": LoaderType.GROUP,
        "reset_random_generator_after_every_epoch": FLAGS.reset_random_generator_after_every_epoch,
        "batch_size": calculate_batch_size(
            dataset_name=camelyon_dataset.dataset_name,
            algorithm=ModelAlgorithm.CONTRIMIX,
            run_on_cluster=FLAGS.run_on_cluster,
            batch_size_per_gpu=FLAGS.batch_size_per_gpu,
        ),
        "uniform_over_groups": True,  # FLAGS.sample_uniform_over_groups,  #
        "distinct_groups": False,  # True,  # If True, enforce groups sampled per batch are distinct.
        "n_groups_per_batch": FLAGS.num_groups_per_training_batch,  # 4
        "scheduler": "linear_schedule_with_warmup",
        "scheduler_kwargs": {"num_warmup_steps": 3},
        "scheduler_metric_name": "scheduler_metric_name",
        "optimizer": "AdamW",
        "lr": 1e-3,  # 1e-4,
        "weight_decay": 1e-3,
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
        "randaugment_n": 2,  # FLAGS.randaugment_n,
        "num_attr_vectors": 5,
        "noise_std": FLAGS.noise_std,
    }

    logger = Logger(fpath=str(log_dir / "log.txt"))

    train_grouper = CombinatorialGrouper(dataset=camelyon_dataset, groupby_fields=config_dict["train_group_by_fields"])

    split_dict_by_names = configure_split_dict_by_names(
        full_dataset=camelyon_dataset, grouper=train_grouper, config_dict=config_dict
    )
    algorithm = initialize_algorithm(
        train_dataset=split_dict_by_names["train"]["dataset"],
        config=config_dict,
        train_grouper=train_grouper,
        num_train_steps=calculate_number_of_training_steps(
            config=config_dict, train_loader=split_dict_by_names["train"]["loader"]
        ),
        convert_to_absorbance_in_between=True,
        loss_weights_by_name={
            "attr_cons_weight": 0.1,
            "self_recon_weight": 0.1,
            "cont_cons_weight": 0.3,
            "entropy_weight": 0.5,
        },
        batch_transform=PostContrimixTransformPipeline(
            transforms=[
                RandomRotation(),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                GaussianNoiseAdder(noise_std=config_dict["noise_std"]),  # 0.02),
            ]
        ),
    )
    print("noise_std: ", config_dict["noise_std"])

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
