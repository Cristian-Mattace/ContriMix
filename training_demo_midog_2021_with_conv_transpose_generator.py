"""Training script for Intrabatch permutation DRIT.

The image generator uses the convolutional tranposed based generator named AbsorbanceImGeneratorWithConvTranspose.
This generator is expected to reduce the gridding artifact from the pixel shuffling while improving the resolution
of the output.
"""
import logging
from pathlib import Path
from typing import Dict
from typing import List

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from constants import INPUT_PATCH_SIZE_PIXELS
from constants import NUM_STAIN_VECTORS
from ip_drit.datamodule import MultiDomainDataModule
from ip_drit.datamodule._dataset import MultiDomainDataset
from ip_drit.loss import ImageConsistencyLossType
from ip_drit.models import GeneratorType
from ip_drit.patch_transform import RGBToTransmittance
from ip_drit.patch_transform import ToTensor
from ip_drit.patch_transform import TransmittanceToAbsorbance
from ip_drit.sampling import generate_sample_lists_by_domain_indices
from ip_drit.sampling import Sample
from ip_drit.training_module import every_n_checkpoint_callback
from ip_drit.training_module import MutliClassTrainingModule
from ip_drit.training_module import top_n_checkpoint_callback
from utils import adjust_num_samples_to_be_a_multiple_of_number_of_gpus_and_batch_size_per_gpu_product
from utils import adjust_samples_by_domain_idx_dict_to_the_target_number_of_sample
from utils import clean_up_all_memory_used_by_cuda
from utils import get_number_of_available_gpus
from utils import loss_string_from_weight_dict
from utils import make_sure_folder_exists
from utils import nearest_even_minibatch_size
from utils import prepare_saving_folders_for_checkpoint
from utils import prepare_training_dataset

logging.basicConfig(level=logging.DEBUG)

MIDOG_FOLDER = Path("/jupyter-users-home/tan-2enguyen/public_data/MIDOG_2021")
data_folder = MIDOG_FOLDER / "images"
sample_folder = MIDOG_FOLDER / "point_samples"
checkpoint_folder = MIDOG_FOLDER / "checkpoints"


make_sure_folder_exists(folders=[data_folder, sample_folder, checkpoint_folder])


def main() -> None:
    """Trains an intrabatch permutation model from MIDOG data."""
    dataset_info = prepare_training_dataset(data_folder=data_folder)
    sample_lists_by_domain_idx = generate_sample_lists_by_domain_indices(
        dataset_info=dataset_info,
        sample_folder=sample_folder,
        patch_size_pixels=INPUT_PATCH_SIZE_PIXELS,
        max_num_samples_per_domain=100000,
    )
    _train_model(sample_lists_by_domain_idx=sample_lists_by_domain_idx)


def _train_model(sample_lists_by_domain_idx: Dict[int, List[Sample]]) -> None:
    """Trains the model."""
    logging.info(f"Model training started...")
    clean_up_all_memory_used_by_cuda()
    NUM_GPUS = get_number_of_available_gpus()
    pretrain_model_path = None
    BATCH_SIZE_PER_GPU = nearest_even_minibatch_size(org_minibatch_size=8)
    loss_weights_by_name = {
        "real_fake_weight": 0.5,
        "recon_weight": 20.0,
        "content_consistency_weight": 10.0,
        "attr_consistency_weight": 1.0,
        "mode_seeking_loss_weight": 5.0,
        "content_channel_covariance_loss_weight": 0.2,
    }

    train_hyperparams = {
        "weight_decay": 0.001,
        "gen_learning_rate": 5e-3,
        "disc_learning_rate": 5e-2,
        "num_epochs": 10000,
        "batch_size": BATCH_SIZE_PER_GPU * NUM_GPUS,
        "num_dataloaders": 20,
        "loss_weights_by_name": loss_weights_by_name,
        "pretrained_model_path": pretrain_model_path,
        "number_gen_optimization_steps_to_update_disc": 1,
        "number_of_steps_to_update_lr": 1,
        "periodically_save_training_results": True,
    }

    num_train_samples = adjust_num_samples_to_be_a_multiple_of_number_of_gpus_and_batch_size_per_gpu_product(
        25000, NUM_GPUS, BATCH_SIZE_PER_GPU, num_domains=3
    )
    num_val_samples = adjust_num_samples_to_be_a_multiple_of_number_of_gpus_and_batch_size_per_gpu_product(
        5000, NUM_GPUS, BATCH_SIZE_PER_GPU, num_domains=1
    )
    logging.info(f"Number of training samples {num_train_samples}, number of validation samples {num_val_samples}")

    experiment_name = (
        f"Stain_separation_{NUM_STAIN_VECTORS}_stain_vectors_{num_train_samples}_"
        + f"bs_{BATCH_SIZE_PER_GPU}_samples_w_{loss_string_from_weight_dict(loss_weights_by_name)}_MIDOG_v1.4"
    )

    logging.info(f" -> experiment_name = {experiment_name}")

    check_val_every_n_epoch = 10
    folders_by_name: Dict[str, Path] = prepare_saving_folders_for_checkpoint(
        save_dir=checkpoint_folder / experiment_name
    )
    callbacks = [
        every_n_checkpoint_callback(save_dir=folders_by_name["every_n_epochs_dir"]),
        top_n_checkpoint_callback(save_dir=folders_by_name["top_n_dir"]),
        EarlyStopping(monitor="val_encoders_generators_total_loss", mode="min", min_delta=0.0001, patience=10),
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=train_hyperparams["num_epochs"],
        enable_progress_bar=True,
        num_sanity_val_steps=1,
        precision=32,
        # logger = [tensorboard_logger],
        logger=[],
        resume_from_checkpoint=False,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=20,
        # profiler="advanced",
        profiler=None,
        accelerator="gpu",
        devices=-1,
        num_nodes=1,
        accumulate_grad_batches=1,
        gradient_clip_val=5,
        strategy="dp",
    )

    val_sample_list_by_domain_index = adjust_samples_by_domain_idx_dict_to_the_target_number_of_sample(
        samples_by_domain_idx={3: sample_lists_by_domain_idx[3]}, target_number_of_samples=num_val_samples
    )

    train_sample_list_by_domain_index = adjust_samples_by_domain_idx_dict_to_the_target_number_of_sample(
        samples_by_domain_idx={d: sample_lists_by_domain_idx[d] for d in [0, 1, 2]},
        target_number_of_samples=num_train_samples,
    )

    data_module = MultiDomainDataModule(
        train_sample_list_by_domain_index=train_sample_list_by_domain_index,
        val_sample_list_by_domain_index=val_sample_list_by_domain_index,
        batch_size=train_hyperparams["batch_size"],
        num_dataloading_workers=train_hyperparams["num_dataloaders"],
        input_patch_size_pixels=INPUT_PATCH_SIZE_PIXELS,
    )

    data_module.setup("fit")

    training_module = MutliClassTrainingModule(
        num_input_channels=3,
        num_stain_vectors=NUM_STAIN_VECTORS,
        num_attribute_enc_out_chans=16,
        train_hyperparams=train_hyperparams,
        test_dataset=MultiDomainDataset(
            sample_list_by_domain_index=train_sample_list_by_domain_index,
            transforms=[RGBToTransmittance(), TransmittanceToAbsorbance(), ToTensor()],
            input_patch_size_pixels=INPUT_PATCH_SIZE_PIXELS,
        )
        if train_hyperparams["periodically_save_training_results"]
        else None,
        generator_type=GeneratorType.WithConvTranspose2d,
        image_consistency_loss_type=ImageConsistencyLossType.L1ImageConsistencyLoss,
    )

    trainer.fit(training_module, data_module, ckpt_path="last")


if __name__ == "__main__":
    main()
