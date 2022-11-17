"""A module that defines the callback functions for training the model."""
import shutil
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint


def every_n_checkpoint_callback(save_dir: Path, every_n_train_epoch: int = 1) -> ModelCheckpoint:
    """Returns a model checkpoint callback, which saves the model every n epochs.

    Args:
        save_dir: The path to the saving directory.
        every_n_train_epoch (optional): A number that specifies the number of epoch interval that the checkpoint will be
            saved. Defaults to 1.
    """
    # Clean up checkpoints from the previous run if we store to the same folder
    shutil.rmtree(save_dir, ignore_errors=True)
    _make_dir_if_not_exist(save_dir)
    return ModelCheckpoint(
        dirpath=str(save_dir),
        every_n_epochs=every_n_train_epoch,
        filename=(
            "periodic_{epoch}_{val_encoders_generators_total_loss:.3f}_"
            + "{train_encoders_generators_total_loss:.3f}_{train_disc_total_loss:.3f}"
        ),
        save_top_k=-1,
    )


def _make_dir_if_not_exist(dir_name: Path) -> None:
    dir_name.mkdir(parents=True, exist_ok=True)


def top_n_checkpoint_callback(save_dir: Path, num_best_models: int = 5) -> ModelCheckpoint:
    """Returns a model checkpoint callback, which saves the top-n performer.

    Args:
        save_dir: The path to the saving directory.
        num_best_models: The number of best performer to save.
    """
    # Clean up checkpoints from the previous run if we store to the same folder
    shutil.rmtree(save_dir, ignore_errors=True)
    _make_dir_if_not_exist(save_dir)
    return ModelCheckpoint(
        dirpath=str(save_dir),
        monitor="val_encoders_generators_total_loss",
        save_top_k=num_best_models,
        mode="min",
        filename=(
            "top_n_{epoch}_{val_encoders_generators_total_loss:.3f}_"
            + "{train_encoders_generators_total_loss:.3f}_{train_disc_total_loss:.3f}"
        ),
    )
