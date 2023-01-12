from typing import Dict
from typing import List

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..datasets._dataset import MultiDomainDataset
from ip_drit.patch_transform import RandomFlip
from ip_drit.patch_transform import RandomRotate
from ip_drit.patch_transform import RGBToTransmittance
from ip_drit.patch_transform import ToTensor
from ip_drit.patch_transform import TransmittanceToAbsorbance
from ip_drit.sampling import Sample


class MultiDomainDataModule(pl.LightningDataModule):
    """A class that defines the data module for DRIT training and validation.

    Args:
        train_sample_list_by_domain_index: A dictionary of training sample lists, keyed by the index of the domain. One
            list for 1 domain.
        val_sample_list_by_domain_index: A dictionary of validation sample lists, keyed by the index of the domain. One
            list for 1 domain.
        input_patch_size_pixels: The size of the input patch for training.
        batch_size (optional): The number of samples per batch. Defaults to 32.
        num_dataloading_workers (optional): The number of worker for data loading. Defaults to 4.
        use_pin_memory (optional): If True, pinned memory will be used. Defaults to True.
    """

    def __init__(
        self,
        train_sample_list_by_domain_index: Dict[int, List[Sample]],
        val_sample_list_by_domain_index: Dict[int, List[Sample]],
        input_patch_size_pixels: int,
        batch_size: int = 32,
        num_dataloading_workers: int = 4,
        use_pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self._train_sample_list_by_domain_index = train_sample_list_by_domain_index
        self._val_sample_list_by_domain_index = val_sample_list_by_domain_index
        self._input_patch_size_pixels = input_patch_size_pixels
        self._batch_size = batch_size
        self._num_dataloading_workers = num_dataloading_workers
        self._use_pin_memory = use_pin_memory

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._train_dataset = MultiDomainDataset(
                sample_list_by_domain_index=self._train_sample_list_by_domain_index,
                transforms=[
                    RandomFlip(),
                    RandomRotate(),
                    RGBToTransmittance(),
                    TransmittanceToAbsorbance(),
                    ToTensor(),
                ],
                input_patch_size_pixels=self._input_patch_size_pixels,
                domain_indices=None,
            )
            self._val_dataset = MultiDomainDataset(
                sample_list_by_domain_index=self._val_sample_list_by_domain_index,
                transforms=[RGBToTransmittance(), TransmittanceToAbsorbance(), ToTensor()],
                input_patch_size_pixels=self._input_patch_size_pixels,
                domain_indices=None,
            )
        else:
            raise ValueError("MultiDomainDataModule datamodule is not defined for non-fit stages!")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_dataloading_workers,
            pin_memory=self._use_pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_dataloading_workers,
            pin_memory=self._use_pin_memory,
        )
