"""A module that defines the TCGA dataset."""
import os
import shutil
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from ._unlabelled_dataset import AbstractUnlabelPublicDataset


class TCGADataset(AbstractUnlabelPublicDataset):
    """Unlabeled TCGA dataset.

    This dataset contains patches from multiple organs from TCGA data.

    Args:
        dataset_dir: The location of the dataset.
        use_full_size (optional): If true, will use the full dataset. Otherwise, use a small dataset,
            which is faster for code development.
        num_samples_per_slide (optional): The number of samples per slides. Defaults to 200.
    """

    _dataset_name: Optional[str] = "tcga_unlabeled"

    _TEST_CENTER = 2
    _SPLIT_INDEX_BY_SPLIT_STRING: Dict[str, int] = {"train": 0, "test": 2}
    _SPLIT_NAME_BY_SPLIT_STRING: Dict[str, str] = {}
    _NUM_ORGAN_SMALL_DATASET = 2
    _ORGAN_TO_INDEX_MAPPING = {
        "BRCA": 0,
        "CRC": 1,
        "RCC": 2,
        "Gastric": 3,
        "Melanoma": 4,
        "NSCLC": 5,
        "Pancreatic": 6,
        "Prostate": 7,
        "HCC": 8,
        "DLBCL": 9,
        "HNSCC": 10,
        "Ovarian": 11,
        "Bladder Cancer": 12,
        "CRC (GSK)": 13,
    }

    def __init__(self, dataset_dir: Path, use_full_size: bool = True, num_samples_per_slides: int = 200) -> None:
        self._version = "1.0"
        super().__init__(dataset_dir=dataset_dir)
        self._original_resolution = (572, 572)
        self._split_names = {"train": "Train", "test": "Test"}
        self._use_full_size = use_full_size
        self._num_samples_per_slides = num_samples_per_slides

        training_slide_by_organs = self._generate_file_names(split="training")
        self._file_names = self._generate_file_list(training_slide_by_organs, split="training")
        self._num_training_slides = len(self._file_names)

        testing_slide_by_organs = self._generate_file_names(split="testing")
        testing_file_names = self._generate_file_list(testing_slide_by_organs, split="testing")
        self._num_testing_slides = len(testing_file_names)
        self._file_names.extend(testing_file_names)

        self._split_array = pd.Series(np.zeros(len(self._file_names)))
        self._split_array[: self._num_training_slides] = self._SPLIT_INDEX_BY_SPLIT_STRING["train"]
        self._split_array[self._num_training_slides :] = self._SPLIT_INDEX_BY_SPLIT_STRING["test"]

        self._metadata_array = self._generate_metadata_array(
            training_slide_by_organs=training_slide_by_organs,
            testing_slide_by_organs=testing_slide_by_organs,
            num_samples=len(self._file_names),
        )
        self._metadata_fields: List[str] = ["organ", "slide_id"]
        print(f"Training {self._num_training_slides} slides, testing {self._num_testing_slides} slides")

    def check_input_and_clean_up(self) -> None:
        from tqdm import tqdm

        for idx in tqdm(range(len(self._file_names))):
            if not os.path.exists(self._file_names[idx]) and self._file_names[idx].parents[0].exists():
                try:
                    shutil.rmtree(str(self._file_names[idx].parents[0]))
                except Exception as e:
                    print(f"Image {self._file_names[idx]} is does not exit, error = {e}. Removed its folder!")
                continue

            try:
                im = self._get_input(idx)
                if im.size != (572, 572):
                    shutil.rmtree(str(self._file_names[idx].parents[0]))
                    print(f"Image {self._file_names[idx]} is faulty with size of {im.size}. Removed!")
            except Exception as e:
                if self._file_names[idx].parents[0].exists():
                    shutil.rmtree(str(self._file_names[idx].parents[0]))
                    print(f"Removed folder {str(self._file_names[idx].parents[0])}, error {e}!")

    def _get_input(self, idx: int) -> Image:
        """Returns an input image in the order of C x H x W."""
        im = Image.fromarray(cv2.imread(str(self._file_names[idx])))
        if im.size != (572, 572):
            raise ValueError(f"Image {self._file_names[idx]} has a shape of {im.size}!")
        return im

    def _generate_file_names(self, split: str) -> Dict[str, List[str]]:
        organs = os.listdir(self._data_dir / split)
        slide_ids_by_organs = defaultdict(list)
        for organ_idx, organ in enumerate(organs):
            if not self._use_full_size and organ not in {"Pancreatic"}:  # organ_idx >= self._NUM_ORGAN_SMALL_DATASET:
                continue

            print(f"Finding samples for {organ}...")
            slide_ids_by_organs[organ] = os.listdir(self._data_dir / split / organ)
        return slide_ids_by_organs

    def _generate_file_list(self, slide_by_organs: Dict[str, List[str]], split: str) -> List[str]:
        return [
            self._data_dir / split / organ / slide_id / f"{i}.png"
            for organ, slide_ids in slide_by_organs.items()
            for slide_id in slide_ids
            for i in range(self._num_samples_per_slides)
        ]

    def _generate_metadata_array(
        self,
        training_slide_by_organs: Dict[str, List[str]],
        testing_slide_by_organs: Dict[str, List[str]],
        num_samples: int,
    ) -> torch.Tensor:
        organ_arr = np.zeros((num_samples,))
        slide_arr = np.zeros((num_samples,))
        sample_idx = 0
        for slide_by_organ_dict in [training_slide_by_organs, testing_slide_by_organs]:
            for organ, slide_ids in slide_by_organ_dict.items():
                for slide_id in slide_ids:
                    organ_arr[sample_idx : sample_idx + self._num_samples_per_slides] = self._ORGAN_TO_INDEX_MAPPING[
                        organ
                    ]
                    slide_arr[sample_idx : sample_idx + self._num_samples_per_slides] = int(slide_id)
                    sample_idx += self._num_samples_per_slides

        return torch.stack((torch.LongTensor(organ_arr), torch.LongTensor(slide_arr)), dim=1)
