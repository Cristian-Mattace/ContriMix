"""A module that defines the unlabelled Camelyon-17 dataset."""
import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ._unlabelled_dataset import AbstractUnlabelPublicDataset
from ._utils import SplitSchemeType
from .camelyon17 import limit_metadata_df
from ip_drit._constants import _UNLABELED_CLASS_INDEX


class CamelyonUnlabeledDataset(AbstractUnlabelPublicDataset):
    """Unlabeled Camelyon17-WILDS dataset.

    This dataset contains patches from all of the slides in the original CAMELYON17 training data, except for the slides
    that were labeled with lesion annotations and therefore used in the labeled Camelyon17Dataset.

    Args:
        dataset_dir: The location of the dataset.
        split_scheme: The splitting scheme.
        use_full_size (optional): If true, will use the full dataset. Otherwise, limit the dataset size to
            40000, which is faster for code development. Defaults to True.
    """

    _dataset_name: Optional[str] = "camelyon17_unlabeled"
    _DOWNLOAD_URL_BY_VERSION: Dict[str, str] = {
        "1.0": "https://worksheets.codalab.org/rest/bundles/0xa78be8a88a00487a92006936514967d2/contents/blob/"
    }

    _OOD_VAL_CENTER = 1
    _TEST_CENTER = 2
    _SPLIT_INDEX_BY_SPLIT_STRING: Dict[str, int] = {"train_unlabeled": 10, "test_unlabeled": 12, "val_unlabeled": 11}
    _NUM_TEST_SAMPLES = 600030
    _NUM_VAL_SAMPLES = 600030
    _NUM_TRAIN_SAMPLES = 1799247
    _SMALL_DATASET_LIMIT = 200000

    def __init__(
        self, dataset_dir: Path, split_scheme: SplitSchemeType = SplitSchemeType.OFFICIAL, use_full_size: bool = True
    ) -> None:
        self._version = "1.0"
        super().__init__(dataset_dir=dataset_dir)
        # self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (96, 96)

        # Read in metadata
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, "metadata.csv"), index_col=0, dtype={"patient": "str"}
        )

        self._use_full_size = use_full_size
        if not self._use_full_size:
            self._metadata_df = limit_metadata_df(self._metadata_df, dataset_limit=self._SMALL_DATASET_LIMIT)

        # Get filenames
        self._file_names = [
            f"patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png"
            for patient, node, x, y in self._metadata_df.loc[:, ["patient", "node", "x_coord", "y_coord"]].itertuples(
                index=False, name=None
            )
        ]

        self._update_split_field_index_of_metadata()
        self._split_array = self._metadata_df["split"].values
        self._metadata_array = torch.stack(
            (
                torch.LongTensor(self._metadata_df["center"].values),
                torch.LongTensor(self._metadata_df["slide"].values),
                # In metadata.csv, all y's value are - 1's,
                torch.LongTensor(self._metadata_df["tumor"].replace(-1, _UNLABELED_CLASS_INDEX).values),
            ),
            dim=1,
        )

        self._metadata_fields: List[str] = ["hospital", "slide", "y"]

    def _update_split_field_index_of_metadata(self) -> None:
        """Replaces the index of the split with a pre-defined value."""
        centers = self._metadata_df["center"]
        test_center_mask = centers == self._TEST_CENTER
        self._metadata_df.loc[test_center_mask, "split"] = self._SPLIT_INDEX_BY_SPLIT_STRING["test_unlabeled"]

        val_center_mask = centers == self._OOD_VAL_CENTER
        self._metadata_df.loc[val_center_mask, "split"] = self._SPLIT_INDEX_BY_SPLIT_STRING["val_unlabeled"]

        train_center_mask = ~centers.isin([self._TEST_CENTER, self._OOD_VAL_CENTER])
        self._metadata_df.loc[train_center_mask, "split"] = self._SPLIT_INDEX_BY_SPLIT_STRING["train_unlabeled"]

        if self._use_full_size:
            if self._metadata_df.loc[test_center_mask].shape[0] != self._NUM_TEST_SAMPLES:
                raise ValueError(f"The number of test sample is not equal to expectation ({self._NUM_TEST_SAMPLES}).")
            if self._metadata_df.loc[val_center_mask].shape[0] != self._NUM_VAL_SAMPLES:
                raise ValueError(f"The number of val sample is not equal to expectation ({self._NUM_VAL_SAMPLES}).")
            if self._metadata_df.loc[train_center_mask].shape[0] != self._NUM_TRAIN_SAMPLES:
                raise ValueError(f"The number of train sample is not equal to expectation ({self._NUM_TRAIN_SAMPLES}).")

    def _get_input(self, idx: int) -> Image:
        """Returns an input image in the order of C x H x W."""
        im_file_name = os.path.join(self._data_dir, self._file_names[idx])
        return Image.open(im_file_name).convert("RGB")
