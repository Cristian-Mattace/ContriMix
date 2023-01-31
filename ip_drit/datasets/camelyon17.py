"""A module that defines a the Camelyon 17 dataset."""
import os
from enum import auto
from enum import Enum
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ._dataset import AbstractPublicDataset
from ip_drit.common.grouper import CombinatorialGrouper
from ip_drit.common.metrics import Accuracy


class SplitSchemeType(Enum):
    """Split scheme type."""

    OFFICIAL = auto()
    MIX_TO_TEST = auto()


class CamelyonDataset(AbstractPublicDataset):
    """A class that defines the Camelyon dataset."""

    _dataset_name: Optional[str] = "Camelyon17_WILDS"
    _DOWNLOAD_URL_BY_VERSION: Dict[str, str] = {
        "1.0": "https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/"
    }
    _OOD_VAL_CENTER = 1
    _TEST_CENTER = 2
    _SPLIT_INDEX_BY_SPLIT_STRING: Dict[str, int] = {"train": 0, "id_val": 1, "test": 2, "ood_val": 3}

    _SPLIT_NAME_BY_SPLIT_STRING: Dict[str, str] = {}

    def __init__(self, dataset_dir: Path, split_scheme: SplitSchemeType = SplitSchemeType.OFFICIAL) -> None:
        self._version = "1.0"
        super().__init__(dataset_dir=dataset_dir)
        self._original_resolution = (96, 96)

        # Read in metadata
        self._metadata_df: pd.DataFrame = pd.read_csv(
            os.path.join(self._data_dir, "metadata.csv"), index_col=0, dtype={"patient": "str"}
        )

        # Hack to reduce the number of samples
        num_of_samples = 10000
        data_centers_values = np.unique(self._metadata_df["center"].values)
        num_centers = len(data_centers_values)
        num_value_per_center = num_of_samples // num_centers
        keep_idxes = []
        for center_idx in data_centers_values:
            keep_idxes.extend(np.where(self._metadata_df["center"].values == center_idx)[0][:num_value_per_center])
        self._metadata_df = self._metadata_df.iloc[keep_idxes]
        # End of hacking.

        self._y_array = torch.LongTensor(self._metadata_df["tumor"].values)

        self._n_classes = 2

        self._file_names: List[str] = [
            f"patches/patient_{p}_node_{n}/patch_patient_{p}_node_{n}_x_{x}_y_{y}.png"
            for p, n, x, y in self._metadata_df.loc[:, ["patient", "node", "x_coord", "y_coord"]].itertuples(
                index=False, name=None
            )
        ]

        self._update_split_field_of_metadata()
        if split_scheme == SplitSchemeType.MIX_TO_TEST:
            self._update_slide_field_for_mix_to_test_split_scheme()

        self._split_array: np.ndarray = self._metadata_df["split"].values
        self._metadata_array: torch.Tensor = torch.stack(
            (
                torch.LongTensor(self._metadata_df["center"].values),
                torch.LongTensor(self._metadata_df["slide"].values),
                torch.LongTensor(self._metadata_df["tumor"].values),
            ),
            dim=1,
        )
        self._metadata_fields: List[str] = ["hospital", "slide", "y"]
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=["slide"])

    def _update_split_field_of_metadata(self) -> None:
        centers = self._metadata_df["center"]
        test_center_mask = centers == self._TEST_CENTER
        self._metadata_df.loc[test_center_mask, "split"] = self._SPLIT_INDEX_BY_SPLIT_STRING["test"]

        val_center_mask = centers == self._OOD_VAL_CENTER
        self._metadata_df.loc[val_center_mask, "split"] = self._SPLIT_INDEX_BY_SPLIT_STRING["ood_val"]

    def _update_slide_field_for_mix_to_test_split_scheme(self):
        # For the mixed-to-test setting,
        # we move slide 23 (corresponding to patient 042, node 3 in the original dataset)
        # from the test set to the training set
        slide_mask = self._metadata_df["slide"] == 23
        self._metadata_df.loc[slide_mask, "split"] = self._SPLIT_INDEX_BY_SPLIT_STRING["train"]

    def get_input(self, idx: int) -> Image:
        """Returns an input image in the order of C x H x W."""
        im_file_name = os.path.join(self._data_dir, self._file_names[idx])
        return Image.open(im_file_name).convert("RGB")

    def eval(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        metadata: torch.Tensor,
        prediction_fn: Optional[Callable] = None,
    ) -> Tuple[Dict, str]:
        """Computes all evaluation metrics.

        Args:
            y_pred: Predictions from a model. By default, they are predicted labels. But they can also be other model
                outputs such that prediction_fn(y_pred) are predicted labels.
            y_true: Ground-truth labels
            metadata: Metadata.
            prediction_fn: A function that turns y_pred into predicted labels.

        Returns:
            A dictionary of evaluation metrics, keyed by the name of the metrics.
            A string summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)
        return self._standard_group_eval(metric, self._eval_grouper, y_pred, y_true, metadata)
