"""A module that defines a the RxRx1 labeled dataset."""
import ctypes
import logging
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import PIL
import torch

from ._dataset import AbstractLabelledPublicDataset

3
from ip_drit.common.grouper import AbstractGrouper
from ip_drit.common.grouper import CombinatorialGrouper
from ip_drit.common.metrics import Accuracy
from ip_drit.datasets import SplitSchemeType


class RxRx1Dataset(AbstractLabelledPublicDataset):
    """A class that defines the labeled RxRx1 dataset.

    This is a modified version of the original RxRx1 dataset.

    Input (x):
        3-channel fluorescent microscopy images of cells

    Label (y):
        y is one of 1,139 classes:
        - 0 to 1107: treatment siRNAs
        - 1108 to 1137: positive control siRNAs
        - 1138: negative control siRNA

    Metadata:
        Each image is annotated with its experiment, plate, well, and site, as
        well as with the id of the siRNA the cells were perturbed with.

    Website:
        https://www.rxrx.ai/rxrx1
        https://www.kaggle.com/c/recursion-cellular-image-classification

    Original publication:
        @inproceedings{taylor2019rxrx1,
            author = {Taylor, J. and Earnshaw, B. and Mabey, B. and Victors, M. and  Yosinski, J.},
            title = {RxRx1: An Image Set for Cellular Morphological Variation Across Many Experimental Batches.},
            year = {2019},
            booktitle = {International Conference on Learning Representations (ICLR)},
            booksubtitle = {AI for Social Good Workshop},
            url = {https://aiforsocialgood.github.io/iclr2019/accepted/track1/pdfs/30_aisg_iclr2019.pdf},
        }

    License:
        This work is licensed under a Creative Commons
        Attribution-NonCommercial-ShareAlike 4.0 International License. To view
        a copy of this license, visit
        http://creativecommons.org/licenses/by-nc-sa/4.0/.

    Args:
        dataset_dir: The location of the dataset.
        split_scheme: The splitting scheme. Supported `split_scheme`: SplitSchemeType.OFFICIAL and
            SplitSchemeType.MIX_TO_TEST.
        use_full_size (optional): If true, will use the full dataset. Otherwise, limit the dataset size to
            40000, which is faster for code development. Defaults to True.
        downsampling_factor (optional): The downsample factor used for downsampling the input image. Defaults to True/
        cache_inputs (optional): If true, the input data will be cached. Defaults to True.
        return_one_hot (optional): If True, return the label as a 1 hot vector. Defaults to False.
    """

    _dataset_name: Optional[str] = "rxrx1"
    _DOWNLOAD_URL_BY_VERSION: Dict[str, str] = {
        "1.0": "https://worksheets.codalab.org/rest/bundles/0x6b7a05a3056a434498f0bb1252eb8440/contents/blob/"
    }
    _OOD_VAL_CENTER = 1
    _TEST_CENTER = 2
    _SPLIT_INDEX_BY_SPLIT_STRING: Dict[str, int] = {"train": 0, "val": 1, "test": 2, "id_test": 3}
    _SPLIT_NAME_BY_SPLIT_STRING: Dict[str, str] = {
        "train": "Train",
        "val": "Validation (OOD)",
        "test": "Test (OOD)",
        "id_test": "Test (ID)",
    }
    _SMALL_DATASET_LIMIT = 40000

    def __init__(
        self,
        dataset_dir: Path,
        split_scheme: SplitSchemeType = SplitSchemeType.MIX_TO_TEST,
        use_full_size: bool = True,
        downsampling_factor: int = 1,
        cache_inputs: bool = True,
        return_one_hot: bool = False,
    ) -> None:
        logging.info("Initializing the RxRx1 data.")
        self._version = "1.0"
        super().__init__(dataset_dir=dataset_dir, return_one_hot=return_one_hot)
        self._downsampling_factor: int = downsampling_factor
        self._original_resolution = (int(256 / downsampling_factor), int(256 / downsampling_factor))
        self._cache_inputs = cache_inputs

        # Read in metadata
        self._metadata_df: pd.DataFrame = pd.read_csv(os.path.join(self._data_dir, "metadata.csv"), index_col=0)

        # TODO: redownload the data and kill this removal. This file was broken by Tan when downsampling images.
        self._metadata_df.drop(self._metadata_df.index[64099], inplace=True)

        if not use_full_size:
            raise ValueError("Limited dataset from RxRx1 is not supported!")

        if split_scheme == SplitSchemeType.OFFICIAL:
            self._split_array: np.ndarray = self._metadata_df.dataset.apply(
                self._SPLIT_INDEX_BY_SPLIT_STRING.get
            ).values

            # Assign Site 2 of the training to in-domain test set
            mask = ((self._metadata_df.site == 2) & (self._metadata_df.dataset == "train")).values
            self._split_array[mask] == self._SPLIT_INDEX_BY_SPLIT_STRING["id_test"]
        elif split_scheme == SplitSchemeType.MIX_TO_TEST:
            self._update_slide_field_for_mix_to_test_split_scheme()
        else:
            raise ValueError(f"split_scheme of {split_scheme} is not supported!")

        self._y_array = torch.LongTensor(self._metadata_df["sirna_id"].values)
        self._n_classes = max(self._metadata_df["sirna_id"]) + 1
        self._y_size = 1
        if len(np.unique(self._metadata_df["sirna_id"])) != self._n_classes:
            raise ValueError("The number of unique siRNA classes does not match expectation!")

        logging.info("Generarting a list of file names for all files.")
        self._input_array: np.ndarray = self._metadata_df.apply(self._file_path_from_row, axis=1).values

        indexed_metadata = self._compute_indexed_metadata()

        self._metadata_array: torch.Tensor = torch.stack(
            (
                torch.LongTensor(indexed_metadata["cell_type"]),
                torch.LongTensor(indexed_metadata["experiment"]),
                torch.LongTensor(self._metadata_df["plate"].values),
                torch.LongTensor(indexed_metadata["well"]),
                torch.LongTensor(self._metadata_df["site"].values),
                torch.LongTensor(self._y_array),
            ),
            dim=1,
        )

        self._metadata_fields: List[str] = ["cell_type", "experiment", "plate", "well", "site", "y"]

        # The evaluation grouper operates ovfer all the slides.
        self._eval_grouper: AbstractGrouper = CombinatorialGrouper(dataset=self, groupby_fields=["cell_type"])
        logging.info(f"Evaluation grouper created for the RxRx1 dataset with {self._eval_grouper.n_groups} groups.")

        if self._cache_inputs:
            self._initialize_cache_array_for_all_data(num_samples=self._metadata_array.shape[0])

    def _update_slide_field_for_mix_to_test_split_scheme(self):
        # Training:   33 experiments total, 1 site per experiment (site 1)
        #             = 19 experiments from the orig training set (site 1)
        #             + 14 experiments from the orig test set (site 1)
        # Validation: same as official split
        # Test:       14 experiments from the orig test set,
        #             1 site per experiment (site 2)
        logging.debug("Update slide fields to MIX_TO_TEST splitting scheme!")
        self._SPLIT_INDEX_BY_SPLIT_STRING: Dict[str, int] = {"train": 0, "val": 1, "test": 2}
        self._SPLIT_NAME_BY_SPLIT_STRING: Dict[str, str] = {"train": "Train", "val": "Validation", "test": "Test"}

        self._split_array: np.ndarray = self._metadata_df.dataset.apply(self._SPLIT_INDEX_BY_SPLIT_STRING.get).values

        # Use half of the training set (site 1) and discard site 2
        self._split_array[((self._metadata_df.dataset == "train") & (self._metadata_df.site == 2)).values] = -1

        # Take all site 1 in the test set and move it to train.
        self._split_array[
            ((self._metadata_df.dataset == "test") & (self._metadata_df.site == 1)).values
        ] = self._SPLIT_INDEX_BY_SPLIT_STRING["train"]

        # Performs the mixing i.e. for each cell type in the test experiments, count how many experiments.
        # The experiments are named with HEGG2-09, HUVEC-22 etc... where the part before the "-" is the name of the
        # cell types. We eliminate the SAME number of training experiments that we have in the test.
        test_exp_count_by_cell_type = defaultdict(int)
        for test_exp in self._metadata_df.loc[self._metadata_df.dataset == "test", "experiment"].unique():
            cell_type = test_exp.split("-")[0]
            test_exp_count_by_cell_type[cell_type] += 1

        # Training experiments are numbered starting from 1 and left-padded with 0's
        training_experiments_to_discard = [
            f"{cell_type}-{num:02}"
            for cell_type, test_exp_count in test_exp_count_by_cell_type.items()
            for num in range(1, test_exp_count + 1)
        ]

        all_train_experiments = self._metadata_df.loc[self._metadata_df.dataset == "train", "experiment"].unique()
        for discard_exp in training_experiments_to_discard:
            if discard_exp not in all_train_experiments:
                raise ValueError(
                    f"discard experiment {discard_exp} is not in train experiments {all_train_experiments}!"
                )
            self._split_array[(self._metadata_df.experiment == discard_exp).values] = -1

    def _initialize_cache_array_for_all_data(self, num_samples: int, num_chans: int = 3) -> None:
        """Initialize an array for caching the data.

        See more details on how to cache all the training data at.
        https://discuss.pytorch.org/t/dataloader-resets-dataset-state/27960/4

        Args:
            num_data_samples: Number of samples for the whole dataset.
            num_chans (optional): The number of training channels. Defaults to 3.
        """
        logging.info("Initializing a shared array for caching all RxRx1 samples!")
        h, w = self._original_resolution
        shared_array = mp.Array(ctypes.c_uint8, num_samples * num_chans * h * w)
        shared_array = np.ctypeslib.as_array(shared_array.get_obj())
        self._shared_array = shared_array.reshape(num_samples, h, w, num_chans)
        for idx in range(num_samples):
            self._shared_array[idx] = np.array(self._get_pil_image_from_input(idx))
            if idx % 10000 == 0:
                logging.info(f"Cached {idx} samples.")
        logging.info(f"Done with caching.")

    def _file_path_from_row(self, row: pd.DataFrame) -> str:
        """Returns the path to the image file from the row dataframe."""
        filepath = os.path.join("images", row.experiment, f"Plate{row.plate}", f"{row.well}_s{row.site}.png")
        return filepath

    def _compute_indexed_metadata(self) -> Dict[str, List[int]]:
        """Computes a dictionary of metadata indexing list, keyed by the field names."""
        indexed_metadata: Dict[str, List[int]] = {}
        self._metadata_map: Dict[str, List] = {}
        for key in ["cell_type", "experiment", "well"]:
            all_values = list(self._metadata_df[key].unique())
            value_to_index_map = {v: idx for idx, v in enumerate(all_values)}
            value_idxs = [value_to_index_map[v] for v in self._metadata_df[key].to_list()]
            self._metadata_map[key] = all_values
            indexed_metadata[key] = value_idxs
        return indexed_metadata

    def _get_pil_image_from_input(self, idx: int) -> PIL.Image:
        """Returns a PIL image from the image with index idx.

         Args:
            idx: Index of a data point

        Returns:
            A numpy array of the input
        """
        image_name = self._generate_image_name(idx)
        if image_name.exists():
            return PIL.Image.open(image_name)
        else:
            full_res_img_path = self._data_dir / self._input_array[idx]
            resized_im = PIL.Image.open(full_res_img_path).resize(
                tuple(int(x / self._downsampling_factor) for x in self._original_resolution), resample=PIL.Image.BICUBIC
            )
            resized_im.save(image_name)
            logging.info(f"Save image named {image_name}")
            return resized_im

    def _get_input(self, idx: int) -> PIL.Image:
        """Returns an input image in the order of C x H x W.

        Args:
            idx: Index of a data point

        Returns:
            An input image of the idx-th data point
        """
        if self._cache_inputs:
            return PIL.Image.fromarray(self._shared_array[idx])
        else:
            return self._get_pil_image_from_input(idx)

    def _generate_image_name(self, idx: int) -> Path:
        """Generates the name of the downsampled image."""
        if self._downsampling_factor == 1:
            return self._data_dir / self._input_array[idx]
        else:
            prefix, ext = self._input_array[idx].split(".")
            file_name = prefix + "_" + str(self._downsampling_factor) + "x_ds." + ext
            return self._data_dir / file_name

    def get_index_for_relevant_images(self, idx: int) -> List[int]:
        """Identifies a set of image indexes that are relevant to the current image indexes.

        Args:
            idx: The index of the current sample.
        """
        pass

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
        return self._standard_group_eval(
            metric=Accuracy(prediction_fn=prediction_fn),
            grouper=self._eval_grouper,
            y_true=y_true,
            y_pred=y_pred,
            metadata=metadata,
            aggregate=True,
        )
