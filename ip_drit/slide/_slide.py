"""A module that defines the interface for slide images."""
from pathlib import Path
from typing import Optional
from typing import Tuple

import numpy as np
import openslide


class Slide:
    """A class that defines the slide information for each slide.

    Args:
        file_name: The full path to the file that we are reading from.
        domain_index (optional): The index of the domain. Defaults to None.
    """

    def __init__(self, file_name: Path, domain_index: Optional[int] = None) -> None:
        self._domain_index: int = domain_index
        self._slide = openslide.open_slide(filename=str(file_name))

    @property
    def slide_shape_2d(self) -> Tuple[int, int]:
        """Returns the shape of the slide in (#rows, #cols)."""
        return self._slide.level_dimensions[0][::-1]

    def __getitem__(self, row_col_slices: Tuple[slice, slice]) -> np.ndarray:
        """Returns a numpy array for a patch given the row and the column slices."""
        row_slice, col_slice = row_col_slices
        num_rows, num_cols = self.slide_shape_2d

        row_slice = self._fix_none_in_slice(row_slice, num_rows)
        col_slice = self._fix_none_in_slice(col_slice, num_cols)

        y_loc_pixels = int(row_slice.start)
        x_loc_pixels = int(col_slice.start)
        patch_height_pixels = row_slice.stop - row_slice.start
        patch_width_pixels = col_slice.stop - col_slice.start
        return np.array(
            self._slide.read_region(
                location=(x_loc_pixels, y_loc_pixels), level=0, size=(patch_width_pixels, patch_height_pixels)
            )
        )[:: row_slice.step, :: col_slice.step, :3]

    @staticmethod
    def _fix_none_in_slice(input_slice: slice, default_slice_end: int) -> slice:
        slice_start = input_slice.start
        if slice_start is None:
            slice_start = 0

        slice_end = input_slice.stop
        if slice_end is None:
            slice_end = default_slice_end

        slice_step = input_slice.step
        if slice_step is None:
            slice_step = 1
        return slice(slice_start, slice_end, slice_step)

    @property
    def domain_index(self) -> int:
        if self._domain_index is None:
            raise ValueError("Domain index was not specified when initialize the Slide object.")
        return self._domain_index
