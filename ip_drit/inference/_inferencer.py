"""A module that performs patch inference."""
import logging
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch

from ip_drit.models import AbsorbanceImGenerator
from ip_drit.models import ContentEncoder


class Inferencer:
    """A class that performs the inferencing based on the trained model.

    Args:
        content_encoder: The trained model of the content encoder.
        gen: The trained model of the generator.
        max_tile_size_pixels (optional): The maximum title of each size that we ran the inference on, excluding the
            margin. The true size that the inferencer will work on will be tile_size_pixels + 2 * tile_margin_pixels.
            Defaults to 512.
        tile_margin_pixels (optional): The size of the outer region surrounding the tile in pixels.
            We don't use the region of the in boundary region to avoid artifacts. Defaults to 0.
        max_polarization_val (optional): All polarization signal larger than this value will be mapped to 255 in the
            output. Default to 0.8.
    """

    def __init__(
        self,
        content_encoder: ContentEncoder,
        gen: AbsorbanceImGenerator,
        max_tile_size_pixels: int = 512,
        tile_margin_pixels: int = 0,
        max_polarization_val: float = 0.8,
    ) -> None:
        self._content_encoder: ContentEncoder = content_encoder.cuda()
        self._gen: AbsorbanceImGenerator = gen.cuda()
        self._max_tile_size_pixels: int = max_tile_size_pixels
        self._tile_margin_pixels: int = tile_margin_pixels
        self._max_polarization_val: float = max_polarization_val
        self._tile_size_pixels: int = self._max_tile_size_pixels
        self._inner_tile_size_pixels: int = self._tile_size_pixels - 2 * tile_margin_pixels

    def infer_one_image(
        self, image: torch.Tensor, z_a: torch.Tensor, postprocessing_transform: Optional[object] = None
    ) -> np.ndarray:
        """Runs the inference on a single image.

        Args:
            image: A numpy image to run the inference on. This is the absorbance image. It must have a shape of
                (H, W, C).
            z_a: The attribute tensor for the domain to reconstruct the image.
            postprocessing_transform (optional): The optional transformation to apply on the top of the image to
                produce the final results. Defaults to None, in which case, no transformation will be performed.

        Returns:
            An numpy image that contains the inference result in the target space. This is the absorbace image.
                Hence, it needs to be converted into the
        """
        z_a = z_a.cuda()
        out_image = np.zeros_like(image)
        image_shape_2d = image.shape[:2]
        for r_slice, c_slice, r_ext_slice, c_ext_slice in self._tile_slices_iterator(image_shape_2d=image_shape_2d):
            logging.info(
                f"Infering in row slice = {r_slice}, col slice = {c_slice}, r_ext_slice = {r_ext_slice}, "
                + f"c_ext_slice = {c_ext_slice}"
            )
            tile = image[r_ext_slice, c_ext_slice, :]
            tile = np.transpose(tile, (2, 0, 1))  # To (C, H, W)
            if tile.dtype == np.uint8:
                tile = tile.astype(np.float32) / 255.0

            tile_output = self._infer_one_tile(tile, z_a=z_a)
            out_image[r_slice, c_slice] = tile_output.transpose((1, 2, 0))

        if postprocessing_transform is not None:
            out_image = postprocessing_transform(out_image)
        return out_image

    def _tile_slices_iterator(
        self, image_shape_2d: Tuple[int, int]
    ) -> Generator[Tuple[slice, slice, slice, slice], None, None]:
        """Returns a tuple of slices to run the inference and place the infered image.

        Returns:
            The row slice to extract the image.
            The column slice to extract the image.
            The row slice to extract the image.
            The column slice to extract the image.
        """
        nrows, ncols = image_shape_2d
        for tile_r in range(self._tile_margin_pixels, nrows - self._tile_margin_pixels, self._inner_tile_size_pixels):
            for tile_c in range(
                self._tile_margin_pixels, ncols - self._tile_margin_pixels, self._inner_tile_size_pixels
            ):

                # Adjust the begining of the tile.
                if tile_r + self._inner_tile_size_pixels + self._tile_margin_pixels > nrows:
                    tile_r = nrows - (self._inner_tile_size_pixels + self._tile_margin_pixels)
                if tile_c + self._inner_tile_size_pixels + self._tile_margin_pixels > ncols:
                    tile_c = ncols - (self._inner_tile_size_pixels + self._tile_margin_pixels)

                r_slice, c_slice = slice(tile_r, tile_r + self._inner_tile_size_pixels), slice(
                    tile_c, tile_c + self._inner_tile_size_pixels
                )
                r_ext_slice, c_ext_slice = self._create_extended_slices_to_cover_current_tile(
                    (r_slice, c_slice), image_shape_2d
                )
                yield (r_slice, c_slice, r_ext_slice, c_ext_slice)

    def _create_extended_slices_to_cover_current_tile(
        self, tile_slices: Tuple[slice, slice], image_shape: Tuple[int, int]
    ):
        """Returns slices of extended row and columns to cover the current tile."""
        row_slice, col_slice = tile_slices
        nrows, ncols = image_shape
        left_limit_to_cover = max(0, col_slice.start - self._tile_margin_pixels)
        right_limit_to_cover = min(ncols, col_slice.stop + self._tile_margin_pixels)

        top_limit_to_cover = max(0, row_slice.start - self._tile_margin_pixels)
        bottom_limit_to_cover = min(nrows, row_slice.stop + self._tile_margin_pixels)

        return slice(top_limit_to_cover, bottom_limit_to_cover), slice(left_limit_to_cover, right_limit_to_cover)

    def _infer_one_tile(self, tile: np.ndarray, z_a: torch.Tensor) -> np.ndarray:
        """Performs the inference for a single tile.

        Args:
            tile: An input tile of dimensions (C, H, W) with all pixel values in [0.0, 1.0] that contains the absorbance
                The shape should be (1, C, H, W).
            z_a: The attribute tensor for the domain to reconstruct the image. The shape should be (1, #attrs, 1, 1).

        Returns:
            The inference result of dimension (H, W)
        """
        tile = torch.from_numpy(tile)
        tile = tile.float()[None, :, :, :]
        tile = tile.cuda()
        with torch.no_grad():
            z_c = self._content_encoder(tile)
            return torch.squeeze(self._gen(z_c, z_a), axis=0).to("cpu").numpy()  # Single channel image
