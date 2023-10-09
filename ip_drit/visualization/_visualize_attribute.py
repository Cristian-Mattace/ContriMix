"""A module to visualize the attribute of the image."""
import numpy as np
import torch
import torchvision
from matplotlib import cm
from torch.nn.functional import interpolate


def visualize_content_channels(org_ims: torch.Tensor, zcs: torch.Tensor, y_true: torch.Tensor) -> np.ndarray:
    """Creates a combined image of the content channels.

    Args:
        org_ims: The original image tensor, in the range of [0.0, 1.0].
    """
    colors = _construct_colormap_list(num_colors=256, colormap=cm.get_cmap("hot"), device=zcs.device)
    NUM_IMAGES_TO_EVAL = 4
    combined_ims = []
    for im_idx in range(NUM_IMAGES_TO_EVAL):
        org_im_8bit = (org_ims[im_idx].unsqueeze(0) * 255.0).type(torch.uint8)
        min_val = torch.quantile(zcs[im_idx], q=0.01)
        max_val = torch.quantile(zcs[im_idx], q=0.99)

        zc = _convert_content_tensor_to_8_bit(zcs[im_idx], min_val=min_val, max_val=max_val)
        nims, _, h, w = zc.shape
        combined_im = torch.zeros([nims, 3, h, w], dtype=torch.uint8, device=zc.device)
        zc = _colorize_zc(zc, colors=colors)
        combined_im = zc
        combined_im_with_org = torch.concat([org_im_8bit, combined_im], dim=0)
        combined_ims.append(combined_im_with_org)

    combined_ims = torch.concat(combined_ims, dim=0)
    grid_im = torchvision.utils.make_grid(combined_ims, nrow=(nims + 1), normalize=False, value_range=[0, 255])
    return grid_im


def _construct_colormap_list(num_colors: int, colormap, device: str) -> torch.Tensor:
    return torch.tensor(
        [[x for x in colormap(idx)[:3]] for idx in range(num_colors)], device=device, dtype=torch.float32
    ).T


def _convert_content_tensor_to_8_bit(zc: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    zc = zc.unsqueeze(1)  # [C, 1, H, W]
    zc = (zc - min_val) / (max_val - min_val) * 255.0
    zc = torch.clamp(zc, min=0, max=255.0)
    return zc.type(torch.uint8)


def _colorize_zc(zc: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
    """Apply to a gray tensor a colormap.

    Adapted from kornia.
    The image data is assumed to be integer values.

    .. image:: _static/img/apply_colormap.png

    Args:
        zc: the input tensor of a gray image.
        colors: A tensor of available colors.

    Returns:
        A RGB tensor with the applied color map into the input_tensor.
    """
    if len(zc.shape) == 4 and zc.shape[1] == 1:  # if (B x 1 X H x W)
        zc = zc[:, 0, ...]  # (B x H x W)
    else:
        raise ValueError("Only 4 dimensional tensor is supported!")

    num_colors = colors.size(1)
    keys = torch.arange(0, num_colors, dtype=zc.dtype, device=zc.device)  # (num_colors)

    index = torch.bucketize(zc, keys)  # shape equals <input_tensor>: (B x H x W) or (H x W)

    color_interp = interpolate(colors[None, ...], size=num_colors, mode="linear")[0, ...]
    output = color_interp[:, index]  # (3 x B x H x W) or (3 x H x W)

    if len(output.shape) == 4:
        output = output.permute(1, 0, -2, -1)  # (B x 3 x H x W)

    return (output * 255.0).type(torch.uint8)  # (B x 3 x H x W) or (3 x H x W)
