"""Generates augmented version of the training images for mitosis detection."""
import logging
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch

from constants import INPUT_PATCH_SIZE_PIXELS
from constants import NUM_STAIN_VECTORS
from constants import TEST_IM_SIZE_PIXELS
from ip_drit.datamodule import MultiDomainDataset
from ip_drit.inference import Inferencer
from ip_drit.models import AbsorbanceImGenerator
from ip_drit.models import AttributeEncoder
from ip_drit.models import ContentEncoder
from ip_drit.patch_transform import AbsorbanceToTransmittance
from ip_drit.patch_transform import RGBToTransmittance
from ip_drit.patch_transform import ToTensor
from ip_drit.patch_transform import TransmittanceToAbsorbance
from ip_drit.sampling import generate_sample_lists_by_domain_indices
from utils import load_trained_model_from_checkpoint
from utils import prepare_training_dataset

TARGET_PATCH_SIZE_PIXELS = 512
MIDOG_FOLDER = Path("/jupyter-users-home/tan-2enguyen/public_data/MIDOG_2021")
data_folder = MIDOG_FOLDER / "images"
sample_folder = MIDOG_FOLDER / "point_samples"
checkpoint_folder = MIDOG_FOLDER / "checkpoints"
mitosis_detection_folder = MIDOG_FOLDER / "patches_for_mitosis_detection"

logging.basicConfig(level=logging.DEBUG)


def main() -> None:
    """Demo script to run the inference."""
    logging.info("Running inference.")
    pretrain_model_path = checkpoint_folder / (
        "Stain_separation_32_stain_vectors_24960_bs_8_samples_w_real_fake_weight_0.5_recon_weight_20.0_content_"
        + "consistency_weight_10.0_attr_consistency_weight_1.0_mode_seeking_loss_weight_5.0_content_channel_"
        + "covariance_loss_weight_0.2_MIDOG_v1.1/every_n/"
        + "periodic_epoch=119_val_encoders_generators_total_loss=3.866_train_encoders_generators_total_loss=3.369_train"
        + "_disc_total_loss=0.179.ckpt"
    )

    enc_c = load_trained_model_from_checkpoint(
        pretrain_model_path,
        network=ContentEncoder(in_channels=3, num_stain_vectors=NUM_STAIN_VECTORS),
        starts_str="_enc_c.",
    ).eval()

    enc_a = load_trained_model_from_checkpoint(
        pretrain_model_path,
        network=AttributeEncoder(in_channels=3, num_stain_vectors=NUM_STAIN_VECTORS),
        starts_str="_enc_a.",
    ).eval()
    gen = load_trained_model_from_checkpoint(
        pretrain_model_path, network=AbsorbanceImGenerator(), starts_str="_gen."
    ).eval()

    # If existing training dataset is available, reuse it.
    dataset_info = prepare_training_dataset(data_folder=data_folder)

    sample_lists_by_domain_idx = generate_sample_lists_by_domain_indices(
        dataset_info=dataset_info,
        sample_folder=sample_folder,
        patch_size_pixels=INPUT_PATCH_SIZE_PIXELS,
        max_num_samples_per_domain=100000,
    )

    all_domain_indices = [0, 1, 2, 3]
    all_sample_list_by_domain_index = {
        domain_idx: sample_lists_by_domain_idx[domain_idx][:3000] for domain_idx in all_domain_indices
    }

    abs_dataset = MultiDomainDataset(
        sample_list_by_domain_index=all_sample_list_by_domain_index,
        transforms=[RGBToTransmittance(), TransmittanceToAbsorbance()],
        input_patch_size_pixels=INPUT_PATCH_SIZE_PIXELS,
    )

    inferencer = Inferencer(content_encoder=enc_c, gen=gen, max_tile_size_pixels=TEST_IM_SIZE_PIXELS)

    # Get a sample image from each domain
    abs_image_by_domain_idx = {
        d: abs_dataset.get_item_with_domain_idx(sample_idx=10, domain_idx=d) for d in all_domain_indices
    }
    absorbance_to_rgb_trans = AbsorbanceToTransmittance()
    trans_image_by_domain_idx = {d: absorbance_to_rgb_trans(im) for d, im in abs_image_by_domain_idx.items()}
    original_montage_im = _combined_image_from_each_domain(image_by_domain_idx=trans_image_by_domain_idx)
    cv2.imwrite("original_montage.png", (original_montage_im * 255.0).astype(np.uint8))

    # Select an image for target attribute
    org_abs_im = abs_image_by_domain_idx[1]
    im_tensor = ToTensor()(org_abs_im)[None, :]
    _visualize_content(im=im_tensor, content_encoder=enc_c)
    z_a0 = enc_a.cuda()(im_tensor.cuda())
    z_c0 = enc_c.cuda()(im_tensor.cuda())
    out_im = gen.cuda()(z_c=z_c0, z_a=z_a0)
    self_recon_image = absorbance_to_rgb_trans(out_im[0].detach().cpu().numpy().transpose((1, 2, 0)))
    cv2.imwrite("original_im.png", (absorbance_to_rgb_trans(org_abs_im) * 255.0).astype(np.uint8))
    cv2.imwrite("self_recon_im.png", (self_recon_image * 255.0).astype(np.uint8))

    transformed_trans_image_by_domain_idx = {
        d: inferencer.infer_one_image(image=abs_im, z_a=z_a0, postprocessing_transform=absorbance_to_rgb_trans)
        for d, abs_im in abs_image_by_domain_idx.items()
    }
    transformed_montage_im = _combined_image_from_each_domain(image_by_domain_idx=transformed_trans_image_by_domain_idx)
    cv2.imwrite("transformed_montage.png", (transformed_montage_im * 255.0).astype(np.uint8))


def _combined_image_from_each_domain(image_by_domain_idx: Dict[int, np.ndarray]) -> None:
    any_im = list(image_by_domain_idx.values())[0]
    num_rows, num_cols, num_chans = any_im.shape
    # TODO: address the factors 2 here in case we have more than 4 domains.
    com_num_rows, com_num_cols = 2 * num_rows, 2 * num_cols
    out_im = np.zeros((com_num_rows, com_num_cols, num_chans), dtype=any_im.dtype)
    for domain_idx, im in image_by_domain_idx.items():
        row_idx = domain_idx // 2
        col_idx = domain_idx % 2
        out_im[row_idx * num_rows : (row_idx + 1) * num_rows, col_idx * num_cols : (col_idx + 1) * num_cols] = im
    return out_im


def _visualize_content(im: torch.Tensor, content_encoder: torch.nn.Module) -> None:
    # Visualize different content channels in the images.
    zc = content_encoder.cuda()(im.cuda())[0].detach().cpu().numpy()
    num_chans, im_num_rows, im_num_cols = zc.shape
    num_chan_sqrt = np.sqrt(num_chans)
    num_row_disp = int(num_chan_sqrt)
    num_col_disp = int(np.ceil(num_chans / num_row_disp))
    combined_num_rows, combined_num_cols = num_row_disp * im_num_rows, num_col_disp * im_num_cols
    combined_im = np.zeros((combined_num_rows, combined_num_cols), dtype=zc.dtype)
    for chan_idx, chan in enumerate(zc):
        row_idx = chan_idx // num_col_disp
        col_idx = chan_idx % num_col_disp
        combined_im[
            row_idx * im_num_rows : (row_idx + 1) * im_num_rows, col_idx * im_num_cols : (col_idx + 1) * im_num_cols
        ] = chan


if __name__ == "__main__":
    main()
