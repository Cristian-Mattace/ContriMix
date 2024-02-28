"""A utility module to generate differnet transforms."""
import copy
from enum import auto
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from PIL import Image

from ip_drit.patch_transform.data_augmentation._randaugment import FIX_MATCH_AUGMENTATION_POOL
from ip_drit.patch_transform.data_augmentation._randaugment import RandAugment


class TransformationType(Enum):
    """The type transformation."""

    WEAK_NORMALIZE_TO_0_1 = auto()
    WEAK = auto()
    RANDAUGMENT = auto()
    RANDAUGMENT_TO_0_1 = auto()
    RXRX1 = auto()
    PDL1 = auto()
    HISTAUGAN_ENCODERS = auto()
    HISTAUGAN_BACKBONE = auto()


_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]


class RandomRotation(nn.Module):
    """A module that does rotation by a multiple of 90 degrees."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = [0, 90, 180, 270]
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = tf.rotate(x, angle)
        return x


def initialize_transform(
    transform_name: Optional[TransformationType],
    # TODO: move the dataset out of here!
    full_dataset,
    config_dict: Dict[str, Any],
    is_training: bool = True,
) -> Optional[Callable]:
    """Generate a transformation pipeline that can be used to transform the input images.

    By default, transforms should take in `x` and return `transformed_x`.

    Args:
        transform_name: The name of the transform to generate.
        full_dataset: The full dataset to apply the transformation on.
        config_dict: A configuration dictionary.
        is_training (optional): If True, the training mode is used. Defaults to True.

    Returns:
        A callable that can be used to transform the input image x.
    """
    base_transform_steps = _get_image_base_transform_steps(config_dict, full_dataset)

    default_normalization = transforms.Normalize(
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN, _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD
    )

    if transform_name is None:
        return None
    elif transform_name == TransformationType.WEAK:
        return _add_weak_transform(base_transform_steps, normalize=True, default_normalization=default_normalization)
    elif transform_name == TransformationType.WEAK_NORMALIZE_TO_0_1:
        return _add_weak_transform(base_transform_steps, normalize=False, default_normalization=default_normalization)
    elif transform_name == TransformationType.RANDAUGMENT:
        return _add_rand_augment_transform(
            config_dict, full_dataset, base_transform_steps, normalize=True, default_normalization=default_normalization
        )
    elif transform_name == TransformationType.RANDAUGMENT_TO_0_1:
        return _add_rand_augment_transform(
            config_dict,
            full_dataset,
            base_transform_steps,
            normalize=False,
            default_normalization=default_normalization,
        )
    elif transform_name == TransformationType.RXRX1:
        return _get_rxrx1_transform(is_training=is_training)
    elif transform_name == TransformationType.HISTAUGAN_BACKBONE:
        return _get_histaugan_backbone_transform(is_training=is_training)
    else:
        raise ValueError(f"Unsupported transformation type!")


def _get_image_base_transform_steps(config, dataset) -> List[Callable]:
    transform_steps = []

    if dataset.original_resolution is not None and min(dataset.original_resolution) != max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))

    if config["target_resolution"] is not None:
        transform_steps.append(transforms.Resize(config["target_resolution"]))

    return transform_steps


def _add_weak_transform(base_transform_steps: List[Callable], normalize: bool, default_normalization):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    weak_transform_steps = copy.deepcopy(base_transform_steps)
    weak_transform_steps.extend([RandomRotation(), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    if normalize:
        weak_transform_steps.append(default_normalization)
    return transforms.Compose(weak_transform_steps)


def _add_rand_augment_transform(
    config: Dict[str, Any], dataset, base_transform_steps: List[Callable], normalize: bool, default_normalization
) -> transforms.Compose:
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = _get_target_resolution(config, dataset)
    strong_transform_steps = copy.deepcopy(base_transform_steps)
    strong_transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=target_resolution),
            RandAugment(n=config["randaugment_n"], augmentation_pool=FIX_MATCH_AUGMENTATION_POOL),
            transforms.ToTensor(),
            default_normalization if normalize else None,
        ]
    )
    return transforms.Compose(strong_transform_steps)


def _get_target_resolution(config, dataset):
    if config["target_resolution"] is not None:
        return config["target_resolution"]
    else:
        return dataset.original_resolution


def _get_rxrx1_transform(is_training: bool):
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angles = [0, 90, 180, 270]
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = tf.rotate(x, angle)
        return x

    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    def normalize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.0] = 1.0
        return tf.normalize(x, mean, std)

    t_normalize = transforms.Lambda(lambda x: normalize(x))

    if is_training:
        return transforms.Compose(
            [t_random_rotation, transforms.RandomHorizontalFlip(), transforms.ToTensor(), t_normalize]
        )
    else:
        return transforms.Compose([transforms.ToTensor(), t_normalize])


def _get_histaugan_backbone_transform(is_training: bool, resize_size_pixels: int = 256, crop_size_pixels: int = 216):
    """Gets the transform for HistauGAN backbone training.

    The image will be resize to that of the input encoder first.
    """
    transforms_pipeline = [transforms.Resize((crop_size_pixels, crop_size_pixels), Image.BICUBIC)]
    transforms_pipeline.append(transforms.RandomHorizontalFlip())
    transforms_pipeline.append(transforms.ToTensor())
    transforms_pipeline.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return transforms.Compose(transforms_pipeline)
