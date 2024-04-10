import torch
import numpy as np
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image


def my_gradcam(model, images, path):
    """
    We want to see in which areas of the images our model focuses most.
 
    Args:
        model   : our model.
        images  : the images to pass to the model.
        path    : where to save the generated images.
    """
    target_layers = [model.features.norm5]
    input_tensor = images

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    for i, _ in enumerate(images):
        grayscale_cam_img = grayscale_cam[i, :]

        transformed_tensor = input_tensor[i]

        original_tensor = transformed_tensor.clone()

        # Converts the tensor to a PIL image
        to_pil = transforms.ToPILImage()
        original_image = to_pil(original_tensor)

        # Restore the original size
        original_image = np.array(original_image) / 255.0

        visualization = show_cam_on_image(original_image, grayscale_cam_img, use_rgb=True)
    
        cv2.imwrite(path + f"/image{i}.png", cv2.hconcat([visualization * 1.0, original_image * 255]))
        
