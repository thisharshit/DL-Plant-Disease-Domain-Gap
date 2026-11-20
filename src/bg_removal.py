"""
Technique 2: Background Removal and Leaf Lesion Isolation.
Strips plain/noisy backgrounds using HSV color masking and morphological filtering
to force models to learn symptom-specific patterns rather than background artifacts.
"""

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms


def remove_background_cv(image: Image.Image) -> Image.Image:
    """
    Applies color segmentation in HSV color space to segment plant leaf tissue
    and mask out complex laboratory or outdoor soil/lighting backgrounds.
    """
    img_np = np.array(image)
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Green to yellow/brown leaf range in HSV
    lower_leaf = np.array([15, 25, 25])
    upper_leaf = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply mask
    segmented = cv2.bitwise_and(img_np, img_np, mask=mask)
    return Image.fromarray(segmented)


class BackgroundRemovalTransform:
    """
    PyTorch-compatible Transform that removes non-leaf background before feeding to CNN.
    """
    def __init__(self, base_transform=None):
        self.base_transform = base_transform

    def __call__(self, img: Image.Image) -> torch.Tensor:
        clean_img = remove_background_cv(img)
        if self.base_transform:
            return self.base_transform(clean_img)
        return transforms.ToTensor()(clean_img)
