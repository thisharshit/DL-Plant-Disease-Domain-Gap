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
    # -------------------------------------------------------------------------
    # VIVA / CONCEPTUAL NOTE: WHY HSV COLOR SPACE INSTEAD OF RGB?
    # - In RGB, pixel values combine color and illumination simultaneously:
    #   if lighting changes (sunlight vs shadow), R, G, and B all shift together,
    #   making thresholding fragile.
    # - In HSV:
    #   H (Hue): Represents pure chromatic wavelength (0-180 in OpenCV: green ~35-85, yellow ~20-30).
    #   S (Saturation): Represents color vibrancy/purity (0 = grayscale, 255 = fully saturated).
    #   V (Value): Represents brightness/illumination intensity (0 = black, 255 = bright).
    # Decoupling Hue from Value allows isolating plant leaf tissue regardless of outdoor lighting.
    # -------------------------------------------------------------------------
    img_np = np.array(image)
    if img_np.ndim == 2:
        # Grayscale edge-case: convert to 3-channel RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    # Convert RGB array to HSV color space
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # -------------------------------------------------------------------------
    # HSV Threshold Range for Foliar Tissue & Lesions:
    # Lower bound [15, 25, 25]: Captures brownish/yellow chlorotic and necrotic lesion margins.
    # Upper bound [90, 255, 255]: Captures healthy dark green foliar lamina.
    # Pixels within [lower_leaf, upper_leaf] become 255 (White), others become 0 (Black).
    # -------------------------------------------------------------------------
    lower_leaf = np.array([15, 25, 25])
    upper_leaf = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

    # -------------------------------------------------------------------------
    # VIVA / CONCEPTUAL NOTE: MORPHOLOGICAL OPERATIONS (NOISE REMOVAL)
    # Binary masks contain speckle noise from dry soil and small holes inside the leaf.
    # 1. Morphological Closing (Dilation followed by Erosion):
    #    Fills small holes and cracks inside the leaf lamina without changing outer shape.
    # 2. Morphological Opening (Erosion followed by Dilation):
    #    Removes small isolated noise specks (e.g., weeds, soil particles in the background).
    # 3. Kernel: cv2.MORPH_ELLIPSE (5x5) matches smooth organic botanical leaf contours
    #    much better than a rigid rectangular kernel.
    # -------------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # -------------------------------------------------------------------------
    # Bitwise AND:
    # Multiplies the original RGB image by the binary mask.
    # Output: Foreground leaf pixels are retained with original RGB color,
    # non-leaf background pixels are clamped to black [0, 0, 0].
    #
    # KEY BENCHMARK FINDING (WHY THIS FAILS ON PLANTDOC FIELD PHOTOS):
    # Naive HSV masking drops accuracy in field conditions because:
    # 1. Harsh direct tropical sunlight creates intense white specular reflections (H clipped).
    # 2. Deep outdoor shadows cause V < 25 (leaf portions falsely masked as black).
    # 3. Brown necrotic lesions blend with brown soil background, causing border clipping.
    # -------------------------------------------------------------------------
    segmented = cv2.bitwise_and(img_np, img_np, mask=mask)
    return Image.fromarray(segmented)


class BackgroundRemovalTransform:
    """
    PyTorch-compatible Transform that removes non-leaf background before feeding to CNN.
    """
    def __init__(self, base_transform=None):
        # Wraps subsequent PyTorch transforms (Resize, ToTensor, Normalize)
        self.base_transform = base_transform

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Step 1: Strip background using HSV morphological filtering
        clean_img = remove_background_cv(img)
        # Step 2: Apply standard CNN normalization pipeline
        if self.base_transform:
            return self.base_transform(clean_img)
        return transforms.ToTensor()(clean_img)
