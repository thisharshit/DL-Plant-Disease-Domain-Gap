"""
Dataset loading, class alignment, and transformation pipelines for PlantVillage (Source Domain)
and PlantDoc (Target Domain).
"""

import os
import glob
import urllib.request
import zipfile
import shutil
from typing import Dict, List, Tuple, Optional, Callable
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 1-to-1 Class Mapping across 8 Aligned Disease Classes:
# Source Domain: PlantVillage (Laboratory)
# Target Domain: PlantDoc (Real Field Conditions)
COMMON_CLASSES_MAP = {
    "Tomato_Early_Blight": {
        "plantvillage": "Tomato___Early_blight",
        "plantdoc": "Tomato Early blight leaf"
    },
    "Tomato_Late_Blight": {
        "plantvillage": "Tomato___Late_blight",
        "plantdoc": "Tomato leaf late blight"
    },
    "Tomato_Bacterial_Spot": {
        "plantvillage": "Tomato___Bacterial_spot",
        "plantdoc": "Tomato leaf bacterial spot"
    },
    "Tomato_Septoria_Spot": {
        "plantvillage": "Tomato___Septoria_leaf_spot",
        "plantdoc": "Tomato Septoria leaf spot"
    },
    "Potato_Early_Blight": {
        "plantvillage": "Potato___Early_blight",
        "plantdoc": "Potato leaf early blight"
    },
    "Potato_Late_Blight": {
        "plantvillage": "Potato___Late_blight",
        "plantdoc": "Potato leaf late blight"
    },
    "Bell_Pepper_Bacterial_Spot": {
        "plantvillage": "Pepper,_bell___Bacterial_spot",
        "plantdoc": "Bell_pepper leaf spot"
    },
    "Apple_Scab": {
        "plantvillage": "Apple___Apple_scab",
        "plantdoc": "Apple Scab leaf"
    }
}

CLASS_NAMES = list(COMMON_CLASSES_MAP.keys())
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)  # Exactly 8 classes aligned across both domains


class PlantDiseaseDataset(Dataset):
    """
    Standard PyTorch Dataset for Plant disease classification.
    Can be used for both Source Domain (PlantVillage) and Target Domain (PlantDoc).
    """
    def __init__(self, image_paths: List[str], labels: List[int], transform: Optional[Callable] = None):
        # ---------------------------------------------------------------------
        # VIVA / CONCEPTUAL NOTE: LAZY LOADING & MEMORY MANAGEMENT
        # We store filepaths (strings) and integer labels in memory, NOT raw pixel arrays.
        # Images are loaded lazily on-the-fly inside __getitem__(). This prevents
        # Out-Of-Memory (OOM) crashes on large multi-gigabyte vision repositories.
        # ---------------------------------------------------------------------
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        # Returns total number of samples; required by PyTorch DataLoader to compute epoch steps
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # ---------------------------------------------------------------------
        # Step-by-step sample retrieval:
        # 1. Fetch image path by index
        # 2. Open via PIL and convert to RGB: Ensures 3 channels (discards PNG alpha channels
        #    and converts 1-channel grayscale images to uniform 3-channel tensors)
        # 3. Apply augmentation/normalization pipeline
        # ---------------------------------------------------------------------
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_standard_transforms(img_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """
    Baseline data augmentation and normalization.
    """
    # -------------------------------------------------------------------------
    # VIVA / CONCEPTUAL NOTE: IMAGENET NORMALIZATION
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225].
    # Why normalize? Pretrained weights (ResNet, ViT, MobileNet) were trained on
    # normalized ImageNet distribution with zero mean and unit variance.
    # Formula: x_norm = (x - mean) / std.
    # This prevents vanishing/exploding gradients in early convolutional layers.
    # -------------------------------------------------------------------------
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    if is_train:
        # Standard training augmentations: modest horizontal/vertical flips
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor(),
            norm
        ])
    else:
        # Validation/Test transform: deterministic resizing without random perturbation
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            norm
        ])


def get_heavy_augmentations(img_size: int = 224) -> transforms.Compose:
    """
    Technique 1: Heavy Field-Simulating Data Augmentations.
    Simulates variations in outdoor natural sunlight, shadow, camera angle, and leaf occlusions.
    """
    # -------------------------------------------------------------------------
    # VIVA / CONCEPTUAL NOTE: SIMULATING COVARIATE SHIFT
    # Why Heavy Augmentations? Models trained on PlantVillage overfit to uniform studio
    # lighting and plain backgrounds. We synthesize natural field perturbations:
    # 1. RandomResizedCrop(0.7, 1.0): Simulates framing variation and multi-scale symptoms.
    # 2. ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.15):
    #    Simulates harsh Indian direct sunlight, specular highlights, and chromatic shifts.
    # 3. GaussianBlur: Simulates smartphone motion blur and out-of-focus optics.
    # 4. RandomErasing (Cutout): Erases random rectangular regions (p=0.3), forcing
    #    the network to recognize distributed lesion patterns rather than a single shortcut.
    # -------------------------------------------------------------------------
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.15),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])


def collect_matched_data(root_dir: str, domain: str = "plantvillage", split: Optional[str] = None) -> Tuple[List[str], List[int]]:
    """
    Scans a dataset directory and filters only the 8 common overlapping classes.
    domain: 'plantvillage' or 'plantdoc'
    split: optional 'train' or 'test' subdirectory filter to avoid data leakage
    """
    # -------------------------------------------------------------------------
    # VIVA / CONCEPTUAL NOTE: DATASET ALIGNMENT & SPLIT ISOLATION
    # PlantVillage contains 38 classes; PlantDoc contains 27 classes.
    # Only 8 classes share exact disease and botanical semantics across both.
    # This function extracts ONLY the matched classes and enforces split isolation
    # (train vs test) to guarantee ZERO data leakage during evaluation.
    # -------------------------------------------------------------------------
    image_paths = []
    labels = []

    target_dir = os.path.join(root_dir, split) if split and os.path.exists(os.path.join(root_dir, split)) else root_dir

    for unified_class, mapping in COMMON_CLASSES_MAP.items():
        sub_folder_name = mapping[domain]
        class_idx = CLASS_TO_IDX[unified_class]

        matched_folders = []
        for dirpath, dirnames, _ in os.walk(target_dir):
            for d in dirnames:
                if sub_folder_name.lower() in d.lower():
                    matched_folders.append(os.path.join(dirpath, d))

        if not matched_folders:
            continue

        for folder in matched_folders:
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                found_files = glob.glob(os.path.join(folder, ext))
                for f in found_files:
                    image_paths.append(f)
                    labels.append(class_idx)

    return image_paths, labels

    target_dir = os.path.join(root_dir, split) if split and os.path.exists(os.path.join(root_dir, split)) else root_dir

    for unified_class, mapping in COMMON_CLASSES_MAP.items():
        sub_folder_name = mapping[domain]
        class_idx = CLASS_TO_IDX[unified_class]

        matched_folders = []
        for dirpath, dirnames, _ in os.walk(target_dir):
            for d in dirnames:
                if sub_folder_name.lower() in d.lower():
                    matched_folders.append(os.path.join(dirpath, d))

        if not matched_folders:
            continue

        for folder in matched_folders:
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                found_files = glob.glob(os.path.join(folder, ext))
                for f in found_files:
                    image_paths.append(f)
                    labels.append(class_idx)

    return image_paths, labels


def download_plantdoc_dataset(target_dir: str = "./data/plantdoc") -> str:
    """
    Clones or downloads the official PlantDoc dataset from GitHub repository.
    """
    os.makedirs(target_dir, exist_ok=True)
    zip_url = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
    zip_path = os.path.join(target_dir, "plantdoc_repo.zip")

    dest_dir = os.path.join(target_dir, "PlantDoc-Dataset-master")
    if not os.path.exists(dest_dir):
        print(f"Downloading PlantDoc dataset from {zip_url}...")
        urllib.request.urlretrieve(zip_url, zip_path)
        print("Extracting PlantDoc dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print("[OK] PlantDoc ready!")

    return dest_dir


def download_plantvillage_dataset(target_dir: str = "./data/plantvillage", max_images_per_class: int = 100) -> str:
    """
    Downloads authentic PlantVillage dataset (Mohanty et al., 2016)
    and selectively extracts the 8 overlapping laboratory classes.
    """
    os.makedirs(target_dir, exist_ok=True)
    zip_url = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
    zip_path = os.path.join(target_dir, "plantvillage_master.zip")

    # Check if all 8 class folders already exist
    pv_classes = [m["plantvillage"] for m in COMMON_CLASSES_MAP.values()]
    existing = [c for c in pv_classes if any(c.lower() in d.lower() for d in os.listdir(target_dir))]
    if len(existing) == len(pv_classes):
        print("[OK] PlantVillage clean laboratory dataset already available.")
        return target_dir

    dest_folder_base = target_dir
    print("Downloading PlantVillage laboratory benchmark dataset from GitHub...")
    urllib.request.urlretrieve(zip_url, zip_path)
    print("Selectively extracting 8 aligned laboratory classes...")

    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.namelist():
            for cls in pv_classes:
                if f"/color/{cls}/" in member and member.lower().endswith((".jpg", ".jpeg", ".png")):
                    filename = os.path.basename(member)
                    if filename:
                        dest_folder = os.path.join(dest_folder_base, cls)
                        os.makedirs(dest_folder, exist_ok=True)
                        if len(os.listdir(dest_folder)) < max_images_per_class:
                            with z.open(member) as src_file, open(os.path.join(dest_folder, filename), "wb") as dst_file:
                                dst_file.write(src_file.read())

    if os.path.exists(zip_path):
        os.remove(zip_path)
    print("[OK] PlantVillage 8-class laboratory dataset extracted successfully!")
    return target_dir
