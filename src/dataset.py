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

# 1-to-1 Class Mapping between PlantVillage (Lab) and PlantDoc (Field)
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
        "plantvillage": "Pepper__bell___Bacterial_spot",
        "plantdoc": "Bell_pepper leaf spot"
    },
    "Apple_Scab": {
        "plantvillage": "Apple___Apple_scab",
        "plantdoc": "Apple Scab leaf"
    },
    "Corn_Rust": {
        "plantvillage": "Corn_(maize)___Common_rust_",
        "plantdoc": "Corn rust leaf"
    },
    "Grape_Black_Rot": {
        "plantvillage": "Grape___Black_rot",
        "plantdoc": "Grape leaf black rot"
    }
}

CLASS_NAMES = list(COMMON_CLASSES_MAP.keys())
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


class PlantDiseaseDataset(Dataset):
    """
    Standard PyTorch Dataset for Plant disease classification.
    Can be used for both Source Domain (PlantVillage) and Target Domain (PlantDoc).
    """
    def __init__(self, image_paths: List[str], labels: List[int], transform: Optional[Callable] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
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
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor(),
            norm
        ])
    else:
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
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])


def collect_matched_data(root_dir: str, domain: str = "plantvillage") -> Tuple[List[str], List[int]]:
    """
    Scans a dataset directory and filters only the common overlapping classes.
    domain: 'plantvillage' or 'plantdoc'
    """
    image_paths = []
    labels = []

    for unified_class, mapping in COMMON_CLASSES_MAP.items():
        sub_folder_name = mapping[domain]
        class_idx = CLASS_TO_IDX[unified_class]

        # Case-insensitive / partial folder search
        matched_folders = []
        for dirpath, dirnames, _ in os.walk(root_dir):
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

    if not os.path.exists(os.path.join(target_dir, "PlantDoc-Dataset-master")):
        print(f"Downloading PlantDoc dataset from {zip_url}...")
        urllib.request.urlretrieve(zip_url, zip_path)
        print("Extracting PlantDoc dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(zip_path)
        print("PlantDoc ready!")

    return os.path.join(target_dir, "PlantDoc-Dataset-master")
