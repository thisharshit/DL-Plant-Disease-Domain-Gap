"""
Training engine for Plant Disease Models with validation tracking and checkpointing.
Supports execution via: python -m src.train
"""

import os
import time
import argparse
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import (
    PlantDiseaseDataset,
    get_standard_transforms,
    get_heavy_augmentations,
    collect_matched_data,
    download_plantdoc_dataset,
    NUM_CLASSES
)
from src.models import get_model
from src.evaluate import evaluate_dataset, compute_domain_gap, generate_benchmark_table


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total * 100.0) if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = (correct / total * 100.0) if total > 0 else 0.0
    return val_loss, val_acc


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 3e-4,
    device: Optional[torch.device] = None,
    checkpoint_path: str = "checkpoints/best_model.pth"
) -> Dict[str, List[float]]:
    """
    Complete training loop with learning rate scheduling and best-checkpoint saving.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    print(f"\n--- Starting Training on {device} for {epochs} epochs ---")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start_time
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Training completed. Best Validation Accuracy: {best_val_acc:.2f}%")
    return history


def main():
    parser = argparse.ArgumentParser(description="Plant Disease Cross-Domain Training & Evaluation Pipeline")
    parser.add_argument("--arch", type=str, default="mobilenet_v2", choices=["resnet18", "resnet50", "mobilenet_v2", "vit_b_16"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device} | Architecture: {args.arch}")

    # Acquire Target Dataset
    plantdoc_root = download_plantdoc_dataset("./data/plantdoc")
    field_train_paths, field_train_labels = collect_matched_data(plantdoc_root, domain="plantdoc", split="train")
    field_test_paths, field_test_labels = collect_matched_data(plantdoc_root, domain="plantdoc", split="test")

    # Fallback if train/test subfolders are flat
    if len(field_test_paths) == 0:
        all_paths, all_labels = collect_matched_data(plantdoc_root, domain="plantdoc")
        split_idx = int(0.2 * len(all_paths))
        field_test_paths, field_test_labels = all_paths[:split_idx], all_labels[:split_idx]
        field_train_paths, field_train_labels = all_paths[split_idx:], all_labels[split_idx:]

    print(f"Loaded {len(field_train_paths)} training images, {len(field_test_paths)} test images across {NUM_CLASSES} classes.")

    train_ds = PlantDiseaseDataset(field_train_paths, field_train_labels, transform=get_heavy_augmentations())
    test_ds = PlantDiseaseDataset(field_test_paths, field_test_labels, transform=get_standard_transforms(is_train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = get_model(args.arch, num_classes=NUM_CLASSES, pretrained=True)
    fit_model(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    print("\n--- Running Final Evaluation on Out-of-Domain Field Test Set ---")
    metrics = evaluate_dataset(model, test_loader, device=device)
    print(f"Final Field Accuracy: {metrics['accuracy']:.2f}% | Field Macro-F1: {metrics['f1_score']:.2f}%")


if __name__ == "__main__":
    main()
