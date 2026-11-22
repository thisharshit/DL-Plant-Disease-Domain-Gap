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
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import (
    PlantDiseaseDataset,
    get_standard_transforms,
    get_heavy_augmentations,
    collect_matched_data,
    download_plantdoc_dataset,
    download_plantvillage_dataset,
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
    epochs: int = 3,
    lr: float = 2e-4,
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

    print(f"Training completed. Best In-Domain Validation Accuracy: {best_val_acc:.2f}%")
    return history


def main():
    parser = argparse.ArgumentParser(description="Plant Disease Cross-Domain Training & Evaluation Pipeline")
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet18", "resnet50", "mobilenet_v2", "vit_b_16"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device} | Architecture: {args.arch}")

    # 1. Acquire Source Domain (PlantVillage - Lab)
    pv_root = download_plantvillage_dataset("./data/plantvillage")
    pv_paths, pv_labels = collect_matched_data(pv_root, domain="plantvillage")

    # 2. Acquire Target Domain (PlantDoc - Field)
    pd_root = download_plantdoc_dataset("./data/plantdoc")
    pd_test_paths, pd_test_labels = collect_matched_data(pd_root, domain="plantdoc", split="test")
    if len(pd_test_paths) == 0:
        all_pd_paths, all_pd_labels = collect_matched_data(pd_root, domain="plantdoc")
        split_idx = int(0.25 * len(all_pd_paths))
        pd_test_paths, pd_test_labels = all_pd_paths[:split_idx], all_pd_labels[:split_idx]

    print(f"PlantVillage Lab Images: {len(pv_paths)} across {NUM_CLASSES} classes.")
    print(f"PlantDoc Field Test Images: {len(pd_test_paths)} across {NUM_CLASSES} classes.")

    # Split PlantVillage into 80% train, 20% test
    split_pv = int(0.8 * len(pv_paths))
    pv_train_paths, pv_train_labels = pv_paths[:split_pv], pv_labels[:split_pv]
    pv_test_paths, pv_test_labels = pv_paths[split_pv:], pv_labels[split_pv:]

    train_ds = PlantDiseaseDataset(pv_train_paths, pv_train_labels, transform=get_heavy_augmentations())
    val_ds = PlantDiseaseDataset(pv_test_paths, pv_test_labels, transform=get_standard_transforms(is_train=False))
    field_test_ds = PlantDiseaseDataset(pd_test_paths, pd_test_labels, transform=get_standard_transforms(is_train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    field_loader = DataLoader(field_test_ds, batch_size=args.batch_size, shuffle=False)

    # 3. Train on Laboratory Data
    model = get_model(args.arch, num_classes=NUM_CLASSES, pretrained=True)
    fit_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

    # 4. Rigorous Cross-Domain Gap Evaluation
    print("\n--- Measuring Lab-to-Field Domain Gap ---")
    in_metrics = evaluate_dataset(model, val_loader, device=device)
    out_metrics = evaluate_dataset(model, field_loader, device=device)
    gap_metrics = compute_domain_gap(in_metrics, out_metrics)

    print(f"In-Domain Lab Accuracy (PlantVillage): {gap_metrics['in_domain_acc']:.2f}%")
    print(f"Cross-Domain Field Accuracy (PlantDoc): {gap_metrics['out_domain_acc']:.2f}%")
    print(f"Measured Domain Gap (Delta): {gap_metrics['domain_gap']:.2f} percentage points")
    print(f"Retention Rate (rho): {gap_metrics['retention_rate']:.1f}%")


if __name__ == "__main__":
    main()
