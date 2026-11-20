"""
Technique 3: Few-Shot Target Domain Fine-Tuning.
Adapts a pre-trained lab model to the field domain (PlantDoc) using a small budget
of labeled field images (e.g., 5-shot or 10-shot per class).
"""

import random
from collections import defaultdict
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import PlantDiseaseDataset, get_standard_transforms
from src.train import train_one_epoch, validate


def sample_k_shots(
    dataset: PlantDiseaseDataset,
    k_shots: int = 5,
    seed: int = 42
) -> Subset:
    """
    Samples exactly k images per class from the target domain dataset.
    """
    random.seed(seed)
    class_indices = defaultdict(list)

    for idx, label in enumerate(dataset.labels):
        class_indices[label].append(idx)

    selected_indices = []
    for cls, idxs in class_indices.items():
        if len(idxs) <= k_shots:
            selected_indices.extend(idxs)
        else:
            selected_indices.extend(random.sample(idxs, k_shots))

    return Subset(dataset, selected_indices)


def adapt_few_shot(
    model: nn.Module,
    target_train_subset: Subset,
    target_val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 5e-5,
    device: torch.device = None
) -> nn.Module:
    """
    Performs low-learning-rate fine-tuning on the k-shot target samples.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Freeze lower layers and fine-tune upper layers or entire backbone with low LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    few_shot_loader = DataLoader(
        target_train_subset,
        batch_size=min(16, len(target_train_subset)),
        shuffle=True
    )

    print(f"\n--- Few-Shot Adaptation ({len(target_train_subset)} field images) ---")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, few_shot_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, target_val_loader, criterion, device)
        print(f"Few-Shot Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Field Val Acc: {val_acc:.2f}%")

    return model
