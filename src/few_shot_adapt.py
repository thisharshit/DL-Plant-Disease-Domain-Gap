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
    # -------------------------------------------------------------------------
    # VIVA / CONCEPTUAL NOTE: FEW-SHOT SAMPLING & DATA LEAKAGE PREVENTION
    # - What is k-shot learning? A domain adaptation regime where only 'k' labeled
    #   target examples per class are available during training (here k=10, total 80 images).
    # - Stratified Grouping: We group indices by class using defaultdict(list)
    #   to guarantee uniform balance across all 8 crop disease categories.
    # - Zero Data Leakage: Crucially, this function must ONLY be called on the
    #   PlantDoc *training* split, never on the official PlantDoc *test* split.
    # -------------------------------------------------------------------------
    random.seed(seed)
    class_indices = defaultdict(list)

    # Collect indices belonging to each distinct class label
    for idx, label in enumerate(dataset.labels):
        class_indices[label].append(idx)

    selected_indices = []
    for cls, idxs in class_indices.items():
        # If class has fewer than k samples, take all; otherwise sample exactly k uniformly
        if len(idxs) <= k_shots:
            selected_indices.extend(idxs)
        else:
            selected_indices.extend(random.sample(idxs, k_shots))

    # Return a lightweight PyTorch Subset referencing only the selected sample indices
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
    # -------------------------------------------------------------------------
    # VIVA / CONCEPTUAL NOTE: FINE-TUNING DYNAMICS & HYPERPARAMETERS
    # - Why a very small Learning Rate (lr = 5e-5)?
    #   The model was already trained on PlantVillage. A large learning rate would cause
    #   "Catastrophic Forgetting" (erasing general feature detectors learned from 1000s of lab images).
    #   A small LR gently adjusts the linear classifier and top conv layers to align
    #   with outdoor illumination, shadows, and natural background statistics.
    # - Why AdamW with Weight Decay (1e-3)?
    #   AdamW decouples weight decay (L2 penalty) from gradient updates:
    #   w_t = w_{t-1} - lr * (m_t / (sqrt(v_t) + eps) + lambda * w_{t-1}).
    #   This strong regularization prevents overfitting on the tiny 80-image few-shot set.
    # -------------------------------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Freeze lower layers and fine-tune upper layers or entire backbone with low LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Batch size: capped at min(16, total samples) to ensure stable mini-batch gradient estimates
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
