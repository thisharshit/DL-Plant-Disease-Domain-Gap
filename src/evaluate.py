"""
Cross-Domain Evaluation & Benchmark Generator:
Measures accuracy on PlantVillage (In-Domain) vs. PlantDoc (Out-of-Domain),
calculates Domain Gap (Delta), Macro F1, and prints formatted benchmark tables.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tabulate import tabulate


def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates a model on a given dataloader and returns standard metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    precision, recall, _, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "f1_score": macro_f1,
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "predictions": all_preds,
        "targets": all_targets
    }


def compute_domain_gap(
    in_domain_metrics: Dict[str, float],
    out_domain_metrics: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculates Domain Gap metrics:
    - Absolute Gap: InDomain_Acc - OutDomain_Acc
    - Retention Rate: (OutDomain_Acc / InDomain_Acc) * 100%
    """
    in_acc = in_domain_metrics["accuracy"]
    out_acc = out_domain_metrics["accuracy"]
    gap = in_acc - out_acc
    retention = (out_acc / in_acc * 100.0) if in_acc > 0 else 0.0

    return {
        "in_domain_acc": in_acc,
        "out_domain_acc": out_acc,
        "domain_gap": gap,
        "retention_rate": retention,
        "in_f1": in_domain_metrics["f1_score"],
        "out_f1": out_domain_metrics["f1_score"]
    }


def generate_benchmark_table(experiments_results: Dict[str, Dict[str, float]]) -> str:
    """
    Generates a clear academic cross-domain benchmark table.
    """
    headers = [
        "Technique / Architecture",
        "In-Domain Acc (Lab %)",
        "Cross-Domain Acc (Field %)",
        "Domain Gap (Δ %)",
        "Field F1 (%)",
        "Retention (%)"
    ]

    rows = []
    for model_name, res in experiments_results.items():
        rows.append([
            model_name,
            f"{res['in_domain_acc']:.2f}%",
            f"{res['out_domain_acc']:.2f}%",
            f"{res['domain_gap']:.2f}%",
            f"{res['out_f1']:.2f}%",
            f"{res['retention_rate']:.1f}%"
        ])

    table_str = tabulate(rows, headers=headers, tablefmt="github")
    return table_str
