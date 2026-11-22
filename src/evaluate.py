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
    # -------------------------------------------------------------------------
    # VIVA / CONCEPTUAL NOTE: MODEL INFERENCE MODE
    # - Why model.eval()?
    #   Toggles layers that behave differently during training vs inference:
    #   1. Dropout: During training, neurons are randomly deactivated. In eval(),
    #      dropout is disabled and full activation weights are used.
    #   2. BatchNorm: During training, mean and variance are computed per batch.
    #      In eval(), frozen running mean and running variance are used.
    # - Why torch.no_grad()?
    #   Disables the dynamic autograd computation graph. Drastically reduces GPU
    #   memory usage and speeds up forward pass computation by ~2-3x.
    # -------------------------------------------------------------------------
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            # Forward pass: compute unnormalized class logits
            outputs = model(images)
            # Take argmax across class dimension to get predicted class index
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # -------------------------------------------------------------------------
    # VIVA / CONCEPTUAL NOTE: EVALUATION METRICS EXPLAINED
    # 1. Top-1 Accuracy: (Total Correct Predictions) / (Total Samples).
    #    Can be misleading if test dataset classes are imbalanced!
    # 2. Macro F1-Score: Macro-averaged harmonic mean of Precision and Recall:
    #    F1_c = 2 * (P_c * R_c) / (P_c + R_c), Macro-F1 = (1/C) * sum(F1_c).
    #    CRUCIAL FOR AGRICULTURE: Gives equal weight to every single crop disease class,
    #    ensuring that high accuracy on a dominant class cannot mask poor detection
    #    on a rare, economically devastating crop blight.
    # 3. Precision vs Recall:
    #    - Precision: Of all leaves flagged as disease X, what % actually have disease X? (Low false alarms)
    #    - Recall: Of all actual infected leaves, what % did the model detect? (Low missed infections)
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    # VIVA / MATHEMATICAL DEFINITION: DOMAIN GAP & RETENTION RATE
    # 1. Absolute Domain Gap (Delta):
    #    Delta = Acc_in (PlantVillage) - Acc_out (PlantDoc)
    #    Measures how many percentage points of diagnostic accuracy are lost
    #    due to covariate shift (background clutter, sunlight, occlusions).
    # 2. Retention Rate (rho):
    #    rho = (Acc_out / Acc_in) * 100%
    #    Quantifies what fraction of the laboratory capability survives
    #    when deployed out-of-domain into real agricultural fields.
    #    Example: 92.19% -> 24.32% gives retention rho = 26.4%.
    # -------------------------------------------------------------------------
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
    # Formats results dictionary into standard GitHub markdown table layout
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
