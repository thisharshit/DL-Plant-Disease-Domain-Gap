"""
Model architectures: ResNet-50, MobileNetV2, and Vision Transformer (ViT).
Provides standard backbones with customizable classification heads.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


def get_model(
    model_name: str = "resnet50",
    num_classes: int = 10,
    pretrained: bool = True
) -> nn.Module:
    """
    Factory function to initialize vision models for Plant Disease Classification.

    Supported model_name options:
    - 'resnet18'
    - 'resnet50'
    - 'mobilenet_v2' (lightweight edge-friendly Indian Agriculture baseline)
    - 'vit_b_16' (Vision Transformer)
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    elif model_name in ["vit", "vit_b_16"]:
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    return model
