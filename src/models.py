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
    # -------------------------------------------------------------------------
    # VIVA / CONCEPTUAL NOTE: TRANSFER LEARNING
    # Why Pretrained Weights? Pretraining on ImageNet-1k (1.28M images, 1000 classes)
    # initializes filters to detect universal visual primitives: edges, color gradients,
    # and textures in early layers, and complex organic shapes in deeper layers.
    # Training from scratch on small agricultural datasets causes severe overfitting.
    # -------------------------------------------------------------------------
    model_name = model_name.lower()

    if model_name == "resnet18":
        # ---------------------------------------------------------------------
        # ResNet-18 Backbone:
        # Uses BasicBlocks with skip connections (Residual: y = F(x) + x).
        # Solves vanishing gradient problem during backpropagation by providing
        # direct identity shortcut paths for gradient flow: dL/dx = dL/dy * (dF/dx + 1).
        # ---------------------------------------------------------------------
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        # Replace 1000-class ImageNet fully-connected (fc) head with task-specific head:
        # Dropout(0.3) randomly zeroes 30% of feature activations during training
        # to prevent co-adaptation of neurons (regularization).
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "resnet50":
        # ---------------------------------------------------------------------
        # ResNet-50 Backbone:
        # Uses BottleneckBlocks (1x1 reduce -> 3x3 conv -> 1x1 expand channel dim).
        # Total parameters: ~25.6M. High feature capacity on clean studio leaves (PlantVillage),
        # but prone to learning spurious background shortcuts (soil/lighting) when
        # transferred to outdoor field conditions (PlantDoc) without adaptation.
        # ---------------------------------------------------------------------
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        # Replace classification head for 8 plant disease categories
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "mobilenet_v2":
        # ---------------------------------------------------------------------
        # MobileNetV2 (Edge Architecture for Indian Agriculture):
        # Key Concept: Inverted Residuals & Linear Bottlenecks with Depthwise Separable Convolutions.
        # - Depthwise Convolution: Spatial filtering per channel (kernel: 3x3).
        # - Pointwise Convolution: Linear combination across channels (kernel: 1x1).
        # - Computational Advantage: Only 3.5M parameters (~14 MB model file),
        #   reducing FLOPs by 8-9x compared to standard convs. Runs efficiently
        #   offline on budget sub-$150 Android smartphones in rural Indian fields.
        # ---------------------------------------------------------------------
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        # Replace classifier head: MobileNetV2 stores final linear layer at index [1]
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    elif model_name in ["vit", "vit_b_16"]:
        # ---------------------------------------------------------------------
        # Vision Transformer (ViT-B/16 - Dosovitskiy et al., 2020):
        # Key Concept: Self-Attention without Convolutions.
        # 1. Patch Tokenization: 224x224 RGB image is sliced into non-overlapping
        #    16x16 patches -> (224/16)^2 = 14x14 = 196 patch tokens.
        # 2. Linear Projection: Each patch (16*16*3 = 768 dimensions) is projected to D=768.
        # 3. Position Embeddings: 1D learnable positional embeddings added to retain spatial order.
        # 4. Multi-Head Self-Attention (MHSA): Global receptive field across all patches from layer 1.
        # Notice: Lacks CNN's translation equivariance & inductive bias; requires pretraining
        # or heavy data augmentation to generalize on cluttered outdoor agricultural backgrounds.
        # ---------------------------------------------------------------------
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        # ViT places its classification head at model.heads.head
        model.heads.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    return model
