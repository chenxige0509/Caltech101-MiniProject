"""
Model definitions: ViT required size and factory for CNN backbones (transfer learning).
"""

import torch
import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES, get_device

# ViT-B/16 expects 224x224 input (14x14 patches)
VIT_REQUIRED_SIZE = 224


def get_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build ImageNet-pretrained model with frozen backbone and new head for num_classes.
    model_name: one of 'resnet18', 'efficientnet_b0', 'vit_b16'
    """
    device = get_device()
    if model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        # Freeze all except fc
        for p in m.parameters():
            p.requires_grad = False
        for p in m.fc.parameters():
            p.requires_grad = True
    elif model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )
        for p in m.parameters():
            p.requires_grad = False
        for p in m.classifier.parameters():
            p.requires_grad = True
    elif model_name == "vit_b16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = m.heads.head.in_features
        m.heads.head = nn.Linear(in_features, num_classes)
        for p in m.parameters():
            p.requires_grad = False
        for p in m.heads.head.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return m.to(device)
