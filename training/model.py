"""Arquitectura de la CNN para clasificar casillas (13 clases)."""
import torch.nn as nn
from torchvision import models

from config import CLASSES, MODEL_NAME


def build_model(name: str = MODEL_NAME, pretrained: bool = True) -> nn.Module:
    """Devuelve la red con la última capa adaptada a len(CLASSES)."""
    n = len(CLASSES)

    if name == "mobilenet_v3_small":
        w = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
    elif name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
    elif name == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, n)
    elif name == "simple_cnn":
        m = SimpleCNN(n)
    else:
        raise ValueError(f"Modelo desconocido: {name}")
    return m


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Congela/descongela todo salvo la cabeza (fase 1 vs fase 2)."""
    head = head_module(model)
    for p in model.parameters():
        p.requires_grad = trainable
    for p in head.parameters():
        p.requires_grad = True


def head_module(model: nn.Module) -> nn.Module:
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "fc"):
        return model.fc
    raise ValueError("No se encuentra la cabeza del modelo")


class SimpleCNN(nn.Module):
    """CNN propia desde cero, para comparar con la preentrenada. Entrada 3x128x64."""

    def __init__(self, n_classes: int):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))