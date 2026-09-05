"""
Neural Network Model Architectures for Global Dog Breed Identification.

Implements MobileNetV2 Transfer Learning Architecture with 356 Softmax outputs.
Base model: torchvision.models.mobilenet_v2(pretrained=True).
Custom classification head: Dropout(0.3) -> Linear(1280, 1024) -> ReLU -> BatchNorm1d -> Dropout(0.3) -> Linear(1024, 356).
Output Softmax order matches data/processed/dog_breeds_356.json (0..355).
"""

import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 356

class DogBreedMobileNet(nn.Module):
    """
    MobileNetV2 Transfer Learning Model for 356 Dog Breed Classes.
    Pretrained ImageNet backbone + Custom Classification Head.
    """
    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=True):
        super(DogBreedMobileNet, self).__init__()
        
        try:
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        except Exception:
            self.backbone = models.mobilenet_v2(pretrained=True)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_upper_layers(self, num_blocks=4):
        """Unfreezes upper feature blocks of MobileNetV2 for Step B fine-tuning."""
        for param in self.backbone.features[-num_blocks:].parameters():
            param.requires_grad = True


def build_model(model_type='transfer', num_classes=NUM_CLASSES, freeze_backbone=True):
    """Factory function to instantiate MobileNetV2 Transfer Model."""
    model = DogBreedMobileNet(num_classes=num_classes, freeze_backbone=freeze_backbone)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ MobileNetV2 Transfer Model Initialized ({num_classes} classes)")
    print(f"  • Total Parameters:     {total_params:,}")
    print(f"  • Trainable Parameters: {trainable_params:,}")
    return model

# Alias for backwards compatibility
get_model = build_model
