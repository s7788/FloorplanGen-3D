"""ResNet-based room type classifier"""
import torch
import torch.nn as nn
from torchvision import models
from typing import List, Dict, Optional


# Room type labels
ROOM_TYPES = [
    "bedroom",
    "living_room",
    "kitchen",
    "bathroom",
    "dining_room",
    "office",
    "hallway",
    "storage",
    "balcony",
    "unknown"
]


class RoomClassifier(nn.Module):
    """
    ResNet-based classifier for room type classification
    
    Uses a pretrained ResNet-18 backbone with custom classifier head
    for room type prediction.
    """
    
    def __init__(self, num_classes: int = len(ROOM_TYPES), pretrained_backbone: bool = True):
        super(RoomClassifier, self).__init__()
        
        # Load pretrained ResNet-18
        self.backbone = models.resnet18(pretrained=pretrained_backbone)
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class RoomClassifierEnsemble(nn.Module):
    """
    Ensemble of room classifiers for improved accuracy
    
    Combines predictions from multiple models using voting or averaging
    """
    
    def __init__(self, num_classes: int = len(ROOM_TYPES), num_models: int = 3):
        super(RoomClassifierEnsemble, self).__init__()
        
        self.models = nn.ModuleList([
            RoomClassifier(num_classes=num_classes, pretrained_backbone=True)
            for _ in range(num_models)
        ])
        self.num_models = num_models
    
    def forward(self, x):
        # Average predictions from all models
        outputs = torch.stack([model(x) for model in self.models])
        return torch.mean(outputs, dim=0)


def create_room_classifier(
    num_classes: int = len(ROOM_TYPES),
    pretrained: bool = False,
    ensemble: bool = False
) -> nn.Module:
    """
    Factory function to create a room classifier
    
    Args:
        num_classes: Number of room types to classify
        pretrained: Whether to load pretrained weights
        ensemble: Whether to use ensemble of models
    
    Returns:
        Room classifier model instance
    """
    if ensemble:
        model = RoomClassifierEnsemble(num_classes=num_classes)
    else:
        model = RoomClassifier(num_classes=num_classes, pretrained_backbone=True)
    
    if pretrained:
        # In production, load pretrained weights here
        # model.load_state_dict(torch.load('path/to/weights.pth'))
        pass
    
    return model


def get_room_type_name(class_idx: int) -> str:
    """
    Get room type name from class index
    
    Args:
        class_idx: Index of the room class
    
    Returns:
        Room type name as string
    """
    if 0 <= class_idx < len(ROOM_TYPES):
        return ROOM_TYPES[class_idx]
    return "unknown"


def get_room_type_index(room_name: str) -> int:
    """
    Get class index from room type name
    
    Args:
        room_name: Name of the room type
    
    Returns:
        Index of the room class
    """
    try:
        return ROOM_TYPES.index(room_name.lower())
    except ValueError:
        return ROOM_TYPES.index("unknown")
