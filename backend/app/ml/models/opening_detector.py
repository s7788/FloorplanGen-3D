"""Object detection model for doors and windows in floorplans"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import List, Dict, Tuple, Optional


# Opening types
OPENING_TYPES = {
    0: "background",
    1: "door",
    2: "window"
}


class OpeningDetector(nn.Module):
    """
    Faster R-CNN based detector for doors and windows
    
    Uses a pretrained Faster R-CNN with ResNet-50 backbone
    Fine-tuned for detecting door and window openings in floorplans
    """
    
    def __init__(self, num_classes: int = 3, pretrained_backbone: bool = True):
        """
        Args:
            num_classes: Number of classes including background (3: background, door, window)
            pretrained_backbone: Whether to use pretrained backbone
        """
        super(OpeningDetector, self).__init__()
        
        # Load pretrained Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(pretrained=pretrained_backbone)
        
        # Get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    def forward(self, images, targets=None):
        """
        Args:
            images: List of images (each a tensor of shape [C, H, W])
            targets: List of target dictionaries (for training)
        
        Returns:
            During training: loss dictionary
            During inference: list of prediction dictionaries
        """
        return self.model(images, targets)


class SimplifiedOpeningDetector(nn.Module):
    """
    Simplified CNN-based detector for doors and windows
    
    A lighter alternative to Faster R-CNN for faster inference
    Uses a sliding window approach with CNN classification
    """
    
    def __init__(self, num_classes: int = 3):
        super(SimplifiedOpeningDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_opening_detector(
    num_classes: int = 3,
    pretrained: bool = False,
    simplified: bool = False
) -> nn.Module:
    """
    Factory function to create an opening detector
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to load pretrained weights
        simplified: Whether to use simplified detector (faster but less accurate)
    
    Returns:
        Opening detector model instance
    """
    if simplified:
        model = SimplifiedOpeningDetector(num_classes=num_classes)
    else:
        model = OpeningDetector(num_classes=num_classes, pretrained_backbone=True)
    
    if pretrained:
        # In production, load pretrained weights here
        # model.load_state_dict(torch.load('path/to/weights.pth'))
        pass
    
    return model


def get_opening_type_name(class_idx: int) -> str:
    """
    Get opening type name from class index
    
    Args:
        class_idx: Index of the opening class
    
    Returns:
        Opening type name as string
    """
    return OPENING_TYPES.get(class_idx, "background")


def filter_detections(
    predictions: Dict[str, torch.Tensor],
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5
) -> Dict[str, List]:
    """
    Filter detections based on confidence and apply NMS
    
    Args:
        predictions: Dictionary with 'boxes', 'labels', 'scores' tensors
        confidence_threshold: Minimum confidence score
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Dictionary with filtered detections
    """
    # Filter by confidence
    keep = predictions['scores'] > confidence_threshold
    
    boxes = predictions['boxes'][keep]
    labels = predictions['labels'][keep]
    scores = predictions['scores'][keep]
    
    # Apply NMS
    from torchvision.ops import nms
    keep_nms = nms(boxes, scores, iou_threshold)
    
    return {
        'boxes': boxes[keep_nms].cpu().numpy().tolist(),
        'labels': labels[keep_nms].cpu().numpy().tolist(),
        'scores': scores[keep_nms].cpu().numpy().tolist()
    }
