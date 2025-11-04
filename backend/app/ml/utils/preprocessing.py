"""Image preprocessing utilities for ML models"""
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional, Union
from torchvision import transforms


def preprocess_for_unet(
    image: Union[np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (512, 512),
    normalize: bool = True
) -> torch.Tensor:
    """
    Preprocess image for U-Net wall segmentation
    
    Args:
        image: Input image (numpy array or PIL Image)
        target_size: Target size (height, width)
        normalize: Whether to normalize pixel values
    
    Returns:
        Preprocessed image tensor [1, C, H, W]
    """
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = Image.fromarray(image)
    
    # Define transforms
    transform_list = [
        transforms.Resize(target_size),
        transforms.ToTensor()
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    transform = transforms.Compose(transform_list)
    
    # Apply transforms and add batch dimension
    tensor = transform(image).unsqueeze(0)
    
    return tensor


def preprocess_for_classifier(
    image: Union[np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> torch.Tensor:
    """
    Preprocess image for room classification
    
    Args:
        image: Input image (numpy array or PIL Image)
        target_size: Target size (height, width)
        normalize: Whether to normalize pixel values
    
    Returns:
        Preprocessed image tensor [1, C, H, W]
    """
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = Image.fromarray(image)
    
    # Define transforms (following ImageNet preprocessing)
    transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(target_size),
        transforms.ToTensor()
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    transform = transforms.Compose(transform_list)
    
    # Apply transforms and add batch dimension
    tensor = transform(image).unsqueeze(0)
    
    return tensor


def preprocess_for_detection(
    image: Union[np.ndarray, Image.Image],
    target_size: Optional[Tuple[int, int]] = None
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess image for object detection
    
    Args:
        image: Input image (numpy array or PIL Image)
        target_size: Optional target size (height, width)
    
    Returns:
        Tuple of (preprocessed image tensor [C, H, W], original size)
    """
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        original_size = (image.shape[0], image.shape[1])
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = Image.fromarray(image)
    else:
        original_size = (image.height, image.width)
    
    # Define transforms
    transform_list = []
    
    if target_size:
        transform_list.append(transforms.Resize(target_size))
    
    transform_list.extend([
        transforms.ToTensor()
    ])
    
    transform = transforms.Compose(transform_list)
    
    # Apply transforms (no batch dimension for detection)
    tensor = transform(image)
    
    return tensor, original_size


def postprocess_segmentation(
    output: torch.Tensor,
    original_size: Tuple[int, int],
    threshold: float = 0.5
) -> np.ndarray:
    """
    Postprocess U-Net segmentation output
    
    Args:
        output: Model output logits [1, C, H, W]
        original_size: Original image size (height, width)
        threshold: Threshold for binary classification
    
    Returns:
        Binary segmentation mask as numpy array [H, W]
    """
    # Apply softmax and get probabilities
    probs = torch.softmax(output, dim=1)
    
    # Get wall probability (class 1)
    wall_prob = probs[0, 1, :, :].cpu().numpy()
    
    # Resize to original size
    wall_prob_resized = cv2.resize(wall_prob, (original_size[1], original_size[0]))
    
    # Apply threshold
    binary_mask = (wall_prob_resized > threshold).astype(np.uint8) * 255
    
    return binary_mask


def extract_room_patch(
    image: np.ndarray,
    center: Tuple[int, int],
    size: int = 128
) -> np.ndarray:
    """
    Extract a patch around a room center for classification
    
    Args:
        image: Input image
        center: Center coordinates (x, y)
        size: Patch size
    
    Returns:
        Extracted patch as numpy array
    """
    h, w = image.shape[:2]
    cx, cy = center
    
    # Calculate patch boundaries
    half_size = size // 2
    x1 = max(0, cx - half_size)
    y1 = max(0, cy - half_size)
    x2 = min(w, cx + half_size)
    y2 = min(h, cy + half_size)
    
    # Extract patch
    patch = image[y1:y2, x1:x2]
    
    # Pad if necessary
    if patch.shape[0] < size or patch.shape[1] < size:
        patch = cv2.copyMakeBorder(
            patch,
            0, max(0, size - patch.shape[0]),
            0, max(0, size - patch.shape[1]),
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
    
    return patch


def augment_image(
    image: Union[np.ndarray, Image.Image],
    rotation_range: float = 15.0,
    flip: bool = True,
    brightness_range: float = 0.2
) -> Union[np.ndarray, Image.Image]:
    """
    Apply data augmentation to image
    
    Args:
        image: Input image
        rotation_range: Maximum rotation angle in degrees
        flip: Whether to apply random horizontal flip
        brightness_range: Brightness adjustment range
    
    Returns:
        Augmented image
    """
    is_numpy = isinstance(image, np.ndarray)
    
    if is_numpy:
        image = Image.fromarray(image)
    
    # Random rotation
    if np.random.random() > 0.5:
        angle = np.random.uniform(-rotation_range, rotation_range)
        image = transforms.functional.rotate(image, angle)
    
    # Random horizontal flip
    if flip and np.random.random() > 0.5:
        image = transforms.functional.hflip(image)
    
    # Random brightness
    if np.random.random() > 0.5:
        factor = 1.0 + np.random.uniform(-brightness_range, brightness_range)
        image = transforms.functional.adjust_brightness(image, factor)
    
    if is_numpy:
        image = np.array(image)
    
    return image
