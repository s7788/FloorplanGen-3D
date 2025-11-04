"""AI-enhanced CV processing service for floorplan analysis"""
import cv2
import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from pathlib import Path

from app.ml.model_manager import model_manager
from app.ml.utils.preprocessing import (
    preprocess_for_unet,
    preprocess_for_classifier,
    preprocess_for_detection,
    postprocess_segmentation,
    extract_room_patch
)
from app.ml.models.room_classifier import get_room_type_name
from app.ml.models.opening_detector import get_opening_type_name, filter_detections


class AIImageProcessor:
    """
    AI-enhanced floorplan image processor using deep learning models
    
    Replaces rule-based CV processing with trained neural networks for:
    - Wall segmentation using U-Net
    - Room classification using ResNet
    - Door/window detection using object detection
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize AI image processor
        
        Args:
            use_gpu: Whether to use GPU if available
        """
        self.device = model_manager.get_device()
        self.min_room_area = 1000  # Minimum area for valid rooms
        
    def process_floorplan(self, image_path: str) -> Dict[str, Any]:
        """
        Process a floorplan image using AI models
        
        Args:
            image_path: Path to the floorplan image
            
        Returns:
            Dictionary containing walls, rooms, doors, and windows
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process walls using U-Net segmentation
        walls, wall_mask = self._detect_walls_ai(image_rgb)
        
        # Process rooms using segmentation + classification
        rooms = self._detect_and_classify_rooms_ai(image_rgb, wall_mask)
        
        # Process doors and windows using object detection
        doors = self._detect_openings_ai(image_rgb, "door")
        windows = self._detect_openings_ai(image_rgb, "window")
        
        return {
            "walls": walls,
            "rooms": rooms,
            "doors": doors,
            "windows": windows,
            "metadata": {
                "image_size": image.shape[:2],
                "processing_version": "2.0-ai-enhanced",
                "device": str(self.device)
            }
        }
    
    def _detect_walls_ai(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Detect walls using U-Net segmentation model
        
        Args:
            image: Input image (RGB)
        
        Returns:
            Tuple of (wall list, binary wall mask)
        """
        # Load U-Net model
        model = model_manager.get_model("unet")
        
        # Preprocess image
        original_size = (image.shape[0], image.shape[1])
        input_tensor = preprocess_for_unet(image).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Postprocess to get binary mask
        wall_mask = postprocess_segmentation(output, original_size, threshold=0.5)
        
        # Extract wall lines from mask
        walls = self._extract_wall_lines(wall_mask)
        
        return walls, wall_mask
    
    def _extract_wall_lines(self, wall_mask: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract wall line segments from binary mask
        
        Args:
            wall_mask: Binary wall segmentation mask
        
        Returns:
            List of wall dictionaries with start/end coordinates
        """
        # Find edges
        edges = cv2.Canny(wall_mask, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        walls = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate wall thickness from mask
                thickness = self._estimate_wall_thickness_from_mask(
                    wall_mask, x1, y1, x2, y2
                )
                
                walls.append({
                    "start": {"x": int(x1), "y": int(y1)},
                    "end": {"x": int(x2), "y": int(y2)},
                    "thickness": float(thickness)
                })
        
        return walls
    
    def _estimate_wall_thickness_from_mask(
        self,
        wall_mask: np.ndarray,
        x1: int, y1: int, x2: int, y2: int
    ) -> float:
        """
        Estimate wall thickness by sampling perpendicular to the line
        
        Args:
            wall_mask: Binary wall mask
            x1, y1, x2, y2: Line endpoints
        
        Returns:
            Estimated thickness in pixels
        """
        # Calculate perpendicular direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 1:
            return 10.0
        
        # Normalize and get perpendicular
        px = -dy / length
        py = dx / length
        
        # Sample at line midpoint
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Sample perpendicular thickness
        thickness = 0
        for dist in range(1, 30):
            # Check both sides
            for sign in [1, -1]:
                sx = int(mx + sign * dist * px)
                sy = int(my + sign * dist * py)
                
                if 0 <= sx < wall_mask.shape[1] and 0 <= sy < wall_mask.shape[0]:
                    if wall_mask[sy, sx] > 0:
                        thickness = dist * 2
        
        return max(5.0, float(thickness))
    
    def _detect_and_classify_rooms_ai(
        self,
        image: np.ndarray,
        wall_mask: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect room regions and classify their types
        
        Args:
            image: Input image (RGB)
            wall_mask: Binary wall mask
        
        Returns:
            List of room dictionaries with type classification
        """
        # Invert wall mask to get room regions
        room_mask = cv2.bitwise_not(wall_mask)
        
        # Find contours (room boundaries)
        contours, _ = cv2.findContours(
            room_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Load room classifier
        classifier = model_manager.get_model("room_classifier")
        
        rooms = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter small regions
            if area < self.min_room_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # Classify room type
            room_type, confidence = self._classify_room_type(
                image, (cx, cy), classifier
            )
            
            rooms.append({
                "id": f"room_{i}",
                "area": float(area),
                "center": {"x": cx, "y": cy},
                "bounds": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "type": room_type,
                "confidence": float(confidence)
            })
        
        return rooms
    
    def _classify_room_type(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        classifier: torch.nn.Module
    ) -> Tuple[str, float]:
        """
        Classify room type using ResNet classifier
        
        Args:
            image: Input image (RGB)
            center: Room center coordinates
            classifier: Room classification model
        
        Returns:
            Tuple of (room_type, confidence)
        """
        # Extract patch around room center
        patch = extract_room_patch(image, center, size=224)
        
        # Preprocess for classifier
        input_tensor = preprocess_for_classifier(patch).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = classifier(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
        
        room_type = get_room_type_name(pred_class.item())
        
        return room_type, confidence.item()
    
    def _detect_openings_ai(
        self,
        image: np.ndarray,
        opening_type: str
    ) -> List[Dict[str, Any]]:
        """
        Detect doors or windows using object detection
        
        Args:
            image: Input image (RGB)
            opening_type: Type of opening ('door' or 'window')
        
        Returns:
            List of opening detections
        """
        # Load detection model
        detector = model_manager.get_model("opening_detector")
        
        # Preprocess image
        input_tensor, original_size = preprocess_for_detection(image)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            # For simplified detector
            if hasattr(detector, 'classifier'):
                # Sliding window approach for simplified detector
                detections = self._sliding_window_detection(
                    image, detector, opening_type
                )
            else:
                # Faster R-CNN approach
                predictions = detector([input_tensor])[0]
                detections = self._process_detection_output(
                    predictions, opening_type
                )
        
        return detections
    
    def _sliding_window_detection(
        self,
        image: np.ndarray,
        detector: torch.nn.Module,
        opening_type: str
    ) -> List[Dict[str, Any]]:
        """
        Sliding window detection for simplified detector
        
        Args:
            image: Input image
            detector: Detection model
            opening_type: Type of opening to detect
        
        Returns:
            List of detections
        """
        # For MVP, return empty list
        # In production, implement sliding window with NMS
        return []
    
    def _process_detection_output(
        self,
        predictions: Dict[str, torch.Tensor],
        opening_type: str
    ) -> List[Dict[str, Any]]:
        """
        Process detection model output
        
        Args:
            predictions: Model predictions
            opening_type: Type of opening to filter
        
        Returns:
            List of filtered detections
        """
        # Map opening type to class index
        type_map = {"door": 1, "window": 2}
        target_class = type_map.get(opening_type, 0)
        
        # Filter detections
        filtered = filter_detections(
            predictions,
            confidence_threshold=0.5,
            iou_threshold=0.5
        )
        
        # Convert to output format
        detections = []
        for box, label, score in zip(
            filtered['boxes'],
            filtered['labels'],
            filtered['scores']
        ):
            if label == target_class:
                x1, y1, x2, y2 = box
                detections.append({
                    "bbox": {
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1)
                    },
                    "type": opening_type,
                    "confidence": float(score)
                })
        
        return detections
