# ML Models for FloorplanGen-3D Phase 2

This directory contains the deep learning models used for AI-enhanced floorplan processing.

## Models

### 1. U-Net Wall Segmentation (`unet_wall_segmentation.pth`)
- **Purpose**: Semantic segmentation of walls in floorplan images
- **Architecture**: U-Net with ResNet backbone
- **Input**: RGB image (512x512)
- **Output**: Binary segmentation mask (wall vs. background)

### 2. Room Type Classifier (`room_classifier.pth`)
- **Purpose**: Classify room types (bedroom, kitchen, bathroom, etc.)
- **Architecture**: ResNet-18 with custom classification head
- **Input**: RGB image patch (224x224)
- **Output**: 10 room type classes

### 3. Opening Detector (`opening_detector.pth`)
- **Purpose**: Detect and classify doors and windows
- **Architecture**: Faster R-CNN or simplified CNN
- **Input**: RGB image (variable size)
- **Output**: Bounding boxes with labels (door/window)

## Model Training

### Prerequisites
- Python 3.11+
- PyTorch 2.1+
- CUDA-capable GPU (recommended)
- Training dataset (to be collected)

### Data Collection
For each model, you'll need:

1. **Wall Segmentation**
   - Floorplan images
   - Binary masks with wall annotations
   - Recommended: 1000+ samples

2. **Room Classification**
   - Cropped room patches
   - Room type labels
   - Recommended: 500+ samples per class

3. **Opening Detection**
   - Floorplan images
   - Bounding box annotations for doors/windows
   - Recommended: 2000+ annotations

### Training Scripts
Training scripts will be added in future updates. For now, models use pretrained backbones with randomly initialized heads.

## Usage

Models are automatically loaded by the `ModelManager` class when AI processing is enabled:

```python
from app.services.ai_image_processor import AIImageProcessor

processor = AIImageProcessor()
result = processor.process_floorplan("path/to/floorplan.jpg")
```

## Model Format

Models are saved as PyTorch `.pth` files containing the state dictionary:

```python
torch.save(model.state_dict(), 'model_name.pth')
```

## ONNX Export

For optimized inference, models can be exported to ONNX format:

```python
from app.ml.model_manager import model_manager

model_manager.to_onnx(
    "unet",
    output_path="models/unet_wall_segmentation.onnx"
)
```

## Performance Targets (Phase 2)

- Wall detection accuracy: > 90%
- Room classification accuracy: > 85%
- Door/window detection mAP: > 80%
- Inference time: < 30 seconds per image

## Notes

- Models are not included in the repository due to size
- For development/testing, the system works with randomly initialized weights
- For production use, train models on your dataset and place weights in this directory
- Consider using model quantization for faster inference on CPU
