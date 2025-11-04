# Phase 2: AI Enhancement - Technical Documentation

## Overview

Phase 2 replaces the rule-based computer vision processing with deep learning models to achieve higher accuracy in floorplan analysis.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AI Processing Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input Image → Preprocessing → Model Inference → Post-       │
│                                                  processing   │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   U-Net      │  │   ResNet     │  │  Faster RCNN │      │
│  │ Wall Segment │  │ Room Classify│  │ Door/Window  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. U-Net Wall Segmentation

**Purpose**: Semantic segmentation of walls in floorplan images

**Architecture**:
- Encoder: 5 downsampling blocks (64→128→256→512→1024 channels)
- Decoder: 5 upsampling blocks with skip connections
- Output: 2-channel logits (wall vs. background)

**Input**: RGB image (512x512)
**Output**: Binary segmentation mask

**Training**:
```python
from app.ml.models.unet import create_unet
from app.ml.utils.training import Trainer

model = create_unet(n_channels=3, n_classes=2)
trainer = Trainer(model, train_loader, val_loader)
trainer.train(num_epochs=50)
```

### 2. Room Type Classification

**Purpose**: Classify detected rooms into types (bedroom, kitchen, etc.)

**Architecture**:
- Backbone: ResNet-18 (pretrained on ImageNet)
- Custom head: FC(512) → ReLU → Dropout(0.3) → FC(10)
- Output: 10 room type classes

**Room Types**:
1. Bedroom
2. Living Room
3. Kitchen
4. Bathroom
5. Dining Room
6. Office
7. Hallway
8. Storage
9. Balcony
10. Unknown

**Input**: RGB image patch (224x224)
**Output**: Room type probabilities

### 3. Door/Window Detection

**Purpose**: Detect and localize doors and windows

**Architecture Options**:

**Option A - Faster R-CNN** (Higher accuracy):
- Backbone: ResNet-50 with FPN
- RPN for region proposals
- ROI heads for classification and bbox regression

**Option B - Simplified CNN** (Faster inference):
- 3 Conv layers with pooling
- Adaptive pooling + FC layers
- Sliding window detection

**Input**: RGB image (variable size)
**Output**: Bounding boxes with labels

## Model Manager

The `ModelManager` class handles model lifecycle:

```python
from app.ml.model_manager import model_manager

# Load model (cached after first load)
model = model_manager.get_model("unet")

# Get device
device = model_manager.get_device()

# Export to ONNX
model_manager.to_onnx("unet", "unet.onnx")

# Unload to free memory
model_manager.unload_model("unet")
```

**Features**:
- Singleton pattern for efficiency
- Lazy loading
- Model caching
- GPU/CPU support
- ONNX export

## Preprocessing Pipeline

### For U-Net Segmentation

```python
from app.ml.utils.preprocessing import preprocess_for_unet

tensor = preprocess_for_unet(
    image,
    target_size=(512, 512),
    normalize=True
)
```

Steps:
1. Resize to 512x512
2. Convert to tensor
3. Normalize with ImageNet stats

### For Room Classification

```python
from app.ml.utils.preprocessing import preprocess_for_classifier

tensor = preprocess_for_classifier(
    patch,
    target_size=(224, 224),
    normalize=True
)
```

Steps:
1. Resize to 256x256
2. Center crop to 224x224
3. Convert to tensor
4. Normalize with ImageNet stats

### For Detection

```python
from app.ml.utils.preprocessing import preprocess_for_detection

tensor, orig_size = preprocess_for_detection(image)
```

Steps:
1. Optional resize
2. Convert to tensor
3. No normalization (model handles it)

## Post-processing

### Segmentation

```python
from app.ml.utils.preprocessing import postprocess_segmentation

mask = postprocess_segmentation(
    output,
    original_size=(height, width),
    threshold=0.5
)
```

Steps:
1. Apply softmax
2. Extract wall probability
3. Resize to original size
4. Apply threshold

### Detection

```python
from app.ml.models.opening_detector import filter_detections

filtered = filter_detections(
    predictions,
    confidence_threshold=0.5,
    iou_threshold=0.5
)
```

Steps:
1. Filter by confidence
2. Apply NMS (Non-Maximum Suppression)
3. Convert to output format

## Configuration

Add to `.env`:

```bash
# Toggle AI processing
USE_AI_PROCESSING=true

# Model directory
MODEL_DIR=./models

# GPU usage
USE_GPU=true
```

## Usage

### Switching Between Processors

```python
from app.services.processor_factory import get_image_processor

# Returns AI processor if USE_AI_PROCESSING=true
# Otherwise returns rule-based processor
processor = get_image_processor()
result = processor.process_floorplan("image.jpg")
```

### Direct AI Processing

```python
from app.services.ai_image_processor import AIImageProcessor

processor = AIImageProcessor(use_gpu=True)
result = processor.process_floorplan("image.jpg")
```

## Training Pipeline

### 1. Data Preparation

Organize dataset:
```
data/
├── wall_segmentation/
│   ├── images/
│   └── masks/
├── room_classification/
│   ├── bedroom/
│   ├── kitchen/
│   └── ...
└── opening_detection/
    ├── images/
    └── annotations/
```

### 2. Data Augmentation

```python
from app.ml.utils.preprocessing import augment_image

augmented = augment_image(
    image,
    rotation_range=15.0,
    flip=True,
    brightness_range=0.2
)
```

### 3. Training

```python
from app.ml.utils.training import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer
)

trainer.train(num_epochs=50, save_best=True)
```

### 4. Evaluation

```python
from app.ml.utils.training import calculate_dice_score, calculate_iou

dice = calculate_dice_score(predictions, targets)
iou = calculate_iou(predictions, targets)
```

## Performance Optimization

### 1. ONNX Export

```python
model_manager.to_onnx(
    "unet",
    output_path="models/unet.onnx"
)
```

Benefits:
- Faster inference
- Cross-platform compatibility
- Optimized for deployment

### 2. Model Quantization

```python
import torch.quantization as quantization

quantized_model = quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)
```

Benefits:
- Reduced model size
- Faster CPU inference
- Lower memory usage

### 3. Batch Processing

Process multiple images in a batch:
```python
batch = torch.stack([img1, img2, img3])
with torch.no_grad():
    outputs = model(batch)
```

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Wall Detection Accuracy | > 90% | TBD* |
| Room Classification Accuracy | > 85% | TBD* |
| Opening Detection mAP | > 80% | TBD* |
| Inference Time (GPU) | < 10s | TBD* |
| Inference Time (CPU) | < 30s | TBD* |

*TBD: To be determined after model training

## Next Steps

1. **Data Collection**: Gather and annotate floorplan dataset
2. **Model Training**: Train models on collected data
3. **Hyperparameter Tuning**: Optimize model architectures and training parameters
4. **Performance Testing**: Benchmark on test set
5. **Production Deployment**: Deploy trained models

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or image size:
```python
preprocess_for_unet(image, target_size=(256, 256))
```

### Slow Inference

1. Enable GPU: `USE_GPU=true`
2. Export to ONNX
3. Use model quantization
4. Reduce image resolution

### Low Accuracy

1. Collect more training data
2. Increase data augmentation
3. Train for more epochs
4. Use ensemble of models

## References

- U-Net: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- ResNet: [He et al., 2015](https://arxiv.org/abs/1512.03385)
- Faster R-CNN: [Ren et al., 2015](https://arxiv.org/abs/1506.01497)
