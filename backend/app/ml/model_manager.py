"""Model manager for loading and managing ML models"""
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from app.ml.models.unet import create_unet
from app.ml.models.room_classifier import create_room_classifier
from app.ml.models.opening_detector import create_opening_detector

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages loading and caching of ML models
    
    Implements singleton pattern to ensure models are loaded only once
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model storage
        self.models: Dict[str, torch.nn.Module] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        
        # Default model paths (can be configured via environment)
        self.model_dir = Path("models")
        self.model_paths = {
            "unet": self.model_dir / "unet_wall_segmentation.pth",
            "room_classifier": self.model_dir / "room_classifier.pth",
            "opening_detector": self.model_dir / "opening_detector.pth"
        }
    
    def load_model(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
        force_reload: bool = False
    ) -> torch.nn.Module:
        """
        Load a model by name
        
        Args:
            model_name: Name of the model ('unet', 'room_classifier', 'opening_detector')
            model_path: Optional custom path to model weights
            force_reload: Force reload even if already loaded
        
        Returns:
            Loaded model
        """
        # Return cached model if available
        if model_name in self.models and not force_reload:
            logger.info(f"Using cached {model_name} model")
            return self.models[model_name]
        
        logger.info(f"Loading {model_name} model...")
        
        # Create model based on name
        if model_name == "unet":
            model = self._load_unet(model_path)
        elif model_name == "room_classifier":
            model = self._load_room_classifier(model_path)
        elif model_name == "opening_detector":
            model = self._load_opening_detector(model_path)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        # Cache the model
        self.models[model_name] = model
        logger.info(f"{model_name} model loaded successfully")
        
        return model
    
    def _load_unet(self, model_path: Optional[Path] = None) -> torch.nn.Module:
        """Load U-Net wall segmentation model"""
        model = create_unet(n_channels=3, n_classes=2, pretrained=False)
        
        path = model_path or self.model_paths["unet"]
        if path.exists():
            logger.info(f"Loading U-Net weights from {path}")
            model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            logger.warning(f"No pretrained weights found at {path}. Using randomly initialized model.")
            logger.warning("For production use, train the model and provide weights file.")
        
        return model
    
    def _load_room_classifier(self, model_path: Optional[Path] = None) -> torch.nn.Module:
        """Load room classification model"""
        model = create_room_classifier(pretrained=False, ensemble=False)
        
        path = model_path or self.model_paths["room_classifier"]
        if path.exists():
            logger.info(f"Loading room classifier weights from {path}")
            model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            logger.warning(f"No pretrained weights found at {path}. Using randomly initialized model.")
            logger.warning("For production use, train the model and provide weights file.")
        
        return model
    
    def _load_opening_detector(self, model_path: Optional[Path] = None) -> torch.nn.Module:
        """Load opening detection model"""
        # Use simplified detector for now (lighter and faster)
        model = create_opening_detector(num_classes=3, pretrained=False, simplified=True)
        
        path = model_path or self.model_paths["opening_detector"]
        if path.exists():
            logger.info(f"Loading opening detector weights from {path}")
            model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            logger.warning(f"No pretrained weights found at {path}. Using randomly initialized model.")
            logger.warning("For production use, train the model and provide weights file.")
        
        return model
    
    def get_model(self, model_name: str) -> torch.nn.Module:
        """
        Get a loaded model (load if not already loaded)
        
        Args:
            model_name: Name of the model
        
        Returns:
            Loaded model
        """
        if model_name not in self.models:
            self.load_model(model_name)
        return self.models[model_name]
    
    def unload_model(self, model_name: str):
        """
        Unload a model from memory
        
        Args:
            model_name: Name of the model to unload
        """
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Unloaded {model_name} model")
    
    def unload_all_models(self):
        """Unload all models from memory"""
        self.models.clear()
        logger.info("Unloaded all models")
    
    def get_device(self) -> torch.device:
        """Get the device used for inference"""
        return self.device
    
    def to_onnx(self, model_name: str, output_path: Path, **kwargs):
        """
        Export a model to ONNX format for optimization
        
        Args:
            model_name: Name of the model to export
            output_path: Path to save ONNX model
            **kwargs: Additional arguments for torch.onnx.export
        """
        model = self.get_model(model_name)
        
        # Create dummy input based on model type
        if model_name == "unet":
            dummy_input = torch.randn(1, 3, 512, 512, device=self.device)
        elif model_name == "room_classifier":
            dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        elif model_name == "opening_detector":
            dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            **kwargs
        )
        
        logger.info(f"Exported {model_name} to ONNX at {output_path}")


# Global model manager instance
model_manager = ModelManager()
