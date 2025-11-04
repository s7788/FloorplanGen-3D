"""Training utilities for ML models"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Tuple, Dict, Any
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FloorplanDataset(Dataset):
    """
    Dataset for floorplan images and annotations
    
    Supports wall segmentation, room classification, and opening detection tasks
    """
    
    def __init__(
        self,
        data_dir: Path,
        task: str = "segmentation",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Args:
            data_dir: Directory containing dataset
            task: Task type ('segmentation', 'classification', 'detection')
            transform: Optional transform for images
            target_transform: Optional transform for targets
        """
        self.data_dir = Path(data_dir)
        self.task = task
        self.transform = transform
        self.target_transform = target_transform
        
        # Load file list based on task
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset file list"""
        # Implementation depends on dataset structure
        # This is a placeholder for demonstration
        pass
    
    def __len__(self) -> int:
        """Return dataset size"""
        return 0  # Placeholder
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index"""
        # Implementation depends on task
        # This is a placeholder for demonstration
        pass


class Trainer:
    """
    Trainer for ML models
    
    Handles training loop, validation, checkpointing, and logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Path = Path("checkpoints")
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Optional validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to use for training
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Default criterion and optimizer if not provided
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(
                    f'Train Epoch: {self.current_epoch} '
                    f'[{batch_idx * len(inputs)}/{len(self.train_loader.dataset)} '
                    f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                    f'Loss: {loss.item():.6f}'
                )
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model
        
        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        logger.info(
            f'Validation: Average loss: {avg_loss:.4f}, '
            f'Accuracy: {correct}/{total} ({accuracy:.2f}%)'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(self, num_epochs: int, save_best: bool = True):
        """
        Train the model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            save_best: Whether to save best model checkpoint
        """
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            # Validate
            if self.val_loader:
                val_metrics = self.validate()
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                
                # Save best model
                if save_best and val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pth')
                    logger.info(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        logger.info("Training completed")
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Dice score for segmentation
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth segmentation mask
    
    Returns:
        Dice score (0-1)
    """
    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Intersection over Union for segmentation
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth segmentation mask
    
    Returns:
        IoU score (0-1)
    """
    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)
