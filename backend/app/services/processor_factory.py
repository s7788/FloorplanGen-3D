"""Factory for creating image processor instances"""
from app.core.config import settings
from app.services.image_processor import ImageProcessor
from app.services.ai_image_processor import AIImageProcessor


def get_image_processor():
    """
    Factory function to get the appropriate image processor
    
    Returns either the AI-enhanced processor or the rule-based processor
    based on configuration settings.
    
    Returns:
        Image processor instance
    """
    if settings.USE_AI_PROCESSING:
        return AIImageProcessor(use_gpu=settings.USE_GPU)
    else:
        return ImageProcessor()
