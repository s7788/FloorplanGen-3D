"""Celery tasks for asynchronous processing"""
from celery import Celery
from app.core.config import settings
from app.services.processor_factory import get_image_processor
from app.services.model_generator import ModelGenerator
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any

# Initialize Celery
celery_app = Celery(
    "floorplangen",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Ensure models directory exists at module load time
models_dir = Path(settings.UPLOAD_DIR) / "models"
models_dir.mkdir(parents=True, exist_ok=True)


@celery_app.task(bind=True)
def process_floorplan(self, job_id: str, file_path: str) -> Dict[str, Any]:
    """
    Process a floorplan image asynchronously
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the uploaded floorplan image
        
    Returns:
        Processing result with model URL
    """
    try:
        # Update status to processing
        self.update_state(
            state='PROCESSING',
            meta={'progress': 10, 'message': 'Starting image analysis...'}
        )
        
        # Step 1: Process image with CV (uses AI if enabled)
        processor = get_image_processor()
        analysis = processor.process_floorplan(file_path)
        
        self.update_state(
            state='PROCESSING',
            meta={'progress': 50, 'message': 'Generating 3D model...'}
        )
        
        # Step 2: Generate 3D model
        generator = ModelGenerator()
        output_path = models_dir / f"{job_id}.gltf"
        model_path = generator.generate_3d_model(analysis, str(output_path))
        
        self.update_state(
            state='PROCESSING',
            meta={'progress': 90, 'message': 'Finalizing...'}
        )
        
        # Save analysis result
        analysis_path = models_dir / f"{job_id}_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Return success result
        result = {
            'status': 'completed',
            'progress': 100,
            'message': 'Processing completed successfully',
            'model_url': f'/api/v1/models/{job_id}.gltf',
            'analysis_url': f'/api/v1/models/{job_id}_analysis.json',
            'completed_at': datetime.utcnow().isoformat()
        }
        
        return result
        
    except Exception as e:
        # Update state to failed
        self.update_state(
            state='FAILED',
            meta={
                'progress': 0,
                'message': f'Processing failed: {str(e)}',
                'error': str(e)
            }
        )
        raise
