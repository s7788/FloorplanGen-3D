"""File upload API endpoint"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from datetime import datetime
import uuid
import os
from pathlib import Path
from app.models.schemas import UploadResponse, JobStatus
from app.core.config import settings
from app.services.image_processor import ImageProcessor

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_floorplan(file: UploadFile = File(...)):
    """
    Upload a 2D floorplan image (JPG/PNG)
    
    Returns a job_id for tracking the processing status
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    # Read file content
    content = await file.read()
    
    # Validate file size
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create upload directory if not exists
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_path = upload_dir / f"{job_id}{file_ext}"
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Queue processing task (placeholder for now)
    # In production, this would trigger a Celery task
    # process_floorplan.delay(job_id, str(file_path))
    
    return UploadResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="File uploaded successfully. Processing queued.",
        upload_time=datetime.utcnow()
    )
