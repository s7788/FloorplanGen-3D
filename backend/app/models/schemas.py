"""Data models for the application"""
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Job processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    """Response model for file upload"""
    job_id: str
    status: JobStatus
    message: str
    upload_time: datetime


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: JobStatus
    progress: int  # 0-100
    message: str
    result_url: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class FloorplanAnalysis(BaseModel):
    """Floorplan analysis result"""
    walls: list[Dict[str, Any]]
    rooms: list[Dict[str, Any]]
    doors: list[Dict[str, Any]]
    windows: list[Dict[str, Any]]
    metadata: Dict[str, Any]
