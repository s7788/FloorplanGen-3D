"""Job status API endpoint"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from app.models.schemas import JobStatusResponse, JobStatus

router = APIRouter()

# In-memory job storage (replace with Redis/Database in production)
job_storage = {}


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the processing status of a job
    
    Returns current status, progress, and result URL if completed
    """
    # Mock response for now - in production, query from Redis/Database
    if job_id not in job_storage:
        # Return a default pending status for demonstration
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            progress=0,
            message="Job is queued for processing",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    return job_storage[job_id]
