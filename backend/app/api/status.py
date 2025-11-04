"""Job status API endpoint"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from app.models.schemas import JobStatusResponse, JobStatus
from app.tasks import celery_app
from celery.result import AsyncResult

router = APIRouter()


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the processing status of a job
    
    Returns current status, progress, and result URL if completed
    """
    # Query Celery task result
    task_result = AsyncResult(job_id, app=celery_app)
    
    # Default response
    now = datetime.utcnow()
    
    if task_result.state == 'PENDING':
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            progress=0,
            message="Job is queued for processing",
            created_at=now,
            updated_at=now
        )
    elif task_result.state == 'PROCESSING':
        info = task_result.info or {}
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.PROCESSING,
            progress=info.get('progress', 50),
            message=info.get('message', 'Processing in progress...'),
            created_at=now,
            updated_at=now
        )
    elif task_result.state == 'SUCCESS':
        result = task_result.result or {}
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            message=result.get('message', 'Processing completed'),
            result_url=result.get('model_url'),
            created_at=now,
            updated_at=now
        )
    elif task_result.state == 'FAILURE':
        info = task_result.info or {}
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.FAILED,
            progress=0,
            message=info.get('message', 'Processing failed'),
            error=str(task_result.info) if task_result.info else 'Unknown error',
            created_at=now,
            updated_at=now
        )
    else:
        # Unknown state
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            progress=0,
            message=f"Unknown state: {task_result.state}",
            created_at=now,
            updated_at=now
        )
