"""Model download API endpoint"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from app.core.config import settings

router = APIRouter()


@router.get("/models/{filename}")
async def get_model(filename: str):
    """
    Download a generated 3D model or analysis file
    
    Args:
        filename: Name of the file (e.g., job_id.gltf)
        
    Returns:
        File content
    """
    # Validate filename to prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Check file extension
    allowed_extensions = ['.gltf', '.glb', '.json']
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Construct file path
    models_dir = Path(settings.UPLOAD_DIR) / "models"
    file_path = models_dir / filename
    
    # Check if file exists
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type
    media_type = "application/json"
    if filename.endswith('.gltf'):
        media_type = "model/gltf+json"
    elif filename.endswith('.glb'):
        media_type = "model/gltf-binary"
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )
