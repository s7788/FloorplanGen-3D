"""Model download API endpoint"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import re
import os
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
        
    Security:
        Multiple layers of defense against path traversal:
        1. Strict regex validation (only alphanumeric, _, -, and allowed extensions)
        2. os.path.basename to strip directory components
        3. Path resolution and verification within models directory
        
        Note: CodeQL may flag this as path injection, but it's a false positive
        due to the strict validation that prevents any malicious paths.
    """
    # Strict validation: only allow alphanumeric, hyphens, underscores, and dots
    # This prevents any path traversal attempts
    if not re.match(r'^[a-zA-Z0-9_-]+\.(gltf|glb|json)$', filename):
        raise HTTPException(status_code=400, detail="Invalid filename format")
    
    # Check file extension (redundant but explicit)
    allowed_extensions = ['.gltf', '.glb', '.json']
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Construct file path - use os.path.basename to ensure no directory traversal
    # Even though filename is validated, this satisfies static analysis
    safe_filename = os.path.basename(filename)
    models_dir = Path(settings.UPLOAD_DIR) / "models"
    file_path = models_dir / safe_filename
    
    # Verify the resolved path is within the models directory
    try:
        file_path = file_path.resolve()
        models_dir_resolved = models_dir.resolve()
        file_path.relative_to(models_dir_resolved)
    except (ValueError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    # Check if file exists
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type based on validated extension
    media_type = "application/json"
    if safe_filename.endswith('.gltf'):
        media_type = "model/gltf+json"
    elif safe_filename.endswith('.glb'):
        media_type = "model/gltf-binary"
    
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=safe_filename
    )
