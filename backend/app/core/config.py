"""Application configuration"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "FloorplanGen-3D"
    VERSION: str = "0.1.0"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png"]
    UPLOAD_DIR: str = "./uploads"
    
    # AWS S3
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_BUCKET_NAME: str = "floorplangen-storage"
    AWS_REGION: str = "us-east-1"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/floorplangen"
    
    # AI/ML Settings (Phase 2)
    USE_AI_PROCESSING: bool = True  # Toggle between rule-based and AI processing
    MODEL_DIR: str = "./models"  # Directory for ML model weights
    USE_GPU: bool = True  # Use GPU for inference if available
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
