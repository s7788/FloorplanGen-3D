"""Main FastAPI application entry point"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload, status, models
from app.core.config import settings

app = FastAPI(
    title="FloorplanGen-3D API",
    description="API for converting 2D floorplans to 3D models",
    version="0.1.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(status.router, prefix="/api/v1", tags=["status"])
app.include_router(models.router, prefix="/api/v1", tags=["models"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FloorplanGen-3D API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
