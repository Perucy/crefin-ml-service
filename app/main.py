"""
    Main FastAPI appliication
    file contains our API endpoints
"""

from fastapi import FastAPI
from app.core.config import settings

app = FastAPI(
    title="Crefin ML Service",
    description="ML-powered client intelligence",
    version=settings.service_version,
)

@app.get('/')
async def root():
    """
        Root endpoint  - basic health check
        Returns service info.
    """
    return {
        "service_name": settings.service_name,
        "version": settings.service_version,
        "status": "healthy",
        "message": "Crefin ML Service is up and running!"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": settings.service_name,
        "version": settings.service_version,
        "environment": settings.environment
    }