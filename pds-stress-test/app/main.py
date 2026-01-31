"""
FastAPI Application Entry Point

Main application setup and configuration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import router as api_v1_router
from app.config import settings

# Create FastAPI application
app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description=(
        "Production-grade pre-implementation policy stress-testing system. "
        "Simulates multiple plausible futures under uncertainty for the PDS "
        "biometric authentication policy."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_v1_router, prefix=settings.api_v1_prefix)


@app.get("/")
def root() -> dict:
    """Root endpoint - API information."""
    return {
        "name": settings.project_name,
        "version": settings.version,
        "status": "operational",
        "docs": "/docs",
        "api_v1": settings.api_v1_prefix,
    }


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.env == "development",
    )
