"""API v1 router aggregation."""

from fastapi import APIRouter

from app.api.v1 import pipeline, runs

router = APIRouter()

# Include all v1 routers
router.include_router(runs.router)
router.include_router(pipeline.router)

