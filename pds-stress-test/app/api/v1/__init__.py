"""API v1 router aggregation."""

from fastapi import APIRouter

from app.api.v1.router import runs_router, pipeline_router
from app.api.v1.generator import router as generator_router
from app.api.v1.advanced import advanced_router

router = APIRouter()

# Include all v1 routers
router.include_router(runs_router)
router.include_router(pipeline_router)
router.include_router(generator_router)
router.include_router(advanced_router)

