# api/routes.py
from fastapi import APIRouter
from api.endpoints import router as endpoints_router
from api.health import router as health_router

router = APIRouter()

# Include health endpoints
router.include_router(health_router, tags=["health"])

# Include main API endpoints
router.include_router(endpoints_router, prefix="/api", tags=["api"])
