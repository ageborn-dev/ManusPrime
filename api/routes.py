# api/routes.py
from fastapi import APIRouter
from api.endpoints import router as endpoints_router

router = APIRouter()

# Include endpoints
router.include_router(endpoints_router, prefix="/api", tags=["api"])