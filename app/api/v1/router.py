"""
Aggregate all v1 endpoint routers under the /v1 prefix.
"""

from fastapi import APIRouter

from app.api.v1.endpoints.analyze import router as analyze_router
from app.api.v1.endpoints.cluster import router as cluster_router
from app.api.v1.endpoints.llm_analyze import router as llm_analyze_router

v1_router = APIRouter(prefix="/v1")
v1_router.include_router(analyze_router)
v1_router.include_router(cluster_router)
v1_router.include_router(llm_analyze_router)
