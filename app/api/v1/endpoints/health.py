"""
Health check endpoints.

GET /health     — root-level health (no auth required, used by Docker/k8s probes)
GET /v1/health  — versioned alias with richer model status
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.registry import get_registry
from app.schemas.health import HealthResponse, ModelStatus

router = APIRouter(tags=["Health"])

# Recorded at import time — a rough approximation of process start.
_START_TIME = time.time()


def _build_health_response() -> HealthResponse:
    settings = get_settings()
    registry = get_registry()

    model_statuses = [
        ModelStatus(
            name=m.name,
            loaded=m.loaded,
            load_time_ms=m.load_time_ms,
            error=m.error,
        )
        for m in registry.all_models()
    ]

    return HealthResponse(
        status=registry.overall_status(),
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(time.time() - _START_TIME, 1),
        models=model_statuses,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns the overall service status and the load status of each ML model. "
        "Does not require authentication. Suitable for Docker HEALTHCHECK and "
        "Kubernetes liveness/readiness probes."
    ),
    tags=["Health"],
)
async def health_root() -> HealthResponse:
    return _build_health_response()


@router.get(
    "/v1/health",
    response_model=HealthResponse,
    summary="Service health check (versioned alias)",
    include_in_schema=True,
    tags=["Health"],
)
async def health_v1() -> HealthResponse:
    return _build_health_response()
