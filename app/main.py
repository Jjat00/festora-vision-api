"""
FastAPI application entrypoint.

Startup sequence:
  1. Configure structured logging.
  2. Load all ML models (once) inside the lifespan context.
  3. Register versioned routers.
  4. Register global exception handlers.
  5. Optionally attach rate limiter.

The app is served by Uvicorn (see Dockerfile CMD).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.router import v1_router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.models.registry import load_all_models

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan handler.
    Everything before `yield` runs at startup; everything after at shutdown.
    """
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)

    logger.info(
        "Starting %s v%s [%s]",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )
    logger.info(
        "Auth: %s | Rate limiting: %s",
        "enabled" if settings.auth_enabled else "disabled (open mode)",
        "enabled" if settings.rate_limit_enabled else "disabled",
    )

    # Load ML models — this is the longest part of startup (~30–120s on CPU).
    load_all_models(settings)

    logger.info("Service ready.")
    yield

    logger.info("Shutting down %s.", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="festora-vision-api",
        description=(
            "An independent, reusable image analysis service.\n\n"
            "Provides blur detection, technical quality scoring, aesthetic scoring, "
            "emotion analysis, CLIP embeddings, batch analysis, and DBSCAN clustering.\n\n"
            "**Authentication**: Pass your API key in the `X-Api-Key` header. "
            "Authentication is disabled when the `API_KEY` environment variable is unset.\n\n"
            "**Open source**: https://github.com/Jjat00/festora-vision-api"
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "festora-vision-api",
            "url": "https://github.com/Jjat00/festora-vision-api",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        lifespan=lifespan,
    )

    # CORS — restrict in production via environment variable if needed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Optional rate limiting via slowapi.
    if settings.rate_limit_enabled:
        _attach_rate_limiter(app, settings)

    # Routers.
    app.include_router(health_router)   # /health (no prefix, no auth)
    app.include_router(v1_router)       # /v1/analyze, /v1/cluster

    # Global exception handlers (must come after routers).
    register_exception_handlers(app)

    return app


def _attach_rate_limiter(app: FastAPI, settings: object) -> None:
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore
        from slowapi.errors import RateLimitExceeded
        from slowapi.util import get_remote_address

        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[
                f"{settings.rate_limit_per_minute}/minute"  # type: ignore[attr-defined]
            ],
        )
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        logger.info(
            "Rate limiter: %d req/min per IP",
            settings.rate_limit_per_minute,  # type: ignore[attr-defined]
        )
    except ImportError:
        logger.warning(
            "slowapi is not installed. Rate limiting is disabled. "
            "Add `slowapi` to requirements.txt to enable it."
        )


app = create_app()
