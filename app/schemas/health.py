"""
Health check response schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelStatus(BaseModel):
    name: str = Field(description="Internal model name.")
    loaded: bool = Field(description="True if the model is ready to serve requests.")
    load_time_ms: float | None = Field(
        default=None,
        description="Time taken to load this model at startup, in milliseconds.",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the model failed to load.",
    )


class HealthResponse(BaseModel):
    """Response body for GET /health and GET /v1/health"""

    status: Literal["ok", "degraded", "unhealthy"] = Field(
        description=(
            "ok        — all models loaded, service fully operational.\n"
            "degraded  — one or more optional models failed to load; "
            "            core functionality still available.\n"
            "unhealthy — critical models are unavailable."
        )
    )
    version: str = Field(description="Service version string.", examples=["1.0.0"])
    environment: str = Field(examples=["production"])
    uptime_seconds: float = Field(description="Seconds since the process started.")
    models: list[ModelStatus] = Field(
        description="Status of each ML model loaded at startup."
    )
