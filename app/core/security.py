"""
API key authentication dependency.

Usage in a route:
    @router.get("/protected")
    async def protected(auth: AuthDep) -> ...:
        ...

When `API_KEY` env var is empty the dependency is a no-op so the service
works without authentication in development mode.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

_KEY_HEADER = APIKeyHeader(
    name="X-Api-Key",
    auto_error=False,      # we raise a custom error below
    description="API key for service authentication. "
                "Set the `API_KEY` environment variable on the server to enable.",
)


async def verify_api_key(
    key: Annotated[str | None, Security(_KEY_HEADER)],
) -> None:
    """
    FastAPI dependency that enforces API key authentication.

    - If `API_KEY` env var is blank: authentication is disabled, all requests pass.
    - If `API_KEY` env var is set: the incoming `X-Api-Key` header must match exactly.
    """
    settings = get_settings()

    if not settings.auth_enabled:
        # Open mode — no key required.
        return

    if not key or key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "unauthorized",
                "message": "Missing or invalid X-Api-Key header.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )


# Convenience type alias used as a FastAPI dependency.
AuthDep = Annotated[None, Depends(verify_api_key)]
