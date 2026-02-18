"""
Structured error responses and global exception handlers.

Every error returned by the API follows this envelope:

    {
        "error": "snake_case_code",
        "message": "Human-readable description.",
        "detail": { ... }   // optional, only in development mode
    }
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Canonical error envelope
# --------------------------------------------------------------------------- #

def error_response(
    code: str,
    message: str,
    status_code: int,
    detail: Any = None,
) -> JSONResponse:
    body: dict[str, Any] = {"error": code, "message": message}
    if detail is not None:
        body["detail"] = detail
    return JSONResponse(status_code=status_code, content=body)


# --------------------------------------------------------------------------- #
# Custom exception classes
# --------------------------------------------------------------------------- #

class VisionAPIError(Exception):
    """Base exception for all domain errors raised inside services."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ImageFetchError(VisionAPIError):
    def __init__(self, message: str) -> None:
        super().__init__("image_fetch_error", message, status.HTTP_422_UNPROCESSABLE_ENTITY)


class ImageDecodeError(VisionAPIError):
    def __init__(self, message: str) -> None:
        super().__init__("image_decode_error", message, status.HTTP_422_UNPROCESSABLE_ENTITY)


class ImageTooLargeError(VisionAPIError):
    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        super().__init__(
            "image_too_large",
            f"Image is {size_bytes:,} bytes; maximum allowed is {max_bytes:,} bytes.",
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )


class ModelNotReadyError(VisionAPIError):
    def __init__(self, model_name: str) -> None:
        super().__init__(
            "model_not_ready",
            f"Model '{model_name}' is not loaded. Check /health for model status.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class BatchTooLargeError(VisionAPIError):
    def __init__(self, received: int, max_allowed: int) -> None:
        super().__init__(
            "batch_too_large",
            f"Batch contains {received} images; maximum allowed is {max_allowed}.",
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )


class ClusteringError(VisionAPIError):
    def __init__(self, message: str) -> None:
        super().__init__("clustering_error", message, status.HTTP_422_UNPROCESSABLE_ENTITY)


# --------------------------------------------------------------------------- #
# FastAPI exception handlers — register via register_exception_handlers()
# --------------------------------------------------------------------------- #

def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(VisionAPIError)
    async def vision_api_error_handler(
        request: Request, exc: VisionAPIError
    ) -> JSONResponse:
        logger.warning("VisionAPIError [%s]: %s", exc.code, exc.message)
        return error_response(exc.code, exc.message, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.debug("RequestValidationError: %s", exc.errors())
        return error_response(
            code="validation_error",
            message="Request body or query parameters failed validation.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(),
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        return error_response(
            code="validation_error",
            message="Internal data validation error.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return error_response(
            code="internal_error",
            message="An unexpected error occurred. Please try again later.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
