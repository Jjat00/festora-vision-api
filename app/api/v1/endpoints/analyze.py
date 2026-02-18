"""
Single-image and batch analysis endpoints.

POST /v1/analyze        — analyse one image
POST /v1/analyze/batch  — analyse up to MAX_BATCH_SIZE images concurrently
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends

from app.core.config import get_settings
from app.core.errors import BatchTooLargeError
from app.core.security import AuthDep
from app.models.registry import get_registry
from app.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
)
from app.schemas.common import AnalysisFlags
from app.services.analysis_orchestrator import analyse_single_image

router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post(
    "",
    response_model=AnalyzeResponse,
    summary="Analyse a single image",
    description=(
        "Submit a single image (URL or base64) and receive blur, quality, "
        "aesthetic, emotion, and/or embedding results depending on the flags set. "
        "The service performs no database writes — all results are returned in the response."
    ),
    responses={
        401: {"description": "Missing or invalid X-Api-Key header."},
        413: {"description": "Image exceeds the maximum allowed size."},
        422: {"description": "Validation error or image could not be fetched/decoded."},
        503: {"description": "One or more required models are not ready."},
    },
)
async def analyze_single(
    body: AnalyzeRequest,
    _auth: AuthDep,
) -> AnalyzeResponse:
    registry = get_registry()
    flags = AnalysisFlags(
        run_blur=body.run_blur,
        run_quality=body.run_quality,
        run_emotion=body.run_emotion,
        run_embedding=body.run_embedding,
    )
    result = await analyse_single_image(body.image, flags, registry)
    return AnalyzeResponse(result=result)


@router.post(
    "/batch",
    response_model=BatchAnalyzeResponse,
    summary="Analyse multiple images in one request",
    description=(
        "Submit a list of images (mix of URLs and base64 is allowed). "
        "All images are analysed concurrently. "
        "Per-image failures are isolated — a single bad image does not fail the entire batch. "
        f"Maximum batch size is configurable via the MAX_BATCH_SIZE environment variable "
        f"(default: 20)."
    ),
    responses={
        401: {"description": "Missing or invalid X-Api-Key header."},
        422: {"description": "Validation error or batch exceeds size limit."},
        503: {"description": "One or more required models are not ready."},
    },
)
async def analyze_batch(
    body: BatchAnalyzeRequest,
    _auth: AuthDep,
) -> BatchAnalyzeResponse:
    settings = get_settings()
    if len(body.images) > settings.max_batch_size:
        raise BatchTooLargeError(len(body.images), settings.max_batch_size)

    registry = get_registry()
    flags = AnalysisFlags(
        run_blur=body.run_blur,
        run_quality=body.run_quality,
        run_emotion=body.run_emotion,
        run_embedding=body.run_embedding,
    )

    t0 = time.perf_counter()

    # Fan out all images concurrently; per-image errors are captured inside
    # analyse_single_image and returned in the result's `error` field.
    tasks = [
        analyse_single_image(image_input, flags, registry)
        for image_input in body.images
    ]
    results = await asyncio.gather(*tasks)

    elapsed = (time.perf_counter() - t0) * 1000
    succeeded = sum(1 for r in results if r.error is None)

    return BatchAnalyzeResponse(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=list(results),
        total_processing_time_ms=round(elapsed, 2),
    )
