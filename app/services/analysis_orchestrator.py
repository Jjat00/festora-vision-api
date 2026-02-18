"""
Analysis orchestrator — coordinates all individual services for a single image.

Responsibilities:
  1. Load the image (URL or base64).
  2. Run only the requested analysers (based on AnalysisFlags).
  3. Collect results into an ImageAnalysisResult.
  4. Surface per-image errors without crashing the whole batch.

All CPU/GPU-bound work (quality, embedding) runs synchronously inside this
coroutine. DeepFace is the only analyser that explicitly uses to_thread
because it can hold the GIL for several seconds.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.models.registry import ModelRegistry
from app.schemas.analysis import ImageAnalysisResult
from app.schemas.common import AnalysisFlags, ImageInput
from app.services.blur_service import analyse_blur
from app.services.embedding_service import analyse_embedding
from app.services.emotion_service import analyse_emotion
from app.services.quality_service import analyse_aesthetic, analyse_quality
from app.utils.image_loader import load_image

logger = logging.getLogger(__name__)


async def analyse_single_image(
    image_input: ImageInput,
    flags: AnalysisFlags,
    registry: ModelRegistry,
) -> ImageAnalysisResult:
    """
    Run the requested analyses on one image.
    Returns an ImageAnalysisResult; the `error` field is set on failure
    so callers (batch handler) can continue with remaining images.
    """
    t0 = time.perf_counter()
    source_url = str(image_input.url) if image_input.url else None

    try:
        image = await load_image(image_input)
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.warning("Image load failed for id=%s: %s", image_input.image_id, exc)
        return ImageAnalysisResult(
            image_id=image_input.image_id,
            source_url=source_url,
            processing_time_ms=round(elapsed, 2),
            error=f"{type(exc).__name__}: {exc}",
        )

    blur_result = None
    quality_result = None
    aesthetic_result = None
    emotion_result = None
    embedding_result = None
    error: str | None = None

    try:
        # Blur — pure OpenCV, always fast.
        if flags.run_blur:
            blur_result = analyse_blur(image)

        # Quality + aesthetic share the same NIMA pass; run them together.
        if flags.run_quality:
            quality_result = await asyncio.to_thread(
                analyse_quality, image, registry
            )
            aesthetic_result = await asyncio.to_thread(
                analyse_aesthetic, image, registry
            )

        # Emotion — thread-pooled because DeepFace can hold the GIL.
        if flags.run_emotion:
            emotion_result = await analyse_emotion(image, registry)

        # Embedding — CLIP inference.
        if flags.run_embedding:
            embedding_result = await asyncio.to_thread(
                analyse_embedding, image, registry
            )

    except Exception as exc:
        logger.exception(
            "Analysis error for image id=%s: %s", image_input.image_id, exc
        )
        error = f"{type(exc).__name__}: {exc}"

    elapsed = (time.perf_counter() - t0) * 1000

    return ImageAnalysisResult(
        image_id=image_input.image_id,
        source_url=source_url,
        width_px=image.width,
        height_px=image.height,
        blur=blur_result,
        quality=quality_result,
        aesthetic=aesthetic_result,
        emotion=emotion_result,
        embedding=embedding_result,
        processing_time_ms=round(elapsed, 2),
        error=error,
    )
