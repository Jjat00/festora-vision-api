"""
Emotion analysis via DeepFace.

DeepFace runs synchronously and is CPU-heavy. We offload it to a thread pool
via `asyncio.to_thread` in the route handler to keep the async event loop free.

Returns None gracefully when no face is detected — this is not treated as an
error, only as an expected outcome for non-portrait images.
"""

from __future__ import annotations

import logging

import numpy as np

from app.core.config import get_settings
from app.core.errors import ModelNotReadyError
from app.models.registry import ModelRegistry
from app.schemas.analysis import EmotionEntry, EmotionResult
from app.utils.image_loader import LoadedImage

logger = logging.getLogger(__name__)


def _analyse_emotion_sync(bgr: np.ndarray, detector_backend: str) -> EmotionResult | None:
    """
    Synchronous DeepFace call — always run inside asyncio.to_thread.
    Returns None if no face is found.
    """
    from deepface import DeepFace  # type: ignore

    try:
        results = DeepFace.analyze(
            img_path=bgr,
            actions=["emotion"],
            detector_backend=detector_backend,
            enforce_detection=True,    # raises if no face found
            silent=True,
        )
    except ValueError:
        # DeepFace raises ValueError when no face is detected.
        logger.debug("No face detected in image — emotion analysis skipped.")
        return None

    faces: list[EmotionEntry] = []
    for face in results:
        # DeepFace ≥ 0.0.93 adds left_eye/right_eye as (x, y) tuples to the
        # region dict. Keep only the four integer bounding-box keys.
        raw_region = face.get("region", {})
        region = {k: v for k, v in raw_region.items() if k in ("x", "y", "w", "h")}
        faces.append(
            EmotionEntry(
                dominant_emotion=face["dominant_emotion"],
                scores={k: round(v, 3) for k, v in face["emotion"].items()},
                face_confidence=round(face.get("face_confidence", 0.0), 4),
                region=region,
            )
        )

    return EmotionResult(faces_detected=len(faces), faces=faces)


async def analyse_emotion(image: LoadedImage, registry: ModelRegistry) -> EmotionResult | None:
    """Async wrapper — runs DeepFace in a thread pool executor."""
    import asyncio

    if not registry.deepface_ready.loaded:
        raise ModelNotReadyError("deepface")

    settings = get_settings()
    result = await asyncio.to_thread(
        _analyse_emotion_sync, image.bgr, settings.deepface_detector
    )
    return result
