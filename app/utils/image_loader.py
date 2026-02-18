"""
Image loading utilities.

Accepts both URL and base64 inputs; returns a numpy array (H, W, C) in BGR
order (OpenCV convention) plus PIL Image for model inference.
Enforces size limits and safe URL validation.
"""

from __future__ import annotations

import io
import logging
from typing import NamedTuple

import httpx
import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.errors import ImageDecodeError, ImageFetchError, ImageTooLargeError
from app.schemas.common import ImageInput

logger = logging.getLogger(__name__)

# Allowlist of schemes that the service will fetch from.
_ALLOWED_SCHEMES = {"http", "https"}


class LoadedImage(NamedTuple):
    pil: Image.Image            # RGB PIL image for model inference
    bgr: np.ndarray             # BGR numpy array for OpenCV
    width: int
    height: int


async def load_image(image_input: ImageInput) -> LoadedImage:
    """
    Fetch or decode the image described by `image_input`.
    Returns a LoadedImage named-tuple.
    Raises VisionAPIError subclasses on failure.
    """
    settings = get_settings()

    if image_input.url is not None:
        raw_bytes = await _fetch_url(str(image_input.url), settings)
    else:
        # base64 — already decoded by Pydantic Base64Bytes
        raw_bytes = bytes(image_input.data)  # type: ignore[arg-type]

    return _decode_bytes(raw_bytes, settings)


async def _fetch_url(url: str, settings: object) -> bytes:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ImageFetchError(f"Unsupported URL scheme '{parsed.scheme}'. Only http/https allowed.")

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=settings.max_image_fetch_timeout_seconds,  # type: ignore[attr-defined]
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.TimeoutException:
        raise ImageFetchError(f"Request to '{url}' timed out after "
                              f"{settings.max_image_fetch_timeout_seconds}s.")  # type: ignore
    except httpx.HTTPStatusError as exc:
        raise ImageFetchError(
            f"HTTP {exc.response.status_code} fetching '{url}'."
        )
    except httpx.RequestError as exc:
        raise ImageFetchError(f"Network error fetching '{url}': {exc}")

    raw = response.content
    max_bytes: int = settings.max_image_size_bytes  # type: ignore[attr-defined]
    if len(raw) > max_bytes:
        raise ImageTooLargeError(len(raw), max_bytes)

    return raw


def _decode_bytes(raw_bytes: bytes, settings: object) -> LoadedImage:
    max_bytes: int = settings.max_image_size_bytes  # type: ignore[attr-defined]
    if len(raw_bytes) > max_bytes:
        raise ImageTooLargeError(len(raw_bytes), max_bytes)

    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        width, height = pil_image.size
    except Exception as exc:
        raise ImageDecodeError(f"Could not decode image bytes: {exc}")

    # OpenCV expects BGR numpy array.
    bgr = np.array(pil_image)[:, :, ::-1].copy()

    return LoadedImage(pil=pil_image, bgr=bgr, width=width, height=height)
