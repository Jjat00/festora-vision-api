"""Unit tests for blur / sharpness detection."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.services.blur_service import BLUR_THRESHOLD, analyse_blur
from app.utils.image_loader import LoadedImage


def _make_loaded_image(arr: np.ndarray) -> LoadedImage:
    pil = Image.fromarray(arr[:, :, ::-1])  # BGR -> RGB
    return LoadedImage(pil=pil, bgr=arr, width=arr.shape[1], height=arr.shape[0])


def test_sharp_image_not_blurry() -> None:
    """A high-contrast checkerboard is detected as sharp."""
    checker = np.zeros((64, 64, 3), dtype=np.uint8)
    checker[::8, :] = 255
    checker[:, ::8] = 255
    img = _make_loaded_image(checker)

    result = analyse_blur(img)

    assert result.is_blurry is False
    assert result.laplacian_variance > BLUR_THRESHOLD


def test_flat_image_is_blurry() -> None:
    """A uniform grey image has near-zero variance — blurry."""
    grey = np.full((64, 64, 3), 128, dtype=np.uint8)
    img = _make_loaded_image(grey)

    result = analyse_blur(img)

    assert result.is_blurry is True
    assert result.laplacian_variance < BLUR_THRESHOLD


def test_result_fields_are_present() -> None:
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    result = analyse_blur(_make_loaded_image(arr))

    assert result.blur_threshold == BLUR_THRESHOLD
    assert isinstance(result.laplacian_variance, float)
    assert isinstance(result.is_blurry, bool)
