"""
Blur / sharpness detection using OpenCV Laplacian variance.

The Laplacian operator approximates the second derivative of an image.
High-frequency edges produce large variance; blurry images lose those
edges and produce near-zero variance.

Threshold: 100 — empirically good for typical DSLR / smartphone photos.
Photographers can tune this with the BLUR_THRESHOLD env var in the future.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.schemas.analysis import BlurResult
from app.utils.image_loader import LoadedImage

BLUR_THRESHOLD = 100.0


def analyse_blur(image: LoadedImage) -> BlurResult:
    """
    Compute Laplacian variance on the greyscale image.
    Pure CPU operation, no ML model required.
    """
    grey = cv2.cvtColor(image.bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var: float = float(cv2.Laplacian(grey, cv2.CV_64F).var())

    return BlurResult(
        laplacian_variance=round(laplacian_var, 4),
        is_blurry=laplacian_var < BLUR_THRESHOLD,
        blur_threshold=BLUR_THRESHOLD,
    )
