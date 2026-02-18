"""
Technical quality and aesthetic scoring via PyIQA.

Models used:
  - BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator):
      No-reference metric. Scores 0–100; lower = better quality.
      Measures natural scene statistics — compression artefacts, noise.

  - NIMA (Neural Image Assessment, technical variant):
      Predicts perceptual quality on a 1–10 scale; higher = better.

  - NIMA (aesthetic / KonIQ-trained):
      Predicts aesthetic appeal on a 1–10 scale; higher = better.

All three models are loaded once in the registry and reused across requests.
"""

from __future__ import annotations

import io

import torch
from PIL import Image

from app.core.errors import ModelNotReadyError
from app.models.registry import ModelRegistry
from app.schemas.analysis import AestheticResult, QualityResult
from app.utils.image_loader import LoadedImage


def _pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a [1, C, H, W] float tensor in [0, 1]."""
    import torchvision.transforms.functional as F  # type: ignore

    return F.to_tensor(pil_image).unsqueeze(0)


def _quality_bucket(brisque: float, nima_technical: float) -> str:
    """
    Derive a categorical quality label.
    BRISQUE low AND NIMA high → 'high'.
    """
    if brisque < 25 and nima_technical >= 6.0:
        return "high"
    if brisque < 50 and nima_technical >= 4.5:
        return "medium"
    return "low"


def _aesthetic_label(nima_score: float) -> str:
    # nima_score is already scaled to [0, 10] (nima-koniq raw × 10).
    if nima_score >= 6.5:
        return "high"
    if nima_score >= 5.0:
        return "medium"
    return "low"


def analyse_quality(image: LoadedImage, registry: ModelRegistry) -> QualityResult:
    if not registry.pyiqa_brisque.loaded:
        raise ModelNotReadyError("pyiqa_brisque")
    if not registry.pyiqa_nima_technical.loaded:
        raise ModelNotReadyError("pyiqa_nima_technical")

    tensor = _pil_to_tensor(image.pil)

    with torch.no_grad():
        brisque_raw = registry.pyiqa_brisque.instance(tensor)
        nima_t_raw = registry.pyiqa_nima_technical.instance(tensor)

    brisque = float(brisque_raw.item())
    nima_t = float(nima_t_raw.item())

    return QualityResult(
        brisque_score=round(brisque, 3),
        nima_technical_score=round(nima_t, 3),
        overall_quality=_quality_bucket(brisque, nima_t),
    )


def analyse_aesthetic(image: LoadedImage, registry: ModelRegistry) -> AestheticResult:
    if not registry.pyiqa_nima_aesthetic.loaded:
        raise ModelNotReadyError("pyiqa_nima_aesthetic")

    tensor = _pil_to_tensor(image.pil)

    with torch.no_grad():
        nima_a_raw = registry.pyiqa_nima_aesthetic.instance(tensor)

    # nima-koniq outputs scores in [0, 1]; multiply by 10 to match the
    # conventional NIMA [0, 10] scale used throughout the API and the DB schema.
    nima_a = float(nima_a_raw.item()) * 10

    return AestheticResult(
        nima_aesthetic_score=round(nima_a, 3),
        aesthetic_label=_aesthetic_label(nima_a),
    )
