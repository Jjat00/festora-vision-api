"""
CLIP image embedding via HuggingFace Transformers.

The returned vector is L2-normalised so cosine similarity == dot product.
This is the correct format for DBSCAN on cosine distance.

The model and processor are loaded once at startup and retrieved from the
registry on every call — no per-request model construction.
"""

from __future__ import annotations

import numpy as np
import torch

from app.core.config import get_settings
from app.core.errors import ModelNotReadyError
from app.models.registry import ModelRegistry
from app.schemas.analysis import EmbeddingResult
from app.utils.image_loader import LoadedImage


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def analyse_embedding(image: LoadedImage, registry: ModelRegistry) -> EmbeddingResult:
    if not registry.clip_model.loaded:
        raise ModelNotReadyError("clip")
    if not registry.clip_processor.loaded:
        raise ModelNotReadyError("clip_processor")

    settings = get_settings()
    processor = registry.clip_processor.instance
    model = registry.clip_model.instance

    inputs = processor(images=image.pil, return_tensors="pt")

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    vector = _l2_normalize(features.squeeze().cpu().numpy()).tolist()

    return EmbeddingResult(
        model=settings.clip_model_name,
        dimensions=len(vector),
        vector=vector,
    )
