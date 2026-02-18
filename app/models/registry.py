"""
Model registry — single source of truth for all loaded ML models.

All models are loaded exactly once at application startup via the
`lifespan` context manager in `main.py`.  Route handlers retrieve
models through `get_registry()` — no model is ever constructed inside
a request handler.

Design decisions
----------------
- Models are stored as plain attributes on a dataclass so IDEs
  provide full type-checking and autocompletion.
- `load_time_ms` per model is recorded for the /health endpoint.
- Each model is wrapped in a try/except so a single bad model
  does not prevent the service from starting (degraded mode).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    name: str
    instance: Any | None = None
    loaded: bool = False
    load_time_ms: float | None = None
    error: str | None = None


@dataclass
class ModelRegistry:
    """Container for all ML model instances."""

    clip_model: LoadedModel = field(default_factory=lambda: LoadedModel("clip"))
    clip_processor: LoadedModel = field(default_factory=lambda: LoadedModel("clip_processor"))
    pyiqa_brisque: LoadedModel = field(default_factory=lambda: LoadedModel("pyiqa_brisque"))
    pyiqa_nima_technical: LoadedModel = field(
        default_factory=lambda: LoadedModel("pyiqa_nima_technical")
    )
    pyiqa_nima_aesthetic: LoadedModel = field(
        default_factory=lambda: LoadedModel("pyiqa_nima_aesthetic")
    )
    # DeepFace is a module-level import — we store a sentinel
    deepface_ready: LoadedModel = field(default_factory=lambda: LoadedModel("deepface"))

    def all_models(self) -> list[LoadedModel]:
        return [
            self.clip_model,
            self.clip_processor,
            self.pyiqa_brisque,
            self.pyiqa_nima_technical,
            self.pyiqa_nima_aesthetic,
            self.deepface_ready,
        ]

    def overall_status(self) -> str:
        statuses = [m.loaded for m in self.all_models()]
        if all(statuses):
            return "ok"
        if any(statuses):
            return "degraded"
        return "unhealthy"


# Module-level singleton — populated during startup lifespan.
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    if _registry is None:
        raise RuntimeError(
            "ModelRegistry has not been initialised. "
            "Ensure `load_all_models()` is called inside the lifespan handler."
        )
    return _registry


def load_all_models(settings: Any) -> ModelRegistry:
    """
    Load every ML model and return a populated ModelRegistry.
    Called once from the FastAPI lifespan context manager.
    Failures are logged but do not raise — the service starts in degraded mode.
    """
    global _registry
    registry = ModelRegistry()

    # ------------------------------------------------------------------ #
    # CLIP  (HuggingFace Transformers)
    # ------------------------------------------------------------------ #
    _load_clip(registry, settings)

    # ------------------------------------------------------------------ #
    # PyIQA  (BRISQUE + NIMA)
    # ------------------------------------------------------------------ #
    _load_pyiqa(registry, settings)

    # ------------------------------------------------------------------ #
    # DeepFace
    # ------------------------------------------------------------------ #
    _load_deepface(registry)

    _registry = registry
    loaded = sum(1 for m in registry.all_models() if m.loaded)
    total = len(registry.all_models())
    logger.info("Model loading complete: %d/%d models ready", loaded, total)
    return registry


def _timed_load(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


def _load_clip(registry: ModelRegistry, settings: Any) -> None:
    try:
        from transformers import CLIPModel, CLIPProcessor  # type: ignore

        logger.info("Loading CLIP model: %s", settings.clip_model_name)
        model, ms = _timed_load(
            CLIPModel.from_pretrained,
            settings.clip_model_name,
            cache_dir=settings.models_dir,
        )
        registry.clip_model = LoadedModel(
            name="clip", instance=model, loaded=True, load_time_ms=ms
        )

        processor, ms2 = _timed_load(
            CLIPProcessor.from_pretrained,
            settings.clip_model_name,
            cache_dir=settings.models_dir,
        )
        registry.clip_processor = LoadedModel(
            name="clip_processor", instance=processor, loaded=True, load_time_ms=ms2
        )
        logger.info("CLIP loaded in %.0f ms", ms + ms2)
    except Exception as exc:
        err = str(exc)
        logger.error("Failed to load CLIP: %s", err)
        registry.clip_model = LoadedModel(name="clip", error=err)
        registry.clip_processor = LoadedModel(name="clip_processor", error=err)


def _load_pyiqa(registry: ModelRegistry, settings: Any) -> None:
    try:
        import pyiqa  # type: ignore

        logger.info("Loading PyIQA BRISQUE")
        brisque_model, ms = _timed_load(pyiqa.create_metric, settings.pyiqa_brisque_model)
        registry.pyiqa_brisque = LoadedModel(
            name="pyiqa_brisque", instance=brisque_model, loaded=True, load_time_ms=ms
        )
        logger.info("BRISQUE loaded in %.0f ms", ms)

        logger.info("Loading PyIQA NIMA (technical)")
        nima_t, ms2 = _timed_load(pyiqa.create_metric, settings.pyiqa_nima_model)
        registry.pyiqa_nima_technical = LoadedModel(
            name="pyiqa_nima_technical", instance=nima_t, loaded=True, load_time_ms=ms2
        )

        logger.info("Loading PyIQA NIMA (aesthetic)")
        nima_a, ms3 = _timed_load(pyiqa.create_metric, "nima-koniq")
        registry.pyiqa_nima_aesthetic = LoadedModel(
            name="pyiqa_nima_aesthetic", instance=nima_a, loaded=True, load_time_ms=ms3
        )
        logger.info("NIMA models loaded in %.0f ms", ms2 + ms3)
    except Exception as exc:
        err = str(exc)
        logger.error("Failed to load PyIQA models: %s", err)
        for attr in ("pyiqa_brisque", "pyiqa_nima_technical", "pyiqa_nima_aesthetic"):
            setattr(registry, attr, LoadedModel(name=attr, error=err))


def _load_deepface(registry: ModelRegistry) -> None:
    try:
        import deepface  # noqa: F401 — validate import only at startup
        logger.info("DeepFace imported successfully")
        registry.deepface_ready = LoadedModel(
            name="deepface", instance=True, loaded=True, load_time_ms=0
        )
    except Exception as exc:
        err = str(exc)
        logger.error("Failed to import DeepFace: %s", err)
        registry.deepface_ready = LoadedModel(name="deepface", error=err)
