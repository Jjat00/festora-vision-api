"""
Shared pytest fixtures.

Strategy: we never load actual ML models in unit tests.
The model registry is patched with pre-built MockRegistry instances
that return canned responses so tests are fast and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings


# ------------------------------------------------------------------ #
# Mock model instances
# ------------------------------------------------------------------ #

class MockBrisque:
    def __call__(self, tensor: Any) -> Any:
        import torch
        return torch.tensor(18.5)


class MockNimaTechnical:
    def __call__(self, tensor: Any) -> Any:
        import torch
        return torch.tensor(7.1)


class MockNimaAesthetic:
    def __call__(self, tensor: Any) -> Any:
        import torch
        return torch.tensor(6.8)


class MockCLIPModel:
    def get_image_features(self, **kwargs: Any) -> Any:
        import torch
        return torch.randn(1, 512)


class MockCLIPProcessor:
    def __call__(self, images: Any, return_tensors: str = "pt") -> dict[str, Any]:
        import torch
        return {"pixel_values": torch.randn(1, 3, 224, 224)}


# ------------------------------------------------------------------ #
# Mock registry
# ------------------------------------------------------------------ #

from app.models.registry import LoadedModel, ModelRegistry  # noqa: E402


def make_mock_registry() -> ModelRegistry:
    from dataclasses import replace

    r = ModelRegistry()
    r.clip_model = LoadedModel("clip", instance=MockCLIPModel(), loaded=True, load_time_ms=10.0)
    r.clip_processor = LoadedModel("clip_processor", instance=MockCLIPProcessor(), loaded=True, load_time_ms=5.0)
    r.pyiqa_brisque = LoadedModel("pyiqa_brisque", instance=MockBrisque(), loaded=True, load_time_ms=8.0)
    r.pyiqa_nima_technical = LoadedModel("pyiqa_nima_technical", instance=MockNimaTechnical(), loaded=True, load_time_ms=9.0)
    r.pyiqa_nima_aesthetic = LoadedModel("pyiqa_nima_aesthetic", instance=MockNimaAesthetic(), loaded=True, load_time_ms=9.0)
    r.deepface_ready = LoadedModel("deepface", instance=True, loaded=True, load_time_ms=0.0)
    return r


# ------------------------------------------------------------------ #
# Test client fixture
# ------------------------------------------------------------------ #

@pytest.fixture(scope="session")
def mock_registry() -> ModelRegistry:
    return make_mock_registry()


@pytest.fixture(scope="session")
def client(mock_registry: ModelRegistry) -> TestClient:
    """
    Return a FastAPI TestClient with:
      - API_KEY authentication disabled (open mode)
      - model registry pre-populated with mocks
    """
    import app.models.registry as registry_module

    # Patch the registry singleton before the app is created.
    registry_module._registry = mock_registry

    # Clear the settings cache so env changes in tests take effect.
    get_settings.cache_clear()

    from app.main import create_app

    with patch.dict("os.environ", {"API_KEY": "", "RATE_LIMIT_ENABLED": "false"}):
        test_app = create_app()
        with TestClient(test_app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture(scope="session")
def authed_client(mock_registry: ModelRegistry) -> TestClient:
    """TestClient with API_KEY=test-secret enforced."""
    import app.models.registry as registry_module

    registry_module._registry = mock_registry
    get_settings.cache_clear()

    from app.main import create_app

    with patch.dict("os.environ", {"API_KEY": "test-secret", "RATE_LIMIT_ENABLED": "false"}):
        test_app = create_app()
        with TestClient(test_app, raise_server_exceptions=False) as c:
            yield c


# ------------------------------------------------------------------ #
# Shared test helpers
# ------------------------------------------------------------------ #

TINY_RED_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)
