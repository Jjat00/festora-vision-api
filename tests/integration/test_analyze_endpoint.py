"""
Integration tests for POST /v1/analyze and POST /v1/analyze/batch.

These tests use the FastAPI TestClient with a mock model registry.
No real ML inference is performed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import TINY_RED_PNG_B64


# ------------------------------------------------------------------ #
# Single image endpoint
# ------------------------------------------------------------------ #

class TestAnalyzeSingle:
    def test_base64_image_returns_200(self, client: TestClient) -> None:
        response = client.post(
            "/v1/analyze",
            json={
                "image": {"data": TINY_RED_PNG_B64, "image_id": "test_001"},
                "run_blur": True,
                "run_quality": True,
                "run_emotion": False,
                "run_embedding": True,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["api_version"] == "1.0"
        assert body["result"]["image_id"] == "test_001"
        assert body["result"]["blur"] is not None
        assert body["result"]["emotion"] is None   # run_emotion=False

    def test_invalid_image_data_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/v1/analyze",
            json={"image": {"data": "not-valid-base64!!!"}},
        )
        assert response.status_code == 422

    def test_missing_image_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/analyze", json={"run_blur": True})
        assert response.status_code == 422

    def test_both_url_and_data_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/v1/analyze",
            json={
                "image": {
                    "url": "https://example.com/img.jpg",
                    "data": TINY_RED_PNG_B64,
                }
            },
        )
        assert response.status_code == 422


# ------------------------------------------------------------------ #
# Batch endpoint
# ------------------------------------------------------------------ #

class TestAnalyzeBatch:
    def test_batch_two_images_returns_200(self, client: TestClient) -> None:
        response = client.post(
            "/v1/analyze/batch",
            json={
                "images": [
                    {"data": TINY_RED_PNG_B64, "image_id": "p1"},
                    {"data": TINY_RED_PNG_B64, "image_id": "p2"},
                ],
                "run_blur": True,
                "run_quality": False,
                "run_emotion": False,
                "run_embedding": False,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert body["succeeded"] == 2
        assert body["failed"] == 0
        assert len(body["results"]) == 2

    def test_empty_batch_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/v1/analyze/batch",
            json={"images": []},
        )
        assert response.status_code == 422

    def test_batch_too_large_returns_422(self, client: TestClient) -> None:
        images = [{"data": TINY_RED_PNG_B64}] * 25   # exceeds default MAX_BATCH_SIZE=20
        response = client.post(
            "/v1/analyze/batch",
            json={"images": images},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "batch_too_large"


# ------------------------------------------------------------------ #
# Authentication tests
# ------------------------------------------------------------------ #

class TestAuthentication:
    def test_missing_key_returns_401(self, authed_client: TestClient) -> None:
        response = authed_client.post(
            "/v1/analyze",
            json={"image": {"data": TINY_RED_PNG_B64}},
        )
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorized"

    def test_wrong_key_returns_401(self, authed_client: TestClient) -> None:
        response = authed_client.post(
            "/v1/analyze",
            json={"image": {"data": TINY_RED_PNG_B64}},
            headers={"X-Api-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_correct_key_returns_200(self, authed_client: TestClient) -> None:
        response = authed_client.post(
            "/v1/analyze",
            json={
                "image": {"data": TINY_RED_PNG_B64},
                "run_blur": True,
                "run_quality": False,
                "run_emotion": False,
                "run_embedding": False,
            },
            headers={"X-Api-Key": "test-secret"},
        )
        assert response.status_code == 200
