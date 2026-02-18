"""Integration tests for GET /health."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert "status" in body
        assert "version" in body
        assert "models" in body
        assert isinstance(body["models"], list)

    def test_health_no_auth_required(self, authed_client: TestClient) -> None:
        """Health endpoint must not require API key."""
        response = authed_client.get("/health")
        assert response.status_code == 200

    def test_versioned_health_alias(self, client: TestClient) -> None:
        response = client.get("/v1/health")
        assert response.status_code == 200
