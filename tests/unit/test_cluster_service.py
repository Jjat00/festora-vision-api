"""Unit tests for DBSCAN clustering service."""

from __future__ import annotations

import numpy as np
import pytest

from app.schemas.cluster import ClusterRequest
from app.services.cluster_service import run_clustering


def _unit_vectors(n: int, dim: int = 512) -> list[list[float]]:
    """Generate `n` random L2-normalised vectors."""
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).tolist()


def test_two_identical_images_same_cluster() -> None:
    vec = _unit_vectors(1)[0]
    request = ClusterRequest(
        embeddings=[vec, vec],
        image_ids=["img_a", "img_b"],
        eps=0.35,
        min_samples=2,
    )
    response = run_clustering(request)

    assert response.num_clusters == 1
    assert response.num_noise == 0
    assert response.assignments[0].cluster_id == response.assignments[1].cluster_id


def test_orthogonal_vectors_are_noise() -> None:
    """Two completely orthogonal vectors have cosine distance 1 — noise at eps=0.35."""
    v1 = [1.0] + [0.0] * 511
    v2 = [0.0, 1.0] + [0.0] * 510
    request = ClusterRequest(
        embeddings=[v1, v2],
        image_ids=["img_a", "img_b"],
        eps=0.35,
        min_samples=2,
    )
    response = run_clustering(request)

    assert response.num_clusters == 0
    assert response.num_noise == 2


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="length"):
        ClusterRequest(
            embeddings=_unit_vectors(3),
            image_ids=["a", "b"],   # wrong length
        )


def test_response_assignments_preserve_order() -> None:
    ids = [f"img_{i}" for i in range(5)]
    vecs = _unit_vectors(5)
    request = ClusterRequest(embeddings=vecs, image_ids=ids)
    response = run_clustering(request)

    assert [a.image_id for a in response.assignments] == ids
