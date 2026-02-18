"""
DBSCAN clustering on CLIP embedding vectors.

The service accepts pre-computed embeddings from the caller — it has no
database and never fetches images itself at this endpoint.

Distance metric: cosine  (correct for L2-normalised CLIP vectors).
sklearn DBSCAN with metric='cosine' computes 1 - cosine_similarity, so
an eps of 0.35 means images whose cosine similarity >= 0.65 are neighbours.

Typical practical guidance for CLIP clusters:
    eps=0.30 → very tight clusters (nearly identical images)
    eps=0.35 → similar content / same scene
    eps=0.50 → loose thematic similarity
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN  # type: ignore

from app.core.errors import ClusteringError
from app.schemas.cluster import (
    ClusterAssignment,
    ClusterRequest,
    ClusterResponse,
    ClusterSummary,
)

logger = logging.getLogger(__name__)


def run_clustering(request: ClusterRequest) -> ClusterResponse:
    t0 = time.perf_counter()

    if len(request.embeddings) < 2:
        raise ClusteringError("At least 2 embeddings are required for clustering.")

    vectors = np.array(request.embeddings, dtype=np.float32)

    expected_dim = vectors.shape[1]
    for i, vec in enumerate(vectors):
        if len(vec) != expected_dim:
            raise ClusteringError(
                f"Embedding at index {i} has {len(vec)} dimensions; "
                f"expected {expected_dim}."
            )

    try:
        db = DBSCAN(
            eps=request.eps,
            min_samples=request.min_samples,
            metric="cosine",
            n_jobs=-1,        # use all CPU cores
        ).fit(vectors)
    except Exception as exc:
        raise ClusteringError(f"DBSCAN failed: {exc}")

    labels: list[int] = db.labels_.tolist()

    assignments: list[ClusterAssignment] = [
        ClusterAssignment(
            image_id=image_id,
            cluster_id=label,
            is_noise=(label == -1),
        )
        for image_id, label in zip(request.image_ids, labels)
    ]

    # Build cluster summaries (exclude noise).
    cluster_map: dict[int, list[str]] = defaultdict(list)
    for assignment in assignments:
        if not assignment.is_noise:
            cluster_map[assignment.cluster_id].append(assignment.image_id)

    summaries: list[ClusterSummary] = [
        ClusterSummary(cluster_id=cid, size=len(ids), image_ids=ids)
        for cid, ids in sorted(cluster_map.items())
    ]

    num_noise = sum(1 for a in assignments if a.is_noise)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return ClusterResponse(
        eps_used=request.eps,
        min_samples_used=request.min_samples,
        total_images=len(request.image_ids),
        num_clusters=len(summaries),
        num_noise=num_noise,
        assignments=assignments,
        clusters=summaries,
        processing_time_ms=round(elapsed_ms, 2),
    )
