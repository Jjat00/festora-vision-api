"""
Clustering endpoint.

POST /v1/cluster — run DBSCAN on caller-supplied CLIP embeddings.

The caller is responsible for supplying embeddings (obtained from /v1/analyze
or /v1/analyze/batch). The service performs pure computation and returns
cluster assignments immediately, with no persistence.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.core.security import AuthDep
from app.schemas.cluster import ClusterRequest, ClusterResponse
from app.services.cluster_service import run_clustering

router = APIRouter(prefix="/cluster", tags=["Clustering"])


@router.post(
    "",
    response_model=ClusterResponse,
    summary="Cluster images by CLIP embedding similarity",
    description=(
        "Submit a list of L2-normalised 512D CLIP vectors (from /v1/analyze) "
        "and receive DBSCAN cluster assignments. "
        "Images assigned cluster_id=-1 are considered noise (no cluster). "
        "Tune `eps` and `min_samples` to control cluster tightness."
    ),
    responses={
        401: {"description": "Missing or invalid X-Api-Key header."},
        422: {"description": "Validation error or DBSCAN failed."},
    },
)
async def cluster_embeddings(
    body: ClusterRequest,
    _auth: AuthDep,
) -> ClusterResponse:
    # DBSCAN with sklearn is CPU-bound; offload to thread pool.
    result = await asyncio.to_thread(run_clustering, body)
    return result
