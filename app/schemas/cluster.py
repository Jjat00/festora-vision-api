"""
Request / response schemas for the clustering endpoint.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class ClusterRequest(BaseModel):
    """
    Request body for POST /v1/cluster

    Submit a list of pre-computed CLIP embeddings (returned by /v1/analyze
    or /v1/analyze/batch) plus their identifiers, and get back DBSCAN cluster
    assignments.

    The service has no database — embeddings must be supplied by the caller.
    """

    embeddings: Annotated[
        list[list[float]],
        Field(
            min_length=2,
            description="List of L2-normalised 512-dimensional CLIP vectors. "
                        "Must match the order of `image_ids`.",
        ),
    ]
    image_ids: Annotated[
        list[str],
        Field(
            min_length=2,
            description="Caller-supplied IDs matching the `embeddings` list. "
                        "Echoed back in the cluster assignments.",
        ),
    ]
    eps: float = Field(
        default=0.35,
        ge=0.01,
        le=2.0,
        description="DBSCAN epsilon: maximum cosine distance between two "
                    "samples for them to be considered neighbours. "
                    "Tune downward for tighter clusters.",
    )
    min_samples: int = Field(
        default=2,
        ge=1,
        le=50,
        description="DBSCAN minimum samples: minimum number of images in a "
                    "neighbourhood to form a core point. "
                    "Raise to reduce noise sensitivity.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "embeddings": [
                        [0.021, -0.014, "... 512 floats ..."],
                        [0.019, -0.012, "... 512 floats ..."],
                        [0.95, 0.31, "... 512 floats ..."],
                    ],
                    "image_ids": ["photo_001", "photo_002", "photo_003"],
                    "eps": 0.35,
                    "min_samples": 2,
                }
            ]
        }
    }

    def model_post_init(self, __context: object) -> None:
        if len(self.embeddings) != len(self.image_ids):
            raise ValueError(
                f"`embeddings` length ({len(self.embeddings)}) must equal "
                f"`image_ids` length ({len(self.image_ids)})."
            )


class ClusterAssignment(BaseModel):
    image_id: str
    cluster_id: int = Field(
        description="Integer cluster label. -1 means the image is noise "
                    "(could not be assigned to any cluster).",
    )
    is_noise: bool = Field(
        description="True when cluster_id == -1.",
    )


class ClusterSummary(BaseModel):
    cluster_id: int
    size: int = Field(description="Number of images in this cluster.")
    image_ids: list[str] = Field(description="IDs of images in this cluster.")


class ClusterResponse(BaseModel):
    """Response body for POST /v1/cluster"""

    api_version: str = Field(default="1.0")
    algorithm: str = Field(default="DBSCAN")
    eps_used: float
    min_samples_used: int
    total_images: int
    num_clusters: int = Field(description="Number of clusters found (excluding noise).")
    num_noise: int = Field(description="Number of images classified as noise.")
    assignments: list[ClusterAssignment] = Field(
        description="Per-image cluster assignment in the same order as the input."
    )
    clusters: list[ClusterSummary] = Field(
        description="Cluster summaries (noise cluster excluded)."
    )
    processing_time_ms: float
