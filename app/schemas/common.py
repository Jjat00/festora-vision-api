"""
Shared schema primitives reused across all endpoints.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import AnyHttpUrl, Base64Bytes, BaseModel, Field


# --------------------------------------------------------------------------- #
# Image source — either a URL or raw base64
# --------------------------------------------------------------------------- #

class ImageInputType(str, Enum):
    url = "url"
    base64 = "base64"


class ImageInput(BaseModel):
    """
    A single image to be analysed.

    Exactly one of `url` or `data` must be provided.
    `url`: publicly reachable HTTP/HTTPS URL.
    `data`: raw image bytes encoded as standard base64 (no data-URI prefix).
    """

    url: AnyHttpUrl | None = Field(
        default=None,
        description="Publicly reachable HTTP or HTTPS URL of the image.",
        examples=["https://example.com/photo.jpg"],
    )
    data: Base64Bytes | None = Field(
        default=None,
        description="Raw image bytes encoded as standard base64 (RFC 4648). "
                    "Do NOT include a data-URI prefix (`data:image/...`).",
        examples=["iVBORw0KGgoAAAANSUhEUgAA..."],
    )
    image_id: str | None = Field(
        default=None,
        description="Caller-supplied identifier echoed back in the response. "
                    "Useful for correlating batch results. Max 128 chars.",
        max_length=128,
        examples=["photo_abc123"],
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {"url": "https://cdn.example.com/portrait.jpg", "image_id": "photo_001"},
            {"data": "<base64-encoded-bytes>", "image_id": "photo_002"},
        ]
    }}

    def model_post_init(self, __context: object) -> None:
        if self.url is None and self.data is None:
            raise ValueError("Provide exactly one of `url` or `data`.")
        if self.url is not None and self.data is not None:
            raise ValueError("Provide only one of `url` or `data`, not both.")


# --------------------------------------------------------------------------- #
# Analysis flags — caller decides which analysers run
# --------------------------------------------------------------------------- #

class AnalysisFlags(BaseModel):
    """
    Boolean flags controlling which analyses are performed.
    All default to True so a minimal request runs everything.
    Set individual flags to False to skip expensive analysers.
    """

    run_blur: bool = Field(
        default=True,
        description="Run blur / sharpness detection (OpenCV Laplacian variance). "
                    "Fast — always recommended.",
    )
    run_quality: bool = Field(
        default=True,
        description="Run technical quality scoring (PyIQA BRISQUE + NIMA). "
                    "Moderate cost.",
    )
    run_emotion: bool = Field(
        default=True,
        description="Run facial emotion detection (DeepFace). "
                    "Skipped automatically when no face is detected.",
    )
    run_embedding: bool = Field(
        default=True,
        description="Generate a 512-dimensional CLIP image embedding.",
    )
