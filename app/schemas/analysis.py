"""
Request / response schemas for single-image and batch analysis endpoints.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.schemas.common import AnalysisFlags, ImageInput


# --------------------------------------------------------------------------- #
# Sub-result schemas (one per analyser)
# --------------------------------------------------------------------------- #

class BlurResult(BaseModel):
    laplacian_variance: float = Field(
        description="Laplacian variance of the image. "
                    "Higher = sharper. Typical sharp images: > 100.",
        examples=[312.5],
    )
    is_blurry: bool = Field(
        description="True when `laplacian_variance` is below the blur threshold (100).",
        examples=[False],
    )
    blur_threshold: float = Field(
        default=100.0,
        description="Threshold used for the `is_blurry` classification.",
    )


class QualityResult(BaseModel):
    brisque_score: float = Field(
        description="BRISQUE no-reference quality score (0–100). "
                    "Lower is better. < 20 is considered high quality.",
        examples=[14.3],
    )
    nima_technical_score: float = Field(
        description="NIMA technical quality score (1–10). Higher is better.",
        examples=[6.8],
    )
    overall_quality: Literal["low", "medium", "high"] = Field(
        description="Categorical quality bucket derived from BRISQUE + NIMA.",
        examples=["high"],
    )


class AestheticResult(BaseModel):
    nima_aesthetic_score: float = Field(
        description="NIMA aesthetic score (1–10). "
                    "Reflects predicted human aesthetic preference.",
        examples=[7.2],
    )
    aesthetic_label: Literal["low", "medium", "high"] = Field(
        description="Categorical aesthetic bucket: "
                    "high >= 6.5, medium >= 5.0, low < 5.0.",
        examples=["high"],
    )


class EmotionEntry(BaseModel):
    dominant_emotion: str = Field(
        description="Emotion with the highest confidence score.",
        examples=["happy"],
    )
    scores: dict[str, float] = Field(
        description="Per-emotion confidence scores (0–100) for all detected emotions.",
        examples=[{"happy": 92.1, "neutral": 5.4, "sad": 1.3, "angry": 0.8,
                   "surprise": 0.3, "fear": 0.1, "disgust": 0.0}],
    )
    face_confidence: float = Field(
        description="DeepFace face-detection confidence (0–1).",
        examples=[0.97],
    )
    region: dict[str, int] = Field(
        description="Bounding box of the detected face in pixel coordinates.",
        examples=[{"x": 120, "y": 45, "w": 210, "h": 215}],
    )


class EmotionResult(BaseModel):
    faces_detected: int = Field(
        description="Number of faces detected in the image.",
        examples=[1],
    )
    faces: list[EmotionEntry] = Field(
        description="Per-face emotion analysis results.",
    )


class EmbeddingResult(BaseModel):
    model: str = Field(
        description="HuggingFace model ID used to generate the embedding.",
        examples=["openai/clip-vit-base-patch32"],
    )
    dimensions: int = Field(
        description="Length of the embedding vector.",
        examples=[512],
    )
    vector: list[float] = Field(
        description="L2-normalised 512-dimensional CLIP embedding. "
                    "Suitable for cosine similarity and DBSCAN clustering.",
    )


# --------------------------------------------------------------------------- #
# Composite analysis result for a single image
# --------------------------------------------------------------------------- #

class ImageAnalysisResult(BaseModel):
    image_id: str | None = Field(
        default=None,
        description="Echoed back from the request `image_id` field.",
        examples=["photo_001"],
    )
    source_url: str | None = Field(
        default=None,
        description="Echoed back when the image was submitted as a URL.",
        examples=["https://cdn.example.com/portrait.jpg"],
    )
    width_px: int | None = Field(default=None, description="Image width in pixels.")
    height_px: int | None = Field(default=None, description="Image height in pixels.")
    blur: BlurResult | None = Field(
        default=None,
        description="Blur / sharpness result. Null when `run_blur=false`.",
    )
    quality: QualityResult | None = Field(
        default=None,
        description="Technical quality result. Null when `run_quality=false`.",
    )
    aesthetic: AestheticResult | None = Field(
        default=None,
        description="Aesthetic scoring result. Null when `run_quality=false`. "
                    "Shares the NIMA model with `quality` at no extra cost.",
    )
    emotion: EmotionResult | None = Field(
        default=None,
        description="Emotion analysis result. Null when `run_emotion=false` "
                    "or when no face is detected.",
    )
    embedding: EmbeddingResult | None = Field(
        default=None,
        description="CLIP embedding result. Null when `run_embedding=false`.",
    )
    processing_time_ms: float = Field(
        description="Wall-clock time in milliseconds to analyse this image.",
        examples=[843.2],
    )
    error: str | None = Field(
        default=None,
        description="Non-null only when this image failed to process. "
                    "Other images in the batch are unaffected.",
        examples=["image_fetch_error: Connection timed out."],
    )


# --------------------------------------------------------------------------- #
# Single-image endpoint
# --------------------------------------------------------------------------- #

class AnalyzeRequest(AnalysisFlags, BaseModel):
    """Request body for POST /v1/analyze"""

    image: ImageInput

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image": {
                        "url": "https://cdn.example.com/portrait.jpg",
                        "image_id": "photo_001",
                    },
                    "run_blur": True,
                    "run_quality": True,
                    "run_emotion": True,
                    "run_embedding": True,
                }
            ]
        }
    }


class AnalyzeResponse(BaseModel):
    """Response body for POST /v1/analyze"""

    api_version: str = Field(default="1.0", description="API version string.")
    result: ImageAnalysisResult


# --------------------------------------------------------------------------- #
# Batch endpoint
# --------------------------------------------------------------------------- #

class BatchAnalyzeRequest(AnalysisFlags, BaseModel):
    """Request body for POST /v1/analyze/batch"""

    images: Annotated[
        list[ImageInput],
        Field(
            min_length=1,
            description="List of images to analyse. Maximum 20 per request "
                        "(configurable via MAX_BATCH_SIZE env var).",
        ),
    ]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "images": [
                        {"url": "https://cdn.example.com/photo1.jpg", "image_id": "p1"},
                        {"url": "https://cdn.example.com/photo2.jpg", "image_id": "p2"},
                    ],
                    "run_blur": True,
                    "run_quality": True,
                    "run_emotion": False,
                    "run_embedding": True,
                }
            ]
        }
    }


class BatchAnalyzeResponse(BaseModel):
    """Response body for POST /v1/analyze/batch"""

    api_version: str = Field(default="1.0")
    total: int = Field(description="Total number of images submitted.")
    succeeded: int = Field(description="Number of images successfully analysed.")
    failed: int = Field(description="Number of images that failed.")
    results: list[ImageAnalysisResult]
    total_processing_time_ms: float = Field(
        description="Total wall-clock time for the entire batch."
    )
