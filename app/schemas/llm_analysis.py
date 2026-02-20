"""
Pydantic schemas for the LLM visual analysis endpoint (POST /v1/analyze/llm).

The LLM is prompted to return structured JSON covering composition, pose quality,
issues, highlights, and an overall score. This endpoint complements the local ML
analysis (blur, quality, emotions) with natural-language reasoning that local
models cannot provide.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LlmPhotoAnalysis(BaseModel):
    overall_score: float = Field(
        ..., ge=1.0, le=10.0,
        description="Overall photo quality score from 1.0 (worst) to 10.0 (best), with one decimal place.",
    )
    discard_reason: str | None = Field(
        None,
        description=(
            "Short reason to discard the photo (e.g. 'eyes closed', 'motion blur'). "
            "Null if the photo is acceptable."
        ),
    )
    best_in_group: bool = Field(
        False,
        description="True if this photo stands out as the best in a similar series.",
    )
    composition: str = Field(
        ...,
        description=(
            "Detected composition type (e.g. 'rule of thirds', 'centered', "
            "'symmetrical', 'diagonal', 'candid')."
        ),
    )
    pose_quality: str | None = Field(
        None,
        description=(
            "Pose assessment when people are present "
            "(e.g. 'natural and relaxed', 'stiff', 'turned away'). "
            "Null for landscapes or product shots."
        ),
    )
    background_quality: str = Field(
        ...,
        description=(
            "Background assessment (e.g. 'clean bokeh', 'cluttered', "
            "'distracting elements', 'well lit')."
        ),
    )
    highlights: list[str] = Field(
        default_factory=list,
        description="Up to 5 strong points of the photo.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Up to 5 problems found in the photo.",
    )
    summary: str = Field(
        ...,
        description="One-sentence summary suitable for showing to the photographer.",
    )


class LlmAnalyzeRequest(BaseModel):
    image_url: str = Field(
        ...,
        description=(
            "Publicly accessible URL to the image (e.g. presigned R2/S3 URL). "
            "The image must be reachable by the LLM provider."
        ),
    )
    ref_id: str | None = Field(
        None,
        description="Opaque identifier returned as-is in the response (e.g. photo DB id).",
    )
    model: str = Field(
        "anthropic/claude-opus-4-6",
        description=(
            "LiteLLM model string. Any vision-capable model works: "
            "'anthropic/claude-opus-4-6', 'openai/gpt-4o', 'gemini/gemini-1.5-pro'."
        ),
    )
    custom_prompt: str | None = Field(
        None,
        description=(
            "Override the default Festora analysis prompt. "
            "Must instruct the model to return valid JSON matching LlmPhotoAnalysis."
        ),
    )
    max_tokens: int = Field(500, ge=100, le=2000)


class LlmAnalyzeResponse(BaseModel):
    ref_id: str | None
    analysis: LlmPhotoAnalysis
    model_used: str = Field(..., description="Actual model identifier used by the LLM provider.")
    tokens_used: int
    processing_ms: int
    error: str | None = None
