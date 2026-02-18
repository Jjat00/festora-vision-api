"""
POST /v1/analyze/llm

Sends an image to a vision-capable LLM (via LiteLLM) and returns a structured
analysis covering composition, pose quality, issues, highlights, and an
overall score.

This endpoint complements the local ML analysis (blur, NIMA, DeepFace) with
natural-language reasoning. It is intentionally separate from /v1/analyze to
allow callers to control costs — only invoke this on photos that already
passed the local quality filter.

Supported LLM providers (set the corresponding API key env var):
  Anthropic  → ANTHROPIC_API_KEY  (default: anthropic/claude-opus-4-6)
  OpenAI     → OPENAI_API_KEY     (e.g. openai/gpt-4o)
  Gemini     → GEMINI_API_KEY     (e.g. gemini/gemini-1.5-pro)
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, status

from app.core.security import AuthDep
from app.schemas.llm_analysis import LlmAnalyzeRequest, LlmAnalyzeResponse
from app.services.llm_service import analyse_with_llm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["LLM Analysis"])


@router.post(
    "/llm",
    response_model=LlmAnalyzeResponse,
    summary="Analyse a single image with an LLM",
    description=(
        "Sends the image at `image_url` to the specified vision-capable LLM "
        "(via LiteLLM) and returns structured JSON covering composition, pose, "
        "issues, highlights, and an overall quality score.\n\n"
        "**Cost note**: this endpoint calls an external LLM API. "
        "Apply your own filtering before calling this at scale "
        "(e.g. only analyze photos with `compositeScore > 50`).\n\n"
        "**Provider keys**: set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or "
        "`GEMINI_API_KEY` in the service environment depending on the model used."
    ),
)
async def llm_analyze(body: LlmAnalyzeRequest, _auth: AuthDep) -> LlmAnalyzeResponse:
    t0 = time.monotonic()
    logger.info(
        "LLM analysis request | ref_id=%s model=%s",
        body.ref_id, body.model,
    )

    try:
        analysis, model_used, tokens_used = await analyse_with_llm(
            image_url=body.image_url,
            model=body.model,
            custom_prompt=body.custom_prompt,
            max_tokens=body.max_tokens,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"LLM response parsing failed: {exc}",
        )
    except Exception as exc:
        logger.exception("LLM provider call failed | ref_id=%s", body.ref_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM provider error: {exc}",
        )

    processing_ms = int((time.monotonic() - t0) * 1000)

    return LlmAnalyzeResponse(
        ref_id=body.ref_id,
        analysis=analysis,
        model_used=model_used,
        tokens_used=tokens_used,
        processing_ms=processing_ms,
    )
