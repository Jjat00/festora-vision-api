"""
LLM visual analysis via LiteLLM.

LiteLLM provides a unified interface for 100+ LLM providers using the same
OpenAI-compatible API call. Switching providers requires only changing the
model string:

    "anthropic/claude-opus-4-6"  →  Anthropic Claude
    "openai/gpt-4o"              →  OpenAI GPT-4o
    "gemini/gemini-1.5-pro"      →  Google Gemini

The model is prompted to return a JSON object matching LlmPhotoAnalysis.
This enables structured extraction without post-processing fragility.

Providers and their required env vars:
    Anthropic  → ANTHROPIC_API_KEY
    OpenAI     → OPENAI_API_KEY
    Gemini     → GEMINI_API_KEY (or GOOGLE_API_KEY)

See: https://docs.litellm.ai/docs/providers
"""

from __future__ import annotations

import json
import logging
import time

from app.schemas.llm_analysis import LlmPhotoAnalysis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default prompt
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT = """\
You are an expert photography analyst. Analyze this photograph and respond with \
ONLY a valid JSON object — no markdown, no explanation, just raw JSON.

The JSON must match this exact schema:
{
  "overall_score": <integer 1-10>,
  "discard_reason": <string or null>,
  "best_in_group": <boolean>,
  "composition": <string>,
  "pose_quality": <string or null>,
  "background_quality": <string>,
  "highlights": [<string>, ...],
  "issues": [<string>, ...],
  "summary": <string>
}

Guidelines:
- overall_score: 1 = reject immediately, 10 = portfolio quality
- discard_reason: fill in when eyes are closed, motion blur, severe overexposure, \
  cut-off subjects, etc. Leave null if photo is acceptable.
- best_in_group: true only if this is clearly the best of a similar series
- composition: describe the compositional technique used
- pose_quality: null for non-portrait shots (landscapes, products)
- highlights: up to 5 strong points
- issues: up to 5 problems; empty list if none
- summary: one concise sentence for the photographer

Respond with ONLY the JSON object.\
"""


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

async def analyse_with_llm(
    image_url: str,
    model: str,
    custom_prompt: str | None = None,
    max_tokens: int = 500,
) -> tuple[LlmPhotoAnalysis, str, int]:
    """
    Send an image URL to the specified LLM and parse the JSON response.

    Returns:
        (analysis, model_used, tokens_used)

    Raises:
        ValueError: if the LLM response cannot be parsed as LlmPhotoAnalysis.
        Exception:  if the LiteLLM call fails (network, auth, quota, etc.).
    """
    try:
        from litellm import acompletion  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "litellm is not installed. "
            "Add `litellm` to requirements.txt and rebuild."
        ) from exc

    prompt = custom_prompt or _DEFAULT_PROMPT
    t0 = time.monotonic()

    response = await acompletion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        temperature=0.1,   # low temperature for consistent structured output
    )

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    raw_text = response.choices[0].message.content
    model_used = response.model
    tokens_used = response.usage.total_tokens if response.usage else 0

    logger.debug(
        "LLM analysis complete | model=%s tokens=%d ms=%d",
        model_used, tokens_used, elapsed_ms,
    )

    try:
        data = json.loads(raw_text)
        analysis = LlmPhotoAnalysis(**data)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("LLM returned unparseable JSON: %s | raw=%r", exc, raw_text[:200])
        raise ValueError(f"LLM response could not be parsed: {exc}") from exc

    return analysis, model_used, tokens_used
