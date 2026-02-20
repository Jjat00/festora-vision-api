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
Eres un experto analista de fotografía profesional. Analiza esta fotografía y responde con \
ÚNICAMENTE un objeto JSON válido — sin markdown, sin explicaciones, solo JSON puro.

El JSON debe coincidir exactamente con este esquema:
{
  "overall_score": <decimal 1.0-10.0, un decimal>,
  "discard_reason": <string o null>,
  "best_in_group": <booleano>,
  "composition": <string>,
  "pose_quality": <string o null>,
  "background_quality": <string>,
  "highlights": [<string>, ...],
  "issues": [<string>, ...],
  "summary": <string>
}

Instrucciones:
- overall_score: número decimal con un decimal (ej: 6.3, 7.8, 9.1). 1.0 = rechazar de inmediato, 10.0 = calidad de portafolio. Usa el decimal para diferenciar fotos similares
- discard_reason: indica el motivo cuando haya ojos cerrados, desenfoque por movimiento, \
  sobreexposición severa, sujetos cortados, etc. Dejar null si la foto es aceptable.
- best_in_group: true solo si esta foto es claramente la mejor de una serie similar
- composition: describe la técnica de composición utilizada (ej. "regla de los tercios", \
  "composición centrada", "diagonal", "simétrica", "candid")
- pose_quality: null para fotos sin personas (paisajes, productos)
- highlights: hasta 5 puntos fuertes de la foto
- issues: hasta 5 problemas encontrados; lista vacía si no hay ninguno
- summary: una frase concisa dirigida al fotógrafo

Todos los valores de texto deben estar en español.
Responde con ÚNICAMENTE el objeto JSON.\
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
