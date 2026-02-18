"""
Central configuration loaded from environment variables.
All settings have sensible defaults so the service works out of the box
with `docker compose up` and no manual configuration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ #
    # Service identity
    # ------------------------------------------------------------------ #
    app_name: str = "festora-vision-api"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # ------------------------------------------------------------------ #
    # API authentication
    # Set API_KEY to a non-empty string to enable authentication.
    # Leave blank (default) to run in open / unauthenticated mode
    # (useful for local development and testing).
    # ------------------------------------------------------------------ #
    api_key: str = ""

    # ------------------------------------------------------------------ #
    # Rate limiting  (requires slowapi, enabled by default)
    # Set RATE_LIMIT_ENABLED=false to disable entirely.
    # ------------------------------------------------------------------ #
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60        # requests per client IP per minute
    rate_limit_burst: int = 10             # short-burst tolerance

    # ------------------------------------------------------------------ #
    # Model paths  (pre-downloaded into Docker image at build time)
    # Override these only if you mount a custom model volume.
    # ------------------------------------------------------------------ #
    models_dir: str = "/app/models_cache"

    # CLIP
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # PyIQA – quality / aesthetic models
    pyiqa_brisque_model: str = "brisque"
    pyiqa_nima_model: str = "nima"

    # DeepFace backend  (opencv | ssd | dlib | mtcnn | retinaface)
    deepface_detector: str = "opencv"

    # ------------------------------------------------------------------ #
    # Image fetching
    # ------------------------------------------------------------------ #
    max_image_fetch_timeout_seconds: int = 15
    max_image_size_bytes: int = 30 * 1024 * 1024   # 30 MB

    # ------------------------------------------------------------------ #
    # Batch limits
    # ------------------------------------------------------------------ #
    max_batch_size: int = 20

    # ------------------------------------------------------------------ #
    # Clustering
    # ------------------------------------------------------------------ #
    dbscan_eps_default: float = 0.35
    dbscan_min_samples_default: int = 2

    # ------------------------------------------------------------------ #
    # LLM visual analysis (via LiteLLM)
    # Set the API key for the provider you want to use:
    #   Anthropic → ANTHROPIC_API_KEY
    #   OpenAI    → OPENAI_API_KEY
    #   Gemini    → GEMINI_API_KEY
    # ------------------------------------------------------------------ #
    llm_default_model: str = "anthropic/claude-opus-4-6"
    llm_max_tokens: int = 500
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True          # structured JSON logs in production

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #
    @field_validator("api_key", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip() if v else ""

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings singleton.
    The cache is reset between tests via `get_settings.cache_clear()`.
    """
    return Settings()


# Convenience alias used throughout the codebase.
SettingsDep = Annotated[Settings, Field(default_factory=get_settings)]
