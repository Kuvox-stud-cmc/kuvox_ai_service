"""Application configuration.

All settings are loaded from environment variables (prefix ``KUVOX_``) and/or a
``.env`` file. See ``.env.example`` for the full list with comments.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LLMProvider = Literal["stub", "groq", "openai", "claude"]
# NOTE: extend the LLMProvider literal as real provider implementations land.


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Instantiated once and cached via :func:`get_settings`.
    """

    model_config = SettingsConfigDict(
        env_prefix="KUVOX_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Service ---------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    environment: Environment = "development"
    log_level: LogLevel = "INFO"
    cors_origins: str = "*"

    # --- Kuzu ------------------------------------------------------------
    kuzu_db_path: Path = Path("./data/kuzu")

    # --- Qdrant ----------------------------------------------------------
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None

    # --- Redis -----------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # --- RabbitMQ --------------------------------------------------------
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    rabbitmq_exchange: str = "kuvox.events"
    queue_ingestion: str = "kuvox.ingestion"
    queue_rendering: str = "kuvox.rendering"
    queue_sandbox: str = "kuvox.sandbox"
    media_optimization_requested_queue: str = "media.optimization.requested"
    media_optimization_requested_routing_key: str = "media.optimization.requested"
    media_optimization_completed_routing_key: str = "media.optimization.completed"
    media_optimization_failed_routing_key: str = "media.optimization.failed"
    media_optimization_concurrency: int = 1

    # --- Object storage (S3-compatible) ----------------------------------
    s3_endpoint_url: str = "http://localhost:8333"
    s3_region: str = "us-east-1"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_bucket: str = "kuvox-media"
    s3_raw_bucket: str = "kuvox-raw"
    s3_canonical_bucket: str = "kuvox-canonical"
    s3_proxy_bucket: str = "kuvox-proxy"
    s3_thumbnail_bucket: str = "kuvox-thumbnails"
    s3_temp_bucket: str = "kuvox-temp"
    s3_create_bucket: bool = True

    # --- Media optimization ----------------------------------------------
    media_work_dir: Path = Path("/tmp/kuvox-media")
    media_delete_raw_after_optimization: bool = False
    video_canonical_crf: int = 28
    video_proxy_crf: int = 30
    video_proxy_max_width: int = 1280
    image_max_width: int = 1920
    thumbnail_width: int = 320

    # --- LLM -------------------------------------------------------------
    llm_provider: LLMProvider = "stub"
    groq_llm_api_key: str | None = None
    groq_llm_model: str | None = None
    openai_llm_api_key: str | None = None
    openai_llm_model: str | None = None
    claude_llm_api_key: str | None = None
    claude_llm_model: str | None = None

    # --- Models ----------------------------------------------------------
    model_dir: Path = Field(default=Path("./data/models"))

    @property
    def llm_model(self) -> str:
        """Resolve the model name for the active provider (stub has no model)."""
        match self.llm_provider:
            case "groq":
                return self.groq_llm_model or "groq"
            case "openai":
                return self.openai_llm_model or "openai"
            case "claude":
                return self.claude_llm_model or "claude"
            case _:
                return "stub"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
