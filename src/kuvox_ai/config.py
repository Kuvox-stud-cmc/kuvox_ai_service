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
LLMProvider = Literal["groq", "openai", "claude"]
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
    queue_ingestion: str = "kuvox.ingestion"
    queue_rendering: str = "kuvox.rendering"
    queue_sandbox: str = "kuvox.sandbox"

    # --- Object storage (S3-compatible) ----------------------------------
    s3_endpoint_url: str = "http://localhost:9000"
    s3_region: str = "us-east-1"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "kuvox-media"
    s3_create_bucket: bool = True

    # --- LLM -------------------------------------------------------------
    llm_provider: LLMProvider = "openai"
    groq_llm_api_key: str | None = None
    groq_llm_model: str | None = None
    openai_llm_api_key: str | None = None
    openai_llm_model: str | None = None
    claude_llm_api_key: str | None = None
    claude_llm_model: str | None = None

    # --- Models ----------------------------------------------------------
    model_dir: Path = Field(default=Path("./data/models"))

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
