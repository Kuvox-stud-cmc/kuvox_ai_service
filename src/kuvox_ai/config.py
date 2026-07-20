"""Application configuration.

All settings are loaded from environment variables (prefix ``KUVOX_``) and/or a
``.env`` file. See ``.env.example`` for the full list with comments.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
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
    run_workers: bool = True
    metrics_enabled: bool = True

    # --- Kuzu ------------------------------------------------------------
    kuzu_db_path: Path = Path("./data/kuzu")

    # --- Qdrant ----------------------------------------------------------
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_https: bool = False

    # --- Redis -----------------------------------------------------------
    redis_url: str = "redis://localhost:6380/0"
    redis_username: str | None = None
    redis_password: str | None = None
    redis_connect_timeout_seconds: float = 0.5
    redis_operation_timeout_seconds: float = 0.5
    cache_enabled: bool = False
    cache_key_prefix: str = "kuvox:v1"
    cache_ttl_jitter_percent: int = 10
    cache_max_payload_bytes: int = 1_048_576
    text_embedding_cache_ttl_seconds: int = 604_800
    text_embedding_cache_budget_bytes: int = 100_663_296
    text_embedding_cache_legacy_read_enabled: bool = True
    query_embedding_cache_enabled: bool = False
    query_embedding_single_flight_enabled: bool = False
    ingestion_text_embedding_cache_enabled: bool = False
    visual_embedding_cache_enabled: bool = False
    audio_embedding_cache_enabled: bool = False
    visual_embedding_cache_ttl_seconds: int = 86_400
    audio_embedding_cache_ttl_seconds: int = 86_400
    visual_embedding_cache_budget_bytes: int = 33_554_432
    audio_embedding_cache_budget_bytes: int = 67_108_864
    # Compatibility-only inputs for deployments that have not split Phase 3 flags yet.
    visual_audio_embedding_cache_enabled: bool = False
    visual_audio_embedding_cache_ttl_seconds: int = 86_400
    retrieval_cache_enabled: bool = False
    retrieval_cache_ttl_seconds: int = 60
    retrieval_cache_budget_bytes: int = 67_108_864
    retrieval_single_flight_enabled: bool = False
    single_flight_lock_ttl_seconds: float = 30
    single_flight_wait_seconds: float = 15
    single_flight_poll_milliseconds: int = 50

    # --- RabbitMQ --------------------------------------------------------
    rabbitmq_url: str = "amqp://kuvox:kuvox@localhost:5672/"
    rabbitmq_exchange: str = "kuvox.events"
    rabbitmq_retry_delays_seconds: str = "30,120,600"
    rabbitmq_retry_attempts: int = 3
    queue_sandbox: str = "kuvox.sandbox"
    rendering_requested_queue: str = "kuvox.rendering"
    rendering_requested_routing_key: str = "kuvox.rendering"
    rendering_started_routing_key: str = "rendering.started"
    rendering_completed_routing_key: str = "rendering.completed"
    rendering_failed_routing_key: str = "rendering.failed"
    rendering_concurrency: int = 1
    media_optimization_requested_queue: str = "media.optimization.requested"
    media_optimization_requested_routing_key: str = "media.optimization.requested"
    media_optimization_completed_routing_key: str = "media.optimization.completed"
    media_optimization_failed_routing_key: str = "media.optimization.failed"
    media_optimization_concurrency: int = 1
    ingestion_requested_queue: str = "ingestion.requested"
    ingestion_requested_routing_key: str = "ingestion.requested"
    ingestion_completed_routing_key: str = "ingestion.completed"
    ingestion_failed_routing_key: str = "ingestion.failed"
    ingestion_concurrency: int = 1

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
    s3_create_bucket: bool = False

    # --- Media optimization ----------------------------------------------
    media_work_dir: Path = Path("/tmp/kuvox-media")
    rendering_work_dir: Path = Path("/tmp/kuvox-rendering")
    media_delete_raw_after_optimization: bool = False
    video_canonical_crf: int = 28
    video_proxy_crf: int = 30
    video_proxy_max_width: int = 1280
    image_max_width: int = 1920
    thumbnail_width: int = 320

    # --- Ingestion --------------------------------------------------------
    ingestion_work_dir: Path = Path("/tmp/kuvox-ingestion")
    visual_collection_name: str = "shots_visual"
    visual_embedding_dim: int = 512
    transcript_collection_name: str = "shots_transcript"
    audio_collection_name: str = "shots_audio"
    ocr_collection_name: str = "shots_ocr"
    media_visual_collection_name: str = "media_visual"
    media_audio_collection_name: str = "media_audio"
    media_transcript_collection_name: str = "media_transcript"
    media_ocr_collection_name: str = "media_ocr"
    text_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_embedding_dim: int = 384
    text_embedding_device: str = "auto"
    text_embedding_batch_size: int = 32
    whisper_model_name: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    audio_embedding_dim: int = 1024
    audio_embedding_device: str = "auto"
    audio_embedding_batch_size: int = 16
    ocr_languages: str = "en"
    ocr_gpu: str = "auto"
    ocr_min_confidence: float = 0.3
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    clip_device: str = "auto"
    clip_batch_size: int = 16
    visual_index_timeout_seconds: int = 60
    optional_index_timeout_seconds: int = 60

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
    def ocr_language_list(self) -> list[str]:
        return [language.strip() for language in self.ocr_languages.split(",") if language.strip()]

    @property
    def rabbitmq_retry_delay_list(self) -> list[int]:
        return [
            int(delay.strip())
            for delay in self.rabbitmq_retry_delays_seconds.split(",")
            if delay.strip()
        ]

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @model_validator(mode="after")
    def normalize_optional_secrets(self) -> Settings:
        if self.qdrant_api_key is not None and not self.qdrant_api_key.strip():
            self.qdrant_api_key = None
        if self.redis_username is not None and not self.redis_username.strip():
            self.redis_username = None
        if self.redis_password is not None and not self.redis_password.strip():
            self.redis_password = None

        return self

    @model_validator(mode="after")
    def apply_visual_audio_cache_compatibility_fallbacks(self) -> Settings:
        if "visual_embedding_cache_enabled" not in self.model_fields_set:
            self.visual_embedding_cache_enabled = self.visual_audio_embedding_cache_enabled
        if "audio_embedding_cache_enabled" not in self.model_fields_set:
            self.audio_embedding_cache_enabled = self.visual_audio_embedding_cache_enabled
        if "visual_embedding_cache_ttl_seconds" not in self.model_fields_set:
            self.visual_embedding_cache_ttl_seconds = self.visual_audio_embedding_cache_ttl_seconds
        if "audio_embedding_cache_ttl_seconds" not in self.model_fields_set:
            self.audio_embedding_cache_ttl_seconds = self.visual_audio_embedding_cache_ttl_seconds
        return self

    @model_validator(mode="after")
    def validate_storage_credentials(self) -> Settings:
        has_access_key = bool(self.s3_access_key and self.s3_access_key.strip())
        has_secret_key = bool(self.s3_secret_key and self.s3_secret_key.strip())

        if not has_access_key and not has_secret_key:
            raise ValueError(
                "S3 configuration error: KUVOX_S3_ACCESS_KEY and KUVOX_S3_SECRET_KEY are required."
            )

        if not has_access_key:
            raise ValueError(
                "S3 configuration error: KUVOX_S3_ACCESS_KEY is required when KUVOX_S3_SECRET_KEY is set."
            )

        if not has_secret_key:
            raise ValueError(
                "S3 configuration error: KUVOX_S3_SECRET_KEY is required when KUVOX_S3_ACCESS_KEY is set."
            )

        return self

    @model_validator(mode="after")
    def validate_native_path_configuration(self) -> Settings:
        if os.name == "nt":
            return self

        path_settings = {
            "KUVOX_KUZU_DB_PATH": self.kuzu_db_path,
            "KUVOX_MEDIA_WORK_DIR": self.media_work_dir,
            "KUVOX_RENDERING_WORK_DIR": self.rendering_work_dir,
            "KUVOX_INGESTION_WORK_DIR": self.ingestion_work_dir,
            "KUVOX_MODEL_DIR": self.model_dir,
        }
        for env_name, path in path_settings.items():
            if _looks_like_windows_drive_path(path):
                raise ValueError(
                    f"{env_name} uses a Windows drive path ({path}) but this process is "
                    "not running on Windows. Use a native macOS/Linux path such as "
                    "/tmp/kuvox-media, or run the service from native Windows."
                )

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()


def _looks_like_windows_drive_path(path: Path) -> bool:
    value = str(path)
    return len(value) >= 3 and value[0].isalpha() and value[1] == ":" and value[2] in {"/", "\\"}
