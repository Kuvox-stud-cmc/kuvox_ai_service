from __future__ import annotations

from typing import Any

from kuvox_ai.config import Settings


def settings(**values: Any) -> Settings:
    return Settings(
        _env_file=None,
        s3_access_key="test",
        s3_secret_key="test",
        **values,
    )


def test_visual_and_audio_cache_defaults_are_independently_disabled() -> None:
    configured = settings()

    assert configured.visual_embedding_cache_enabled is False
    assert configured.audio_embedding_cache_enabled is False
    assert configured.visual_embedding_cache_ttl_seconds == 86_400
    assert configured.audio_embedding_cache_ttl_seconds == 86_400
    assert configured.visual_embedding_cache_budget_bytes == 33_554_432
    assert configured.audio_embedding_cache_budget_bytes == 67_108_864


def test_old_combined_settings_are_compatibility_fallbacks() -> None:
    configured = settings(
        visual_audio_embedding_cache_enabled=True,
        visual_audio_embedding_cache_ttl_seconds=123,
    )

    assert configured.visual_embedding_cache_enabled is True
    assert configured.audio_embedding_cache_enabled is True
    assert configured.visual_embedding_cache_ttl_seconds == 123
    assert configured.audio_embedding_cache_ttl_seconds == 123


def test_new_settings_override_old_combined_fallbacks_independently() -> None:
    configured = settings(
        visual_audio_embedding_cache_enabled=True,
        visual_audio_embedding_cache_ttl_seconds=123,
        visual_embedding_cache_enabled=False,
        audio_embedding_cache_enabled=False,
        visual_embedding_cache_ttl_seconds=456,
        audio_embedding_cache_ttl_seconds=789,
    )

    assert configured.visual_embedding_cache_enabled is False
    assert configured.audio_embedding_cache_enabled is False
    assert configured.visual_embedding_cache_ttl_seconds == 456
    assert configured.audio_embedding_cache_ttl_seconds == 789
