from __future__ import annotations

from kuvox_ai.config import Settings
from kuvox_ai.main import _build_state
from kuvox_ai.modules.ingestion.audio_embedding_cache import CachedAudioEmbeddingEncoder
from kuvox_ai.modules.ingestion.text_embedding_cache import CachedIngestionTextEmbeddingEncoder
from kuvox_ai.modules.ingestion.visual_embedding_cache import CachedVisualEmbeddingEncoder
from kuvox_ai.modules.retrieval import CachedQueryTextEmbeddingEncoder


def test_fastapi_state_wires_query_and_ingestion_to_same_shared_key_contract() -> None:
    settings = Settings(
        cache_enabled=True,
        query_embedding_cache_enabled=True,
        ingestion_text_embedding_cache_enabled=True,
        visual_embedding_cache_enabled=True,
        audio_embedding_cache_enabled=True,
        s3_access_key="test",
        s3_secret_key="test",
    )

    state = _build_state(settings)

    query = state.retrieval._text_encoder
    ingestion = state.ingestion._text_encoder
    visual = state.ingestion._visual_encoder
    audio = state.ingestion._audio_encoder
    assert isinstance(query, CachedQueryTextEmbeddingEncoder)
    assert isinstance(ingestion, CachedIngestionTextEmbeddingEncoder)
    assert isinstance(visual, CachedVisualEmbeddingEncoder)
    assert isinstance(audio, CachedAudioEmbeddingEncoder)
    assert query.key_for_text("shared text") == ingestion.key_for_text("shared text")
