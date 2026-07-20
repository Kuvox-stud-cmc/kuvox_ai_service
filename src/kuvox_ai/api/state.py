"""Application state shared across the FastAPI app.

Instances of infrastructure clients and module services hang off this object.
The lifespan handler in :mod:`kuvox_ai.main` populates it and tears it down.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from kuvox_ai.cache import CacheStore
from kuvox_ai.infrastructure import (
    KuzuClient,
    LLMClient,
    ObjectStorageClient,
    QdrantClient,
    RabbitMQClient,
    RedisClient,
)
from kuvox_ai.modules.ingestion import IngestionService
from kuvox_ai.modules.media_optimization import MediaOptimizationService
from kuvox_ai.modules.planning import PlanningService
from kuvox_ai.modules.rendering import RenderingService
from kuvox_ai.modules.retrieval import CachedVideoEditorRetrievalService
from kuvox_ai.modules.sandbox import SandboxService


@dataclass(slots=True)
class AppState:
    """Container for all process-wide singletons."""

    kuzu: KuzuClient
    qdrant: QdrantClient
    redis: RedisClient
    rabbitmq: RabbitMQClient
    storage: ObjectStorageClient
    llm: LLMClient
    cache: CacheStore

    ingestion: IngestionService
    media_optimization: MediaOptimizationService
    retrieval: CachedVideoEditorRetrievalService
    planning: PlanningService
    rendering: RenderingService
    sandbox: SandboxService


def get_state(request: Request) -> AppState:
    """FastAPI dependency that returns the active :class:`AppState`."""
    state = request.app.state.kuvox
    assert isinstance(state, AppState)
    return state
