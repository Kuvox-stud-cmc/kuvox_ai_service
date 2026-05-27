"""Thin async wrappers around external infrastructure dependencies.

Each client owns its connection lifecycle, exposes ``async`` methods, and
provides a ``health_check()`` method used by the ``/health`` endpoint.
No business logic lives here.
"""

from kuvox_ai.infrastructure.kuzu_client import KuzuClient
from kuvox_ai.infrastructure.llm_client import LLMClient, StubLLMClient, build_llm_client
from kuvox_ai.infrastructure.object_storage_client import ObjectStorageClient
from kuvox_ai.infrastructure.qdrant_client import QdrantClient
from kuvox_ai.infrastructure.rabbitmq_client import RabbitMQClient
from kuvox_ai.infrastructure.redis_client import RedisClient

__all__ = [
    "KuzuClient",
    "LLMClient",
    "ObjectStorageClient",
    "QdrantClient",
    "RabbitMQClient",
    "RedisClient",
    "StubLLMClient",
    "build_llm_client",
]
