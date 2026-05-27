"""Shared pytest fixtures."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kuvox_ai.infrastructure import (
    KuzuClient,
    ObjectStorageClient,
    QdrantClient,
    RabbitMQClient,
    RedisClient,
    StubLLMClient,
)


@pytest.fixture
def mock_kuzu() -> AsyncMock:
    m = AsyncMock(spec=KuzuClient)
    m.health_check.return_value = True
    return m


@pytest.fixture
def mock_qdrant() -> AsyncMock:
    m = AsyncMock(spec=QdrantClient)
    m.health_check.return_value = True
    return m


@pytest.fixture
def mock_redis() -> AsyncMock:
    m = AsyncMock(spec=RedisClient)
    m.health_check.return_value = True
    return m


@pytest.fixture
def mock_rabbitmq() -> AsyncMock:
    m = AsyncMock(spec=RabbitMQClient)
    m.health_check.return_value = True
    return m


@pytest.fixture
def mock_storage() -> AsyncMock:
    m = AsyncMock(spec=ObjectStorageClient)
    m.health_check.return_value = True
    return m


@pytest.fixture
def stub_llm() -> StubLLMClient:
    return StubLLMClient()
