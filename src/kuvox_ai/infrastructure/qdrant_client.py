"""Async wrapper around the Qdrant vector database."""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from kuvox_ai.config import Settings
from kuvox_ai.logging import get_logger

logger = get_logger(__name__)


class QdrantClient:
    """Connection holder for Qdrant. Wraps the native ``AsyncQdrantClient``."""

    def __init__(self, host: str, port: int, api_key: str | None) -> None:
        self._host = host
        self._port = port
        self._api_key = api_key
        self._client: AsyncQdrantClient | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> QdrantClient:
        return cls(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantClient is not connected")
        return self._client

    async def connect(self) -> None:
        logger.info("qdrant.connecting", host=self._host, port=self._port)
        self._client = AsyncQdrantClient(
            host=self._host,
            port=self._port,
            api_key=self._api_key,
        )
        logger.info("qdrant.connected")

    async def close(self) -> None:
        if self._client is None:
            return
        logger.info("qdrant.closing")
        await self._client.close()
        self._client = None
        logger.info("qdrant.closed")

    async def health_check(self) -> bool:
        try:
            if self._client is None:
                return False
            await self._client.get_collections()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("qdrant.health_check_failed", error=str(exc))
            return False
