"""Async wrapper around Redis (cache / session state)."""

from __future__ import annotations

from typing import cast

import redis.asyncio as aioredis

from kuvox_ai.config import Settings
from kuvox_ai.logging import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Connection holder for Redis."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._client: aioredis.Redis | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> RedisClient:
        return cls(url=settings.redis_url)

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("RedisClient is not connected")
        return self._client

    async def connect(self) -> None:
        logger.info("redis.connecting", url=self._url)
        self._client = cast(
            aioredis.Redis,
            aioredis.from_url(self._url, decode_responses=True),  # type: ignore[no-untyped-call]
        )
        logger.info("redis.connected")

    async def close(self) -> None:
        if self._client is None:
            return
        logger.info("redis.closing")
        await self._client.aclose()
        self._client = None
        logger.info("redis.closed")

    async def health_check(self) -> bool:
        try:
            if self._client is None:
                return False
            pong = await self._client.ping()
            return bool(pong)
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis.health_check_failed", error=str(exc))
            return False
