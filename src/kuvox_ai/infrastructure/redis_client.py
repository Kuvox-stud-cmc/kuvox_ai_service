"""Async wrapper around Redis (cache / session state)."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import redis.asyncio as aioredis

from kuvox_ai.config import Settings
from kuvox_ai.logging import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Connection holder for Redis."""

    def __init__(
        self,
        url: str,
        *,
        enabled: bool = True,
        username: str | None = None,
        password: str | None = None,
        connect_timeout_seconds: float = 0.5,
        operation_timeout_seconds: float = 0.5,
    ) -> None:
        self._url = url
        self._enabled = enabled
        self._username = username
        self._password = password
        self._connect_timeout_seconds = connect_timeout_seconds
        self._operation_timeout_seconds = operation_timeout_seconds
        self._client: aioredis.Redis | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> RedisClient:
        return cls(
            url=settings.redis_url,
            enabled=settings.cache_enabled,
            username=settings.redis_username,
            password=settings.redis_password,
            connect_timeout_seconds=settings.redis_connect_timeout_seconds,
            operation_timeout_seconds=settings.redis_operation_timeout_seconds,
        )

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("RedisClient is not connected")
        return self._client

    async def connect(self) -> None:
        if not self._enabled:
            logger.info("redis.disabled")
            return
        logger.info("redis.configuring", endpoint=redact_redis_url(self._url))
        self._client = aioredis.from_url(
            self._url,
            username=self._username,
            password=self._password,
            decode_responses=False,
            socket_connect_timeout=self._connect_timeout_seconds,
            socket_timeout=self._operation_timeout_seconds,
            retry_on_timeout=False,
        )
        logger.info("redis.configured", endpoint=redact_redis_url(self._url))

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

    async def get(self, key: str) -> bytes | None:
        value = await self.client.get(key)
        if value is None:
            return None
        return value.encode("utf-8") if isinstance(value, str) else bytes(value)

    async def mget(self, keys: list[str]) -> list[bytes | None]:
        values = await self.client.mget(keys)
        return [
            None
            if value is None
            else value.encode("utf-8")
            if isinstance(value, str)
            else bytes(value)
            for value in values
        ]

    async def set(self, key: str, value: bytes, *, ex: int) -> bool:
        result = await self.client.set(key, value, ex=ex)
        return bool(result)

    async def set_if_absent(self, key: str, value: bytes, *, ttl_milliseconds: int) -> bool:
        result = await self.client.set(key, value, nx=True, px=ttl_milliseconds)
        return bool(result)

    async def exists(self, key: str) -> bool:
        return bool(await self.client.exists(key))

    async def eval(self, script: str, keys: list[str], args: list[bytes]) -> int:
        return int(await self.client.eval(script, len(keys), *keys, *args))

    async def delete(self, key: str) -> int:
        return int(await self.client.delete(key))

    def pipeline(self, *, transaction: bool) -> aioredis.client.Pipeline:
        return self.client.pipeline(transaction=transaction)


def redact_redis_url(url: str) -> str:
    """Return a safe endpoint string that never contains Redis credentials."""
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
