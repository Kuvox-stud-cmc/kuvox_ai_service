"""Async wrapper around RabbitMQ (via aio-pika)."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import aio_pika
from aio_pika.abc import AbstractRobustChannel, AbstractRobustConnection

from kuvox_ai.config import Settings
from kuvox_ai.logging import get_logger

logger = get_logger(__name__)

MessageHandler = Callable[[aio_pika.abc.AbstractIncomingMessage], Awaitable[None]]


class RabbitMQClient:
    """Connection holder for RabbitMQ. Uses a robust (auto-reconnecting) connection."""

    def __init__(self, url: str, *, exchange_name: str = "kuvox.events") -> None:
        self._url = url
        self._exchange_name = exchange_name
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractRobustChannel | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> RabbitMQClient:
        return cls(url=settings.rabbitmq_url, exchange_name=settings.rabbitmq_exchange)

    @property
    def channel(self) -> AbstractRobustChannel:
        if self._channel is None:
            raise RuntimeError("RabbitMQClient is not connected")
        return self._channel

    async def connect(self) -> None:
        logger.info("rabbitmq.connecting")
        self._connection = await aio_pika.connect_robust(self._url)
        self._channel = await self._connection.channel()
        logger.info("rabbitmq.connected")

    async def close(self) -> None:
        if self._connection is None:
            return
        logger.info("rabbitmq.closing")
        await self._connection.close()
        self._connection = None
        self._channel = None
        logger.info("rabbitmq.closed")

    async def declare_queue(self, name: str, *, durable: bool = True) -> aio_pika.abc.AbstractQueue:
        return await self.channel.declare_queue(name, durable=durable)

    async def declare_direct_exchange(self) -> aio_pika.abc.AbstractExchange:
        return await self.channel.declare_exchange(
            self._exchange_name,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )

    async def declare_bound_queue(
        self,
        queue_name: str,
        routing_key: str,
        *,
        durable: bool = True,
    ) -> aio_pika.abc.AbstractQueue:
        exchange = await self.declare_direct_exchange()
        queue = await self.declare_queue(queue_name, durable=durable)
        await queue.bind(exchange, routing_key=routing_key)
        return queue

    async def declare_retry_topology(
        self,
        queue_name: str,
        routing_key: str,
        retry_delays_seconds: list[int],
    ) -> None:
        exchange = await self.declare_direct_exchange()
        await self.declare_bound_queue(queue_name, routing_key)
        for attempt, delay_seconds in enumerate(retry_delays_seconds, start=1):
            retry_queue_name = retry_queue(queue_name, attempt)
            retry_queue_obj = await self.channel.declare_queue(
                retry_queue_name,
                durable=True,
                arguments={
                    "x-message-ttl": delay_seconds * 1000,
                    "x-dead-letter-exchange": self._exchange_name,
                    "x-dead-letter-routing-key": routing_key,
                },
            )
            await retry_queue_obj.bind(exchange, routing_key=retry_queue_name)

        dlq = await self.channel.declare_queue(dead_letter_queue(queue_name), durable=True)
        await dlq.bind(exchange, routing_key=dead_letter_queue(queue_name))

    async def publish(
        self, queue: str, body: bytes, *, content_type: str = "application/json"
    ) -> None:
        await self.channel.default_exchange.publish(
            aio_pika.Message(body=body, content_type=content_type),
            routing_key=queue,
        )

    async def publish_json(self, routing_key: str, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        await self.publish_raw_json(routing_key, body, message_type=routing_key)

    async def publish_raw_json(
        self,
        routing_key: str,
        body: bytes,
        *,
        headers: dict[str, Any] | None = None,
        message_type: str | None = None,
    ) -> None:
        exchange = await self.declare_direct_exchange()
        await exchange.publish(
            aio_pika.Message(
                body=body,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                type=message_type or routing_key,
                headers=headers,
            ),
            routing_key=routing_key,
        )

    async def publish_retry(
        self,
        queue_name: str,
        attempt: int,
        body: bytes,
        *,
        error_code: str,
        error_message: str,
        headers: dict[str, Any] | None = None,
    ) -> None:
        await self.publish_raw_json(
            retry_queue(queue_name, attempt),
            body,
            headers=retry_headers(queue_name, attempt, error_code, error_message, headers),
        )

    async def publish_dlq(
        self,
        queue_name: str,
        body: bytes,
        *,
        error_code: str,
        error_message: str,
        headers: dict[str, Any] | None = None,
    ) -> None:
        await self.publish_raw_json(
            dead_letter_queue(queue_name),
            body,
            headers=retry_headers(
                queue_name, retry_attempt(headers), error_code, error_message, headers
            ),
        )

    async def consume(self, queue: str, handler: MessageHandler) -> None:
        """Bind ``handler`` to messages on ``queue``. Returns once consumption is set up."""
        q = await self.declare_queue(queue)
        await q.consume(handler)
        logger.info("rabbitmq.consuming", queue=queue)

    async def consume_bound_queue(
        self,
        queue_name: str,
        routing_key: str,
        handler: MessageHandler,
        *,
        prefetch_count: int | None = None,
    ) -> None:
        if prefetch_count is not None:
            await self.channel.set_qos(prefetch_count=prefetch_count)
        queue = await self.declare_bound_queue(queue_name, routing_key)
        await queue.consume(handler)
        logger.info("rabbitmq.consuming", queue=queue_name, routing_key=routing_key)

    async def health_check(self) -> bool:
        try:
            if self._connection is None:
                return False
            return not self._connection.is_closed
        except Exception as exc:  # noqa: BLE001
            logger.warning("rabbitmq.health_check_failed", error=str(exc))
            return False


def retry_queue(queue_name: str, attempt: int) -> str:
    return f"{queue_name}.retry.{attempt}"


def dead_letter_queue(queue_name: str) -> str:
    return f"{queue_name}.dlq"


def retry_attempt(headers: dict[str, Any] | None) -> int:
    if not headers:
        return 0
    value = headers.get("x-kuvox-attempt")
    if isinstance(value, int):
        return value
    if isinstance(value, bytes):
        try:
            return int(value.decode("utf-8"))
        except ValueError:
            return 0
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def retry_headers(
    queue_name: str,
    attempt: int,
    error_code: str,
    error_message: str,
    headers: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(headers or {})
    merged["x-kuvox-attempt"] = attempt
    merged["x-kuvox-error-code"] = error_code
    merged["x-kuvox-error-message"] = error_message
    merged["x-kuvox-failed-at"] = datetime.now(UTC).isoformat()
    merged["x-kuvox-source-queue"] = queue_name
    merged.setdefault("x-kuvox-event-type", "")
    merged.setdefault("x-kuvox-event-id", "")
    merged.setdefault("x-kuvox-correlation-id", "")
    return merged
