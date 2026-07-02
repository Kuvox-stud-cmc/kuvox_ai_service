"""Async wrapper around RabbitMQ (via aio-pika)."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
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

    async def publish(
        self, queue: str, body: bytes, *, content_type: str = "application/json"
    ) -> None:
        await self.channel.default_exchange.publish(
            aio_pika.Message(body=body, content_type=content_type),
            routing_key=queue,
        )

    async def publish_json(self, routing_key: str, payload: dict[str, Any]) -> None:
        exchange = await self.declare_direct_exchange()
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        await exchange.publish(
            aio_pika.Message(
                body=body,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                type=routing_key,
            ),
            routing_key=routing_key,
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
