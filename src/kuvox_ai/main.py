"""FastAPI application entry point.

The lifespan handler wires up every infrastructure client and module service
into ``app.state.kuvox``; the request-scoped ``get_state`` dependency reads
from there.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from aio_pika.abc import AbstractIncomingMessage
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kuvox_ai.api.middleware import RequestLoggingMiddleware
from kuvox_ai.api.routes import (
    admin_router,
    health_router,
    planning_router,
    retrieval_router,
)
from kuvox_ai.api.state import AppState
from kuvox_ai.config import Settings, get_settings
from kuvox_ai.infrastructure import (
    KuzuClient,
    ObjectStorageClient,
    QdrantClient,
    RabbitMQClient,
    RedisClient,
    build_llm_client,
)
from kuvox_ai.infrastructure.rabbitmq_client import retry_attempt
from kuvox_ai.logging import configure_logging, get_logger
from kuvox_ai.modules.ingestion import IngestionService
from kuvox_ai.modules.media_optimization import MediaOptimizationService, ffmpeg
from kuvox_ai.modules.planning import PlanningService
from kuvox_ai.modules.rendering import RenderingService
from kuvox_ai.modules.retrieval import RetrievalService
from kuvox_ai.modules.sandbox import SandboxService
from kuvox_ai.workers import (
    ingestion_worker,
    media_optimization_worker,
    rendering_worker,
    sandbox_worker,
)


def _build_state(settings: Settings) -> AppState:
    """Construct (but do not connect) all clients and services."""
    kuzu = KuzuClient.from_settings(settings)
    qdrant = QdrantClient.from_settings(settings)
    redis = RedisClient.from_settings(settings)
    rabbitmq = RabbitMQClient.from_settings(settings)
    storage = ObjectStorageClient.from_settings(settings)
    llm = build_llm_client(settings)

    retrieval = RetrievalService(
        kuzu=kuzu,
        qdrant=qdrant,
        transcript_collection_name=settings.transcript_collection_name,
        ocr_collection_name=settings.ocr_collection_name,
        text_embedding_model_name=settings.text_embedding_model_name,
        text_embedding_device=settings.text_embedding_device,
        text_embedding_batch_size=settings.text_embedding_batch_size,
    )
    media_optimization = MediaOptimizationService(
        storage=storage,
        canonical_bucket=settings.s3_canonical_bucket,
        proxy_bucket=settings.s3_proxy_bucket,
        thumbnail_bucket=settings.s3_thumbnail_bucket,
        work_dir=settings.media_work_dir,
        video_canonical_crf=settings.video_canonical_crf,
        video_proxy_crf=settings.video_proxy_crf,
        video_proxy_max_width=settings.video_proxy_max_width,
        image_max_width=settings.image_max_width,
        thumbnail_width=settings.thumbnail_width,
    )
    return AppState(
        kuzu=kuzu,
        qdrant=qdrant,
        redis=redis,
        rabbitmq=rabbitmq,
        storage=storage,
        llm=llm,
        ingestion=IngestionService(
            kuzu=kuzu,
            qdrant=qdrant,
            storage=storage,
            work_dir=settings.ingestion_work_dir,
            visual_collection_name=settings.visual_collection_name,
            visual_embedding_dim=settings.visual_embedding_dim,
            transcript_collection_name=settings.transcript_collection_name,
            audio_collection_name=settings.audio_collection_name,
            ocr_collection_name=settings.ocr_collection_name,
            media_visual_collection_name=settings.media_visual_collection_name,
            media_audio_collection_name=settings.media_audio_collection_name,
            media_transcript_collection_name=settings.media_transcript_collection_name,
            media_ocr_collection_name=settings.media_ocr_collection_name,
            text_embedding_model_name=settings.text_embedding_model_name,
            text_embedding_dim=settings.text_embedding_dim,
            text_embedding_device=settings.text_embedding_device,
            text_embedding_batch_size=settings.text_embedding_batch_size,
            whisper_model_name=settings.whisper_model_name,
            whisper_device=settings.whisper_device,
            whisper_compute_type=settings.whisper_compute_type,
            audio_embedding_dim=settings.audio_embedding_dim,
            audio_embedding_device=settings.audio_embedding_device,
            audio_embedding_batch_size=settings.audio_embedding_batch_size,
            ocr_languages=settings.ocr_language_list,
            ocr_gpu=settings.ocr_gpu,
            ocr_min_confidence=settings.ocr_min_confidence,
            clip_model_name=settings.clip_model_name,
            clip_pretrained=settings.clip_pretrained,
            clip_device=settings.clip_device,
            clip_batch_size=settings.clip_batch_size,
            visual_index_timeout_seconds=settings.visual_index_timeout_seconds,
            optional_index_timeout_seconds=settings.optional_index_timeout_seconds,
        ),
        media_optimization=media_optimization,
        retrieval=retrieval,
        planning=PlanningService(llm=llm, retrieval=retrieval),
        rendering=RenderingService(storage=storage, work_dir=settings.rendering_work_dir),
        sandbox=SandboxService(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    logger = get_logger(__name__)
    logger.info("app.starting", environment=settings.environment)

    state = _build_state(settings)
    await state.kuzu.connect()
    await state.qdrant.connect()
    await state.redis.connect()
    await state.rabbitmq.connect()
    await state.storage.connect()
    await state.llm.connect()

    app.state.kuvox = state
    if settings.run_workers:
        await _start_workers(state, settings)
    else:
        logger.info("app.workers.disabled")

    logger.info("app.started")
    try:
        yield
    finally:
        logger.info("app.stopping")
        # Close in reverse order, best-effort.
        for closer in (
            state.llm.close,
            state.storage.close,
            state.rabbitmq.close,
            state.redis.close,
            state.qdrant.close,
            state.kuzu.close,
        ):
            try:
                await closer()
            except Exception as exc:  # noqa: BLE001
                logger.warning("app.close_failed", closer=closer.__qualname__, error=str(exc))
        logger.info("app.stopped")


async def _start_workers(state: AppState, settings: Settings) -> None:
    """Register RabbitMQ worker consumers in the FastAPI process."""
    logger = get_logger(__name__)
    try:
        logger.info(
            "media_optimization_worker.ffmpeg_resolved",
            ffmpeg=ffmpeg.resolve_ffmpeg_exe(),
            ffprobe=ffmpeg.resolve_ffprobe_exe(),
            work_dir=str(settings.media_work_dir),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("media_optimization_worker.ffmpeg_resolution_failed", error=str(exc))

    await state.rabbitmq.declare_retry_topology(
        settings.media_optimization_requested_queue,
        settings.media_optimization_requested_routing_key,
        settings.rabbitmq_retry_delay_list,
    )

    async def handle_media_optimization(message: AbstractIncomingMessage) -> None:
        async with message.process(requeue=True):
            headers = dict(message.headers or {})
            await media_optimization_worker.handle_message_body(
                message.body,
                service=state.media_optimization,
                rabbitmq=state.rabbitmq,
                completed_routing_key=settings.media_optimization_completed_routing_key,
                failed_routing_key=settings.media_optimization_failed_routing_key,
                queue_name=settings.media_optimization_requested_queue,
                retry_attempt=retry_attempt(headers),
                max_retry_attempts=settings.rabbitmq_retry_attempts,
                headers=headers,
            )

    await state.rabbitmq.consume_bound_queue(
        settings.media_optimization_requested_queue,
        settings.media_optimization_requested_routing_key,
        handle_media_optimization,
        prefetch_count=settings.media_optimization_concurrency,
    )

    await state.rabbitmq.declare_retry_topology(
        settings.ingestion_requested_queue,
        settings.ingestion_requested_routing_key,
        settings.rabbitmq_retry_delay_list,
    )

    async def handle_ingestion(message: AbstractIncomingMessage) -> None:
        async with message.process(requeue=True):
            headers = dict(message.headers or {})
            await ingestion_worker.handle_message_body(
                message.body,
                service=state.ingestion,
                rabbitmq=state.rabbitmq,
                completed_routing_key=settings.ingestion_completed_routing_key,
                failed_routing_key=settings.ingestion_failed_routing_key,
                queue_name=settings.ingestion_requested_queue,
                retry_attempt=retry_attempt(headers),
                max_retry_attempts=settings.rabbitmq_retry_attempts,
                headers=headers,
            )

    await state.rabbitmq.consume_bound_queue(
        settings.ingestion_requested_queue,
        settings.ingestion_requested_routing_key,
        handle_ingestion,
        prefetch_count=settings.ingestion_concurrency,
    )

    await state.rabbitmq.declare_retry_topology(
        settings.rendering_requested_queue,
        settings.rendering_requested_routing_key,
        settings.rabbitmq_retry_delay_list,
    )

    async def handle_rendering(message: AbstractIncomingMessage) -> None:
        async with message.process(requeue=True):
            headers = dict(message.headers or {})
            await rendering_worker.handle_message_body(
                message.body,
                service=state.rendering,
                rabbitmq=state.rabbitmq,
                started_routing_key=settings.rendering_started_routing_key,
                completed_routing_key=settings.rendering_completed_routing_key,
                failed_routing_key=settings.rendering_failed_routing_key,
                queue_name=settings.rendering_requested_queue,
                retry_attempt=retry_attempt(headers),
                max_retry_attempts=settings.rabbitmq_retry_attempts,
                headers=headers,
            )

    await state.rabbitmq.consume_bound_queue(
        settings.rendering_requested_queue,
        settings.rendering_requested_routing_key,
        handle_rendering,
        prefetch_count=settings.rendering_concurrency,
    )

    await state.rabbitmq.consume(settings.queue_sandbox, sandbox_worker.handle_message)
    logger.info("app.workers.started")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Kuvox AI",
        version="0.1.0",
        description="Graph-augmented retrieval and planning for intelligent video editing.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(health_router)
    app.include_router(retrieval_router)
    app.include_router(planning_router)
    app.include_router(admin_router)
    return app


app = create_app()


def run() -> None:
    """Console-script entry point."""
    settings = get_settings()
    uvicorn.run(
        "kuvox_ai.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
    )


if __name__ == "__main__":
    run()
