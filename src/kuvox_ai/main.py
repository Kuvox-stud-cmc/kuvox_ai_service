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
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from kuvox_ai.api.middleware import RequestLoggingMiddleware
from kuvox_ai.api.routes import (
    admin_router,
    health_router,
    planning_router,
    retrieval_router,
)
from kuvox_ai.api.state import AppState
from kuvox_ai.cache import build_cache_store
from kuvox_ai.config import Settings, get_settings
from kuvox_ai.distributed_lock import build_lock_store
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
from kuvox_ai.modules.ingestion.audio_embedding_cache import build_audio_embedding_encoder
from kuvox_ai.modules.ingestion.text_embedding_cache import (
    build_ingestion_text_embedding_encoder,
)
from kuvox_ai.modules.ingestion.text_encoder import SentenceTransformerTextEncoder
from kuvox_ai.modules.ingestion.visual_embedding_cache import build_visual_embedding_encoder
from kuvox_ai.modules.media_optimization import MediaOptimizationService, ffmpeg
from kuvox_ai.modules.planning import PlanningService
from kuvox_ai.modules.rendering import RenderingService
from kuvox_ai.modules.retrieval import (
    CachedQueryTextEmbeddingEncoder,
    CachedVideoEditorRetrievalService,
    RetrievalService,
)
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
    cache = build_cache_store(settings, redis)
    locks = build_lock_store(settings, redis)

    query_text_encoder = CachedQueryTextEmbeddingEncoder(
        SentenceTransformerTextEncoder(
            model_name=settings.text_embedding_model_name,
            device=settings.text_embedding_device,
            batch_size=settings.text_embedding_batch_size,
        ),
        cache=cache,
        enabled=settings.cache_enabled and settings.query_embedding_cache_enabled,
        model_id=settings.text_embedding_model_name,
        dimension=settings.text_embedding_dim,
        ttl_seconds=settings.text_embedding_cache_ttl_seconds,
        key_prefix=settings.cache_key_prefix,
        legacy_read_enabled=settings.text_embedding_cache_legacy_read_enabled,
        lock_store=locks,
        single_flight_enabled=(
            settings.cache_enabled and settings.query_embedding_single_flight_enabled
        ),
        lock_ttl_seconds=settings.single_flight_lock_ttl_seconds,
        lock_wait_seconds=settings.single_flight_wait_seconds,
        lock_poll_seconds=settings.single_flight_poll_milliseconds / 1000,
    )
    ingestion_text_encoder = build_ingestion_text_embedding_encoder(settings, cache)
    visual_encoder = build_visual_embedding_encoder(settings, cache)
    audio_encoder = build_audio_embedding_encoder(settings, cache)
    authoritative_retrieval = RetrievalService(
        kuzu=kuzu,
        qdrant=qdrant,
        text_encoder=query_text_encoder,
        transcript_collection_name=settings.transcript_collection_name,
        ocr_collection_name=settings.ocr_collection_name,
        text_embedding_model_name=settings.text_embedding_model_name,
        text_embedding_device=settings.text_embedding_device,
        text_embedding_batch_size=settings.text_embedding_batch_size,
    )
    retrieval = CachedVideoEditorRetrievalService(
        authoritative_retrieval,
        cache=cache,
        locks=locks,
        enabled=settings.cache_enabled and settings.retrieval_cache_enabled,
        single_flight_enabled=(
            settings.cache_enabled
            and settings.retrieval_cache_enabled
            and settings.retrieval_single_flight_enabled
        ),
        ttl_seconds=settings.retrieval_cache_ttl_seconds,
        max_payload_bytes=settings.cache_max_payload_bytes,
        key_prefix=settings.cache_key_prefix,
        transcript_collection_name=settings.transcript_collection_name,
        ocr_collection_name=settings.ocr_collection_name,
        text_embedding_model_name=settings.text_embedding_model_name,
        text_embedding_dimension=settings.text_embedding_dim,
        lock_ttl_seconds=settings.single_flight_lock_ttl_seconds,
        lock_wait_seconds=settings.single_flight_wait_seconds,
        lock_poll_seconds=settings.single_flight_poll_milliseconds / 1000,
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
        cache=cache,
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
            text_encoder=ingestion_text_encoder,
            visual_encoder=visual_encoder,
            audio_encoder=audio_encoder,
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
    semantic_dependencies_enabled = (
        settings.media_ingestion_enabled or settings.media_retrieval_enabled
    )
    if semantic_dependencies_enabled:
        await state.kuzu.connect()
        await state.qdrant.connect()
    else:
        logger.info("app.semantic_dependencies.disabled")
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
        closers = [
            state.llm.close,
            state.storage.close,
            state.rabbitmq.close,
            state.redis.close,
        ]
        if semantic_dependencies_enabled:
            closers.extend([state.qdrant.close, state.kuzu.close])
        for closer in closers:
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

    if settings.media_ingestion_enabled:
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
    else:
        logger.info("ingestion_worker.disabled")

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
    if settings.metrics_enabled:

        @app.get("/metrics", include_in_schema=False)
        async def metrics() -> Response:
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

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
