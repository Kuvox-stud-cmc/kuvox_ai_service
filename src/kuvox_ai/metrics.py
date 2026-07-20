"""Low-cardinality Prometheus metrics shared by the API and cache foundation."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

CACHE_OPERATIONS = Counter(
    "kuvox_cache_operations_total",
    "Cache operations by stable outcome.",
    ("service", "operation", "outcome"),
)
REDIS_COMMANDS = Counter(
    "kuvox_redis_commands_total",
    "Redis commands by stable outcome.",
    ("service", "command", "outcome"),
)
REDIS_LATENCY = Histogram(
    "kuvox_redis_command_duration_seconds",
    "Redis command latency.",
    ("service", "command"),
)
CACHE_PAYLOAD_BYTES = Histogram(
    "kuvox_cache_payload_bytes",
    "Cache payload sizes.",
    ("service", "operation"),
    buckets=(64, 256, 1024, 4096, 16384, 65536, 262144, 1048576),
)
CACHE_SCHEMA_MISSES = Counter(
    "kuvox_cache_schema_misses_total",
    "Cache values rejected because their schema version is unsupported.",
    ("service",),
)
CACHE_OVERSIZED_BYPASSES = Counter(
    "kuvox_cache_oversized_bypasses_total",
    "Cache operations bypassed because the payload was oversized.",
    ("service", "operation"),
)
CACHE_CIRCUIT_STATE = Gauge(
    "kuvox_cache_circuit_state",
    "Cache circuit state: 0 closed, 1 open, 2 half-open.",
    ("service",),
)
HTTP_REQUESTS = Counter(
    "kuvox_http_requests_total",
    "HTTP requests by normalized route template.",
    ("service", "method", "route", "status"),
)
HTTP_LATENCY = Histogram(
    "kuvox_http_request_duration_seconds",
    "HTTP request latency by normalized route template.",
    ("service", "method", "route"),
)
RETRIEVAL_STAGE_CALLS = Counter(
    "kuvox_retrieval_stage_calls_total",
    "Retrieval stage calls by stable stage and outcome.",
    ("stage", "outcome"),
)
RETRIEVAL_STAGE_LATENCY = Histogram(
    "kuvox_retrieval_stage_duration_seconds",
    "Retrieval stage latency.",
    ("stage",),
)
SINGLE_FLIGHT_EVENTS = Counter(
    "kuvox_single_flight_events_total",
    "Single-flight events by service, stable component, and outcome.",
    ("service", "component", "outcome"),
)
SINGLE_FLIGHT_WAIT = Histogram(
    "kuvox_single_flight_wait_duration_seconds",
    "Time spent waiting to join single-flight work.",
    ("service", "component"),
)
SINGLE_FLIGHT_HELD_LOCKS = Gauge(
    "kuvox_single_flight_held_locks",
    "Locally held distributed single-flight locks.",
    ("service", "component"),
)
RETRIEVAL_CACHE_OPERATIONS = Counter(
    "kuvox_retrieval_cache_operations_total",
    "Video-editor retrieval cache operations by stable outcome.",
    ("outcome",),
)
RETRIEVAL_CACHE_PAYLOAD_BYTES = Histogram(
    "kuvox_retrieval_cache_payload_bytes",
    "Video-editor retrieval cache payload sizes.",
    ("direction",),
    buckets=(256, 1024, 4096, 16384, 65536, 262144, 1048576),
)
QUERY_EMBEDDING_CACHE_OPERATIONS = Counter(
    "kuvox_query_embedding_cache_operations_total",
    "Query embedding cache operations by stable outcome.",
    ("outcome",),
)
QUERY_EMBEDDING_CACHE_DURATION = Histogram(
    "kuvox_query_embedding_cache_duration_seconds",
    "Query embedding cache operation latency.",
    ("operation",),
)
QUERY_EMBEDDING_CACHE_PAYLOAD_BYTES = Histogram(
    "kuvox_query_embedding_cache_payload_bytes",
    "Query embedding cache payload sizes.",
    ("direction",),
    buckets=(512, 1024, 1536, 2048, 4096, 8192, 16384),
)
QUERY_EMBEDDING_ENCODER_INPUTS = Counter(
    "kuvox_query_embedding_encoder_inputs_total",
    "Texts sent to the query embedding encoder by cache resolution outcome.",
    ("outcome",),
)
INGESTION_TEXT_EMBEDDING_CACHE_OPERATIONS = Counter(
    "kuvox_ingestion_text_embedding_cache_operations_total",
    "Ingestion text embedding cache operations by stable outcome.",
    ("outcome",),
)
INGESTION_TEXT_EMBEDDING_CACHE_DURATION = Histogram(
    "kuvox_ingestion_text_embedding_cache_duration_seconds",
    "Ingestion text embedding cache operation latency.",
    ("operation",),
)
INGESTION_TEXT_EMBEDDING_CACHE_PAYLOAD_BYTES = Histogram(
    "kuvox_ingestion_text_embedding_cache_payload_bytes",
    "Ingestion text embedding cache payload sizes.",
    ("direction",),
    buckets=(512, 1024, 1536, 2048, 4096, 8192, 16384),
)
INGESTION_TEXT_EMBEDDING_ENCODER_INPUTS = Counter(
    "kuvox_ingestion_text_embedding_encoder_inputs_total",
    "Texts sent to the ingestion text encoder by cache resolution outcome.",
    ("outcome",),
)
VISUAL_EMBEDDING_CACHE_OPERATIONS = Counter(
    "kuvox_visual_embedding_cache_operations_total",
    "Visual embedding cache operations by stable outcome.",
    ("outcome",),
)
VISUAL_EMBEDDING_CACHE_DURATION = Histogram(
    "kuvox_visual_embedding_cache_duration_seconds",
    "Visual embedding cache operation latency.",
    ("operation",),
)
VISUAL_EMBEDDING_CACHE_PAYLOAD_BYTES = Histogram(
    "kuvox_visual_embedding_cache_payload_bytes",
    "Visual embedding cache payload sizes.",
    ("direction",),
    buckets=(1024, 1536, 2048, 2560, 4096, 8192, 16384),
)
VISUAL_EMBEDDING_ENCODER_INPUTS = Counter(
    "kuvox_visual_embedding_encoder_inputs_total",
    "Files sent to the visual encoder by cache resolution outcome.",
    ("outcome",),
)
AUDIO_EMBEDDING_CACHE_OPERATIONS = Counter(
    "kuvox_audio_embedding_cache_operations_total",
    "Audio embedding cache operations by stable outcome.",
    ("outcome",),
)
AUDIO_EMBEDDING_CACHE_DURATION = Histogram(
    "kuvox_audio_embedding_cache_duration_seconds",
    "Audio embedding cache operation latency.",
    ("operation",),
)
AUDIO_EMBEDDING_CACHE_PAYLOAD_BYTES = Histogram(
    "kuvox_audio_embedding_cache_payload_bytes",
    "Audio embedding cache payload sizes.",
    ("direction",),
    buckets=(2048, 3072, 4096, 5120, 8192, 16384, 32768),
)
AUDIO_EMBEDDING_ENCODER_INPUTS = Counter(
    "kuvox_audio_embedding_encoder_inputs_total",
    "Files sent to the audio encoder by cache resolution outcome.",
    ("outcome",),
)


def set_circuit_state(state: str) -> None:
    CACHE_CIRCUIT_STATE.labels("ai").set({"closed": 0, "open": 1, "half_open": 2}[state])
