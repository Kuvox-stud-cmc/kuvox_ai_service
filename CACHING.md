# Kuvox AI Service Caching

This document is the AI-service implementation companion to the workspace-level
[`CACHING.md`](../CACHING.md). The workspace document remains the source of truth for
cross-service sequencing, security rules, production readiness, and phase completion.
This file describes only the Redis caches and advisory coordination owned by
`kuvox_ai_service`.

## Status and root-plan alignment

All committed cache and single-flight flags remain disabled by default. Redis is an
optional accelerator; model computation, Qdrant, Kuzu, SeaweedFS, RabbitMQ, and the
ASP.NET-provided searchable scope remain authoritative.

| Root phase | AI-service result | Status |
| --- | --- | --- |
| Phase 0 | Binary-safe Redis client/store, versioned JSON envelope, TTL jitter, payload limit, fail-open circuit breaker, health, and Prometheus metrics | Complete |
| Phase 1 | Shared text-embedding cache used by video-editor queries | Complete |
| Phase 2 | Trusted, revision-scoped video-editor retrieval-result cache | Complete |
| Phases 3–5 | ASP.NET-owned business and editor document caches | No AI changes |
| Phase 6 | Transcript/OCR text, OpenCLIP visual, and MSCLAP audio ingestion embedding caches | Complete |
| Phase 7 | Advisory distributed single-flight for query embedding and retrieval recomputation; graph-neighbor cache declined | Complete |
| Phase 8 | BFF coalescing and HTTP validators | No AI changes |
| Phase 9 | Production topology and scale validation | Future work |

Some evidence filenames use the earlier service-local “Phase 2/3” terminology. The
behavior maps to root Phase 6 as shown above.

## Authoritative data flow and non-goals

The cache may remove repeated computation, but it never changes the trusted request
scope or durable state:

1. The BFF obtains project media and authoritative search revisions from ASP.NET.
2. The BFF sorts and hashes trusted `(mediaId, searchRevision)` pairs and sends the
   resulting `scopeRevision` to the AI service.
3. The AI service may reuse a query embedding or a complete retrieval result.
4. A miss, timeout, corrupt value, Redis error, circuit bypass, eviction, restart, or
   flush runs the authoritative model/Qdrant/Kuzu path.
5. Ingestion may reuse embeddings, but successful Qdrant/Kuzu writes and RabbitMQ
   completion events remain unchanged.

The service does not cache:

- Qdrant writes, Kuzu mutations, RabbitMQ messages, rendered output, media objects,
  generated plans, generated actions, dependency failures, or partial retrievals;
- blank text, failed model results, invalid vectors, empty retrieval results, or
  retrieval responses containing warnings;
- authentication/session state, bearer tokens, credentials, filenames, paths, raw
  query text, transcript text, or OCR text;
- graph neighbors separately, because retrieval-result hits already remove repeated
  graph expansion.

## Configuration and enablement

The global flag and the relevant feature flag must both be enabled. Single-flight also
requires its owning cache to be enabled.

```env
KUVOX_CACHE_ENABLED=false
KUVOX_QUERY_EMBEDDING_CACHE_ENABLED=false
KUVOX_QUERY_EMBEDDING_SINGLE_FLIGHT_ENABLED=false
KUVOX_INGESTION_TEXT_EMBEDDING_CACHE_ENABLED=false
KUVOX_VISUAL_EMBEDDING_CACHE_ENABLED=false
KUVOX_AUDIO_EMBEDDING_CACHE_ENABLED=false
KUVOX_RETRIEVAL_CACHE_ENABLED=false
KUVOX_RETRIEVAL_SINGLE_FLIGHT_ENABLED=false
```

Foundation, TTL, budget, and coordination settings are:

```env
KUVOX_REDIS_URL=redis://localhost:6380/0
KUVOX_REDIS_USERNAME=
KUVOX_REDIS_PASSWORD=
KUVOX_REDIS_CONNECT_TIMEOUT_SECONDS=0.5
KUVOX_REDIS_OPERATION_TIMEOUT_SECONDS=0.5
KUVOX_CACHE_KEY_PREFIX=kuvox:v1
KUVOX_CACHE_TTL_JITTER_PERCENT=10
KUVOX_CACHE_MAX_PAYLOAD_BYTES=1048576

KUVOX_TEXT_EMBEDDING_CACHE_TTL_SECONDS=604800
KUVOX_TEXT_EMBEDDING_CACHE_BUDGET_BYTES=100663296
KUVOX_TEXT_EMBEDDING_CACHE_LEGACY_READ_ENABLED=true
KUVOX_VISUAL_EMBEDDING_CACHE_TTL_SECONDS=86400
KUVOX_VISUAL_EMBEDDING_CACHE_BUDGET_BYTES=33554432
KUVOX_AUDIO_EMBEDDING_CACHE_TTL_SECONDS=86400
KUVOX_AUDIO_EMBEDDING_CACHE_BUDGET_BYTES=67108864
KUVOX_RETRIEVAL_CACHE_TTL_SECONDS=60
KUVOX_RETRIEVAL_CACHE_BUDGET_BYTES=67108864

KUVOX_SINGLE_FLIGHT_LOCK_TTL_SECONDS=30
KUVOX_SINGLE_FLIGHT_WAIT_SECONDS=15
KUVOX_SINGLE_FLIGHT_POLL_MILLISECONDS=50
```

The old combined visual/audio flag and TTL remain compatibility fallbacks only when
the split settings are absent. New deployments must use the independent variables.

## Cache foundation

`RedisClient` uses a binary-safe `redis.asyncio` client. When the global flag is false,
startup does not establish a Redis connection and the disabled store records bypasses.
The enabled store provides `GET`, `MGET`, pipelined `SET`, delete, lock acquisition,
existence checks, and owner-safe Lua release.

Foundation behavior:

- operations are bounded by the configured 500 ms defaults;
- values above 1 MiB bypass independently without poisoning valid batch neighbors;
- TTLs receive configurable jitter, with a minimum effective TTL of one second;
- five consecutive command failures open the process-local circuit for ten seconds;
- one half-open command probes recovery; success closes the circuit;
- Redis exceptions become cache error/bypass outcomes, while `asyncio.CancelledError`
  and authoritative model/dependency exceptions propagate;
- readiness reports Redis as optional/degraded and never fails liveness because of it;
- logs include only redacted endpoints, stable commands, and exception type names.

Structured JSON values use this UTF-8 envelope:

```json
{
  "schema_version": 1,
  "created_at_utc": "2026-07-19T00:00:00Z",
  "payload": {}
}
```

Unknown, incomplete, malformed, or oversized envelopes are misses. Python pickle and
other executable/native serialization formats are forbidden.

## Cache catalog

| Cache | Consumers | Identity | TTL | Value |
| --- | --- | --- | ---: | --- |
| Shared text embedding | Retrieval queries; transcript and OCR ingestion | Canonical text, model ID, dimension, normalization version | 7 days | `KTEV` float32 vector |
| Visual embedding | Shot frames and standalone images | Exact file bytes, OpenCLIP/model pipeline identity, dimension | 24 hours | `KVEV` float32 vector |
| Audio embedding | Shot clips and standalone audio | Exact file bytes, MSCLAP/audio pipeline identity, dimension | 24 hours | `KAEV` float32 vector |
| Video-editor retrieval | Trusted BFF request only | Schema, retrieval config, trusted scope revision, canonical request | 60 seconds | Versioned JSON result |
| Advisory locks | Query embedding and retrieval recomputation | Component plus SHA-256 of cache key | 30 seconds | Random owner token |

### Shared text embeddings

Text canonicalization replaces CRLF and CR with LF and applies Unicode NFC. Case and
all other whitespace remain significant. Whitespace-only values call the authoritative
encoder and are never read from or written to Redis.

```text
kuvox:v1:ai:text-embedding:model:<model-id>:dim:<dimension>:norm:nfc-lf-v1:<sha256-canonical-text>
```

Enabled batches group identical canonical nonblank inputs, issue one `MGET`, encode
each unresolved unique text once, write valid values through one non-transactional
pipeline, and expand vectors back to input order. Query and ingestion consumers share
the namespace intentionally.

`KTEV` is big-endian:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 4 | Magic `KTEV` |
| 4 | 1 | Schema version `1` |
| 5 | 4 | Unsigned dimension |
| 9 | 32 | SHA-256 of UTF-8 model ID |
| 41 | 32 | SHA-256 of UTF-8 `nfc-lf-v1` |
| 73 | `4 × dimension` | IEEE-754 float32 values |

Every hit validates magic, schema, exact byte length, model identity, normalization
identity, dimension, and numeric finiteness. Invalid cached values recompute and repair.
Wrong-dimension, non-finite, or float32-overflow authoritative vectors are returned to
the caller but are not cached.

Earlier builds wrote query values under `ai:query-embedding` with `KQEV` magic. A shared
miss may validate and promote a legacy entry when
`KUVOX_TEXT_EMBEDDING_CACHE_LEGACY_READ_ENABLED=true`. New code never writes legacy
keys. Disable legacy reads only after every legacy writer has been stopped for longer
than its maximum jittered TTL; allow old keys to expire naturally.

### Visual and audio embeddings

File-cache reuse is based on exact bytes, not semantic equivalence. Files are streamed
through SHA-256 in 1 MiB chunks with at most four concurrent hashes. Size,
device/inode, nanosecond mtime, and existence are checked before and after hashing and
encoding. An unstable hash is retried once. If input changes during encoding, the
authoritative vector is returned but not cached.

```text
kuvox:v1:ai:visual-embedding:identity:<sha256-model-pipeline-identity>:dim:<dimension>:content:<sha256-file-bytes>
kuvox:v1:ai:audio-embedding:identity:<sha256-model-pipeline-identity>:dim:<dimension>:content:<sha256-file-bytes>
```

Visual identity includes the OpenCLIP model, pretrained weights, configured dimension,
installed model/runtime library versions, RGB/OpenCLIP preprocessing, normalization,
frame-sampling contract, and standalone-image exact-file contract.

Audio identity includes MSCLAP configuration, dimension, installed model/runtime
library versions, SoundFile loading, channel averaging, model-native resampling,
repeat/truncate behavior, normalization, and FFmpeg full-audio/shot-clip extraction
contracts. Runtime device selection is excluded from both identities.

`KVEV` and `KAEV` use the same big-endian layout as `KTEV`, with their own four-byte
magic and hashes for model and pipeline identities. All hit and authoritative-result
validation rules are identical to the text-vector safety rules.

### Trusted retrieval results

Retrieval caching is allowed only when `scopeRevision` is exactly 64 lowercase/uppercase
hex characters and the trusted request has a project, nonempty media scope, nonblank
canonical query, and supported modalities.

```text
kuvox:v1:ai:retrieval:schema:video-editor-retrieval-v1:config:<sha256-config>:scope:<scope-revision>:<sha256-request>
```

The request hash binds the normalized project ID, sorted/deduplicated media IDs,
canonical query, fixed modality order, `topK`, and graph-expansion choice. The config
hash binds collection names, searchable modalities, text-embedding identity, RRF
constant, candidate limit behavior, evidence limit, graph expansion contract, and the
explicit `none-v1` reranker identity.

The JSON payload includes its configuration identity and a validated
`VideoEditorRetrievalResult`. A hit is rejected and deleted when it has the wrong
configuration, invalid DTO shape, empty results, warnings, wrong project/query, too many
results, or a media ID outside the trusted scope.

Only complete, nonempty results without dependency warnings are written. Blank queries,
empty scopes, missing/malformed revisions, Qdrant/Kuzu partial failures, requested
modality failures, and graph-expansion failures recompute without writing.

## Advisory single-flight

Lock keys are:

```text
kuvox:v1:ai:lock:<component>:<sha256-cache-key>
```

Acquisition uses `SET NX PX` and a random 128-bit owner token. Release uses a
compare-and-delete Lua script and can delete only the caller's lock. The production ACL
grants `EVAL` solely for this owner-safe release while retaining the `kuvox:v1:ai:*`
key restriction.

Single-flight is advisory:

- a leader rechecks Redis after acquiring the lock, then computes authoritative work;
- followers poll with jitter and reuse the completed cache value;
- contention timeout, lock expiry, Redis error, circuit bypass, early leader failure,
  or release failure falls back to authoritative work;
- lock loss may duplicate computation but cannot change result correctness;
- failed, partial, or empty work never becomes a negative cache entry.

There is deliberately no Redis lock around Qdrant writes, Kuzu writes, RabbitMQ delivery,
or any integrity-critical mutation.

## Security and isolation

- The AI service trusts only the BFF-supplied project/media scope revision contract;
  browser-provided unrestricted media IDs must never reach cache-key construction.
- Raw text, paths, content hashes, media IDs, project IDs, users, credentials, and owner
  tokens do not appear in logs, evidence, or metric labels.
- The production `kuvox-ai` ACL can access only `kuvox:v1:ai:*`; the default Redis user
  is disabled and dangerous administrative commands are denied.
- The AI Redis node is cache-only. Do not share it with sessions, rate limits, queues,
  SignalR, or authoritative ephemeral state without a separate capacity/failure review.
- A cache hit cannot expand the trusted media scope, fabricate a successful modality,
  suppress dependency warnings, or make Redis authoritative for search revision.

## Metrics and health

The private `/metrics` endpoint is enabled with `KUVOX_METRICS_ENABLED=true`. Foundation
families are:

```text
kuvox_cache_operations_total{service,operation,outcome}
kuvox_redis_commands_total{service,command,outcome}
kuvox_redis_command_duration_seconds{service,command}
kuvox_cache_payload_bytes{service,operation}
kuvox_cache_schema_misses_total{service}
kuvox_cache_oversized_bypasses_total{service,operation}
kuvox_cache_circuit_state{service}
kuvox_single_flight_events_total{service,component,outcome}
kuvox_single_flight_wait_duration_seconds{service,component}
kuvox_single_flight_held_locks{service,component}
```

Feature metrics cover cache outcomes, cache latency, payload bytes, and inputs sent to
each authoritative encoder for query text, ingestion text, visual, and audio. Retrieval
adds operation and payload families; Qdrant, Kuzu, and query-encoding stage metrics show
authoritative work avoided by warm hits.

Stable labels contain only service, operation, command, component, direction, stage, and
outcome. Free-form input and object identifiers are forbidden.

Health contract:

- `/health/live` is process-only;
- `/health/ready` and `/health` treat Qdrant, Kuzu, RabbitMQ, and object storage according
  to their required-dependency policy;
- Redis disabled reports `disabled`; Redis failure reports optional `degraded` and does
  not make readiness fail.

## Capacity and production topology

The selected production target is a dedicated same-host Redis 7 node named `redis-ai`:

- 384 MiB `maxmemory`, `allkeys-lfu`, AOF with `appendfsync everysec`;
- private Compose network and loopback-only host publishing;
- independent `kuvox-ai` ACL password and `kuvox:v1:ai:*` key access;
- no internal TLS for the same-host trusted network; TLS is mandatory if Redis moves
  off-host or crosses an untrusted boundary.

Initial operational budgets are:

| Domain | Budget | Representative payload | Approximate resident entries |
| --- | ---: | ---: | ---: |
| Shared text embeddings | 96 MiB | 1,609 bytes at dimension 384 | 54,000 |
| Visual embeddings | 32 MiB | 2,121 bytes at dimension 512 | 13,000 |
| Audio embeddings | 64 MiB | 4,169 bytes at dimension 1024 | 14,000 |
| Retrieval results | 64 MiB | Variable JSON, maximum 1 MiB | Measured by distribution |
| Redis overhead/headroom | 128 MiB | Allocator, keys, buffers, AOF/COW, overlap | N/A |

Budgets are operational targets rather than per-prefix hard quotas. Monitor actual
`MEMORY USAGE`, key cardinality, allocator fragmentation, evictions, command latency,
connections, persistence state, circuit state, and authoritative fallback load. Never
use blocking `KEYS` in production; use bounded `SCAN`/exporter sampling.

Root Phase 9 remains open. Before production enablement it must exercise realistic key
cardinality, memory pressure/eviction, restart/AOF recovery, credential rotation,
connection recovery, and authoritative fallback saturation. Sentinel, Cluster,
multi-region, and off-host TLS are future topology gates unless the selected deployment
changes.

## Testing and evidence

Automated coverage:

- [`tests/unit/test_cache.py`](tests/unit/test_cache.py): disabled behavior, hit/miss,
  bulk commands, TTL jitter, payload limits, corruption, timeout, and circuit behavior;
- [`tests/unit/test_query_embedding_cache.py`](tests/unit/test_query_embedding_cache.py):
  query canonicalization, grouping, migration, metrics, isolation, and failures;
- [`tests/unit/test_ingestion_text_embedding_cache.py`](tests/unit/test_ingestion_text_embedding_cache.py)
  and [`tests/unit/test_ingestion_text_cache_consumers.py`](tests/unit/test_ingestion_text_cache_consumers.py):
  transcript/OCR consumers and cross-consumer reuse;
- [`tests/unit/test_file_embedding_cache.py`](tests/unit/test_file_embedding_cache.py):
  exact-byte identity, bounded hashing, unstable files, batch grouping, and failures;
- [`tests/unit/test_retrieval_result_cache.py`](tests/unit/test_retrieval_result_cache.py)
  and [`tests/unit/test_single_flight.py`](tests/unit/test_single_flight.py): trusted
  retrieval identity, result validation, leader/follower behavior, timeout, and cleanup;
- [`tests/integration/test_redis_cache.py`](tests/integration/test_redis_cache.py): real
  Redis bytes, TTL, pipelines, locks/Lua, corruption repair, legacy promotion, binary
  schemas, and half-open recovery.

Reproducible evidence:

```bash
.venv/bin/python scripts/query_embedding_cache_baseline.py \
  --request-json /path/to/video-editor-request.json \
  --repeat 5 \
  --output /path/to/evidence/query-cache.json

.venv/bin/python scripts/text_embedding_cache_evidence.py
.venv/bin/python scripts/visual_audio_embedding_cache_evidence.py
.venv/bin/python scripts/phase7_single_flight_evidence.py
```

Redacted reports are stored in:

- [`docs/evidence/cache/phase1-phase2-local.json`](docs/evidence/cache/phase1-phase2-local.json)
  for real SentenceTransformer query/ingestion reuse, legacy promotion, memory, and
  Redis-outage equivalence;
- [`docs/evidence/cache/phase3-visual-audio-local.json`](docs/evidence/cache/phase3-visual-audio-local.json)
  for real OpenCLIP/MSCLAP cold/warm, exact-byte changes, corrupt repair, memory, and
  outage equivalence;
- [`docs/evidence/cache/phase7-single-flight-local.json`](docs/evidence/cache/phase7-single-flight-local.json)
  for 16-way single-flight, two service instances, lock expiry, leader failure, warm
  hits, and Redis-unavailable fallback.

Evidence reports contain hashes, counts, timings, metric deltas, payload/memory totals,
and redacted endpoints only. They must never contain queries, transcript/OCR contents,
paths, credentials, project/media/user identifiers, or Redis owner tokens.

## Rollout, migration, and rollback

Roll out one feature at a time with the global flag enabled only in the target
environment:

1. Query text embedding cache.
2. Query single-flight after query-cache behavior is accepted.
3. Ingestion text cache.
4. Visual cache, then audio cache after memory and model-latency review.
5. Retrieval-result cache after the BFF/ASP.NET scope-revision contract is present.
6. Retrieval single-flight only after retrieval caching is stable.

For every step, compare disabled/cold/warm/outage response equivalence, cache and Redis
errors, hit rate, model/Qdrant/Kuzu calls, p50/p95 latency, key count, payload size,
memory, evictions, and circuit state.

Rollback uses feature flags and never requires Redis availability:

- disable the affected consumer or single-flight flag;
- use `KUVOX_CACHE_ENABLED=false` to disable all AI cache and lock access;
- allow old keys and locks to expire naturally; do not flush the Redis database;
- keep authoritative model/Qdrant/Kuzu behavior active throughout rollback.

For an incompatible key/value change, introduce a new namespace such as `kuvox:v2`,
stop old writes, deploy new readers/writers, and let the old bounded namespace expire.
Do not perform a blocking all-key migration or broad dual-write. Preserve the old prefix
until application rollback is no longer required.

## Completion criteria

An AI cache change is complete only when:

- the feature remains independently disabled by default;
- keys contain all model, preprocessing, request, and trusted revision identities;
- values validate schema, identity, dimensions/DTO shape, finiteness, and size;
- Redis miss, corruption, timeout, OOM, restart, flush, and connection failure preserve
  authoritative correctness;
- cancellation and authoritative exceptions propagate;
- metrics/logs remain low-cardinality and credential/input safe;
- unit, real-Redis, failure, equivalence, and performance evidence pass;
- configuration examples, production Compose, this document, and the root plan agree;
- rollout and rollback do not require deleting Redis data.
