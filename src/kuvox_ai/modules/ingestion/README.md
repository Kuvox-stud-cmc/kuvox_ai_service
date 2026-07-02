# `ingestion`

## Current MVP 1 scope

This section supersedes the older tiered-ingestion notes below.

The worker consumes `ingestion.requested` from the `kuvox.events` direct
exchange, downloads only the canonical video object, runs FFprobe metadata and
PySceneDetect shot detection, falls back to one full-length shot when needed,
writes `Video`/`Shot` nodes plus `BELONGS_TO`/`NEXT` relationships to Kuzu,
and publishes `ingestion.completed` or `ingestion.failed`.

Qdrant, CLIP, Whisper, OCR, audio features, scenes, and graph algorithms are
future slices and are intentionally not part of MVP 1.

Responsible for turning an uploaded source video into the indexed artifacts
the rest of the pipeline consumes.

Work is split into three tiers so cheap signals are available immediately:

- **Tier 0** — shot boundary detection, thumbnails, file metadata.
- **Tier 1** — visual embeddings (CLIP) and speech transcription + embeddings
  (Whisper).
- **Tier 2** — audio embeddings (CLAP), OCR over rendered frames, structured
  feature extraction (faces, objects, scene labels).

Outputs land in three stores: graph data (videos, shots, entities, edges) in
Kuzu, vectors in Qdrant (one collection per modality), and binary artifacts
(thumbnails, intermediate media) in object storage.

The module is triggered by messages on the `ingestion.requested` RabbitMQ queue;
HTTP callers should not invoke it directly. The worker entry point lives in
`kuvox_ai/workers/ingestion_worker.py`.
