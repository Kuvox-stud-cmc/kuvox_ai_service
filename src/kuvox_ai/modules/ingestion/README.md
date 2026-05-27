# `ingestion`

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

The module is triggered by messages on the `kuvox.ingestion` RabbitMQ queue;
HTTP callers should not invoke it directly. The worker entry point lives in
`kuvox_ai/workers/ingestion_worker.py`.
