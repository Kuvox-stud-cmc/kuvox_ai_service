# `retrieval`

Graph-augmented multimodal search over previously ingested shots.

The video-editor endpoint currently searches transcript and OCR collections. Visual and
audio requests return explicit capability warnings. For each searchable modality, the
service:

1. Embeds the canonicalized query text into the text vector space.
2. Queries the corresponding Qdrant collection for nearest neighbors.
3. (Optional) Expands the hit set by traversing the Kuzu graph — e.g. shots
   sharing entities, scenes, or temporal adjacency with strong hits.

Per-modality rankings are then fused via Reciprocal Rank Fusion (RRF) into a
single `RetrievalResult` of ranked `ScoredShot`s.

Consumed synchronously by the `/retrieval` HTTP endpoint and by the planning
module's reasoning path.

Video-editor query embeddings use the shared text-embedding cache only when both
`KUVOX_CACHE_ENABLED` and `KUVOX_QUERY_EMBEDDING_CACHE_ENABLED` are true. Identical
canonical text may reuse an ingestion-populated vector. Qdrant results, response DTOs, and
graph expansion remain uncached. The HTTP contracts are unchanged. See
[`CACHING.md`](../../../../CACHING.md).
