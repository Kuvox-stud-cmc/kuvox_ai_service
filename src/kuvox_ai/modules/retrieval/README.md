# `retrieval`

Graph-augmented multimodal search over previously ingested shots.

For each modality enabled in the query (visual, transcript, audio, OCR), the
service:

1. Embeds the query text into the modality's vector space.
2. Queries the corresponding Qdrant collection for nearest neighbors.
3. (Optional) Expands the hit set by traversing the Kuzu graph — e.g. shots
   sharing entities, scenes, or temporal adjacency with strong hits.

Per-modality rankings are then fused via Reciprocal Rank Fusion (RRF) into a
single `RetrievalResult` of ranked `ScoredShot`s.

Consumed synchronously by the `/retrieval` HTTP endpoint and by the planning
module's reasoning path.
