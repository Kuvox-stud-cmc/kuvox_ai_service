"""Retrieval service — graph-augmented multimodal search."""

from __future__ import annotations

from kuvox_ai.infrastructure import KuzuClient, QdrantClient
from kuvox_ai.logging import get_logger
from kuvox_ai.modules.retrieval.models import RetrievalQuery
from kuvox_ai.schemas import RetrievalResult

logger = get_logger(__name__)


class RetrievalService:
    """Public interface to the retrieval pipeline.

    For each requested modality, queries the corresponding Qdrant collection,
    optionally expands the hit set through Kuzu graph traversal, then fuses
    per-modality rankings via Reciprocal Rank Fusion.
    """

    def __init__(self, *, kuzu: KuzuClient, qdrant: QdrantClient) -> None:
        self._kuzu = kuzu
        self._qdrant = qdrant

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Run the retrieval pipeline for a single query.

        TODO: implement embed → per-modality search → graph expansion → RRF.
        """
        logger.info("retrieval.retrieve.start", text=query.text, top_k=query.top_k)
        raise NotImplementedError("RetrievalService.retrieve is not implemented yet")
