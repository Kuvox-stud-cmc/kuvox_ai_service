"""Planning service — LangGraph-orchestrated agentic plan generation."""

from __future__ import annotations

from kuvox_ai.infrastructure import LLMClient
from kuvox_ai.logging import get_logger
from kuvox_ai.modules.planning.models import PlanningRequest
from kuvox_ai.modules.retrieval import RetrievalService
from kuvox_ai.schemas import Plan

logger = get_logger(__name__)


class PlanningService:
    """Public interface to the planning pipeline.

    Three paths are dispatched based on input classification:

    * **Direct path** — concrete commands → single LLM structured call → Plan.
    * **Reasoning path** — abstract commands → decompose → retrieve → synthesize.
    * **Code generation fallback** — out-of-schema requests → LLM emits Python
      executed in the sandbox.

    Every returned :class:`Plan` is validated against the operation schema
    before the call returns.
    """

    def __init__(self, *, llm: LLMClient, retrieval: RetrievalService) -> None:
        self._llm = llm
        self._retrieval = retrieval

    async def plan(self, request: PlanningRequest) -> Plan:
        """Produce a validated :class:`Plan` for the given request.

        TODO: implement LangGraph state machine selecting the appropriate
        path. All paths must converge on a Plan that round-trips through
        ``Plan.model_validate`` before being returned.
        """
        logger.info("planning.plan.start", command=request.command)
        raise NotImplementedError("PlanningService.plan is not implemented yet")
