"""Abstract LLM client and a stub implementation.

Real provider integrations belong in sibling modules (e.g. ``openai_client.py``)
and must be selected via ``Settings.llm_provider``. **Do not** hard-code a
provider in this file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from kuvox_ai.config import Settings
from kuvox_ai.logging import get_logger

logger = get_logger(__name__)


class LLMMessage(BaseModel):
    """A single message in an LLM conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient(ABC):
    """Contract every LLM provider must satisfy.

    Three call modes are exposed:

    * :meth:`complete` — free-form text completion.
    * :meth:`complete_structured` — tool/JSON-mode completion conforming to a
      pydantic schema. Used by the planning module.
    * :meth:`complete_code` — code-generation fallback used by the sandbox path.
    """

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str: ...

    @abstractmethod
    async def complete_structured(
        self,
        messages: list[LLMMessage],
        *,
        schema: type[BaseModel],
        max_tokens: int | None = None,
    ) -> BaseModel: ...

    @abstractmethod
    async def complete_code(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
    ) -> str: ...

    @abstractmethod
    async def health_check(self) -> bool: ...


class StubLLMClient(LLMClient):
    """Schema-conformant canned responses so the service runs without an API key.

    Returns deterministic placeholders. Useful for local dev, tests, and CI.
    """

    def __init__(self, model_name: str = "stub-model") -> None:
        self._model_name = model_name

    async def connect(self) -> None:
        logger.info("llm.stub.connected", model=self._model_name)

    async def close(self) -> None:
        logger.info("llm.stub.closed")

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        logger.debug("llm.stub.complete", n_messages=len(messages))
        return "[stub completion]"

    async def complete_structured(
        self,
        messages: list[LLMMessage],
        *,
        schema: type[BaseModel],
        max_tokens: int | None = None,
    ) -> BaseModel:
        logger.debug("llm.stub.complete_structured", schema=schema.__name__)
        # Build a minimal instance using field defaults; works for any schema
        # whose required fields all have defaults. Real implementations must
        # actually call a model and validate the response against ``schema``.
        try:
            return schema()
        except Exception as exc:  # noqa: BLE001
            raise NotImplementedError(
                f"StubLLMClient cannot synthesize an instance of {schema.__name__} "
                "with required fields lacking defaults. Provide a real LLMClient."
            ) from exc

    async def complete_code(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
    ) -> str:
        logger.debug("llm.stub.complete_code", n_messages=len(messages))
        return "# stub code\nresult = None\n"

    async def health_check(self) -> bool:
        return True


def build_llm_client(settings: Settings) -> LLMClient:
    """Factory: select the configured LLM provider implementation.

    TODO: register real providers here (e.g. OpenAI, Anthropic). Keep the
    ``LLMClient`` interface stable — implementations should live in this
    package and be added to this dispatch table.
    """
    provider = settings.llm_provider
    if provider == "stub":
        return StubLLMClient(model_name=settings.llm_model)
    raise NotImplementedError(f"Unknown LLM provider: {provider!r}")
