"""StubLLMClient sanity checks."""

from __future__ import annotations

from pydantic import BaseModel

from kuvox_ai.infrastructure import StubLLMClient
from kuvox_ai.infrastructure.llm_client import LLMMessage


class Empty(BaseModel):
    pass


async def test_stub_complete_returns_string() -> None:
    llm = StubLLMClient()
    await llm.connect()
    out = await llm.complete([LLMMessage(role="user", content="hi")])
    assert isinstance(out, str) and out
    await llm.close()


async def test_stub_complete_structured_yields_schema_instance() -> None:
    llm = StubLLMClient()
    out = await llm.complete_structured([LLMMessage(role="user", content="hi")], schema=Empty)
    assert isinstance(out, Empty)


async def test_stub_complete_code_returns_python() -> None:
    llm = StubLLMClient()
    code = await llm.complete_code([LLMMessage(role="user", content="hi")])
    assert "result" in code


async def test_stub_health_check() -> None:
    assert await StubLLMClient().health_check() is True
