"""Module-local pydantic models for sandbox."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class SandboxJob(BaseModel):
    """Payload pulled off the sandbox RabbitMQ queue."""

    job_id: UUID
    code: str = Field(description="Python source to execute in an isolated container.")
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description="String inputs made available to the script (e.g. as env vars).",
    )
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class SandboxResult(BaseModel):
    """Outcome of a sandbox execution."""

    job_id: UUID
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
