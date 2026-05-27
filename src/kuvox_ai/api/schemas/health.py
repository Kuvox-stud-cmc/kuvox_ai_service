"""Schemas for /health."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

DependencyStatus = Literal["healthy", "unhealthy"]


class DependencyHealth(BaseModel):
    name: str
    status: DependencyStatus


class HealthResponse(BaseModel):
    status: DependencyStatus
    dependencies: list[DependencyHealth]
