"""Schemas for /health."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

DependencyStatus = Literal["healthy", "unhealthy", "disabled"]
OverallStatus = Literal["healthy", "degraded", "unhealthy"]


class DependencyHealth(BaseModel):
    name: str
    status: DependencyStatus
    required: bool


class HealthResponse(BaseModel):
    status: OverallStatus
    dependencies: list[DependencyHealth]
