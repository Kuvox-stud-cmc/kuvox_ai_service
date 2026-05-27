"""Operation schema — the typed instruction set a :class:`Plan` is composed of.

This is a discriminated union; the ``op`` field selects the variant. Three
example operations are defined today (trim, concatenate, transition) so the
end-to-end pipeline can be exercised.

TODO: extend with the full operation set from SRS Chapter 4. To add a new
operation:
  1. Define a new ``BaseModel`` subclass below with ``op: Literal["..."]``.
  2. Add it to the ``Operation`` union at the bottom of this file.
  3. Re-export from ``schemas/__init__.py``.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class _OperationBase(BaseModel):
    """Fields common to every operation."""

    model_config = {"frozen": True}


class TrimOperation(_OperationBase):
    """Cut a sub-range out of a single shot."""

    op: Literal["trim"] = "trim"
    shot_id: UUID
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)


class ConcatenateOperation(_OperationBase):
    """Append shots end-to-end in the order given."""

    op: Literal["concatenate"] = "concatenate"
    shot_ids: list[UUID] = Field(min_length=2)


class TransitionOperation(_OperationBase):
    """Insert a transition between two shots."""

    op: Literal["transition"] = "transition"
    from_shot_id: UUID
    to_shot_id: UUID
    style: Literal["cut", "fade", "dissolve", "wipe"] = "cut"
    duration_seconds: float = Field(default=0.5, ge=0.0)


# Discriminated union — pydantic uses the ``op`` field to pick the variant.
Operation = Annotated[
    TrimOperation | ConcatenateOperation | TransitionOperation,
    Field(discriminator="op"),
]
