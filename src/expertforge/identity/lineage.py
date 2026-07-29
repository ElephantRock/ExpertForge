"""Resume lineage (Issue #6 decision §4).

A frozen record of the parent run/attempt/checkpoint a resume continues from.
Native resumes record all three; ``parent_attempt_id`` is optional only for
imported or legacy provenance. Lineage metadata does not alter the
specification fingerprint.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ResumeLineage"]


class ResumeLineage(BaseModel):
    """Parent execution a resume/fork continues from."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    parent_run_id: str = Field(..., min_length=1)
    # Optional only for imported/legacy provenance where the parent attempt
    # cannot be reconstructed.
    parent_attempt_id: str | None = Field(default=None)
    parent_checkpoint_id: str = Field(..., min_length=1)
