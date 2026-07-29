"""Resume lineage (Issue #6 decision §4; review item 5).

A frozen record of the parent run/attempt/checkpoint a resume continues from.
Native resumes record all three; ``parent_attempt_id`` is optional only for
imported or legacy provenance. Lineage metadata does not alter the
specification fingerprint. Parent IDs are validated against the same strict
path-safe regexes as generated IDs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from expertforge.identity.ids import (
    ATTEMPT_ID_PATTERN,
    RUN_ID_PATTERN,
)

__all__ = ["ResumeLineage"]


def _validate_run_like(v: str) -> str:
    if not RUN_ID_PATTERN.match(v):
        raise ValueError(f"Invalid parent_run_id {v!r}; must match {RUN_ID_PATTERN.pattern}.")
    return v


def _validate_attempt_like(v: str) -> str:
    if not ATTEMPT_ID_PATTERN.match(v):
        raise ValueError(
            f"Invalid parent_attempt_id {v!r}; must match {ATTEMPT_ID_PATTERN.pattern}."
        )
    return v


class ResumeLineage(BaseModel):
    """Parent execution a resume/fork continues from."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    parent_run_id: str = Field(..., min_length=1)
    # Optional only for imported/legacy provenance where the parent attempt
    # cannot be reconstructed.
    parent_attempt_id: str | None = Field(default=None)
    parent_checkpoint_id: str = Field(..., min_length=1)

    @field_validator("parent_run_id")
    @classmethod
    def _validate_parent_run(cls, v: str) -> str:
        return _validate_run_like(v)

    @field_validator("parent_attempt_id")
    @classmethod
    def _validate_parent_attempt(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _validate_attempt_like(v)
