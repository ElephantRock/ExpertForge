"""Per-attempt identity record (Issue #6 decision §5).

A frozen Pydantic v2 record binding a specification fingerprint, a run identity,
an attempt identity, a creation timestamp, and optional resume lineage. Each
record is the unit serialized to ``run-identity.json``.

Two independent versions exist (Issue #6 §versioning):
- the specification-fingerprint envelope version (owned by
  :mod:`expertforge.identity.fingerprint`);
- the identity-record schema version (owned here).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from expertforge.identity.lineage import ResumeLineage

__all__ = ["IDENTITY_SCHEMA_VERSION", "AttemptIdentityRecord"]

# The identity-record (sidecar) schema version. Bumped only on incompatible
# sidecar changes (Issue #6 §versioning).
IDENTITY_SCHEMA_VERSION: int = 1


class AttemptIdentityRecord(BaseModel):
    """One attempt's identity, emitted before training begins."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    identity_schema_version: int = Field(default=IDENTITY_SCHEMA_VERSION)
    specification_fingerprint: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    attempt_id: str = Field(..., min_length=1)
    created_at_utc: datetime
    lineage: ResumeLineage | None = Field(default=None)

    def to_deterministic_json(self) -> bytes:
        """Compact, sorted-key, UTF-8, non-finite-prohibiting JSON of the record.

        Suitable as a stable serialization for the identity sidecar.
        """
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> AttemptIdentityRecord:
        """Validate and construct a record from a parsed sidecar mapping.

        Unknown ``identity_schema_version`` values are rejected here so callers
        cannot silently interpret an incompatible sidecar. Validation is
        non-strict on load because the source is a JSON serialization where
        datetimes legitimately appear as ISO strings; field types and
        constraints still apply.
        """
        version = data.get("identity_schema_version")
        if version != IDENTITY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported identity_schema_version {version!r}; this version "
                f"of ExpertForge understands version {IDENTITY_SCHEMA_VERSION}."
            )
        return cls.model_validate(data, strict=False)
