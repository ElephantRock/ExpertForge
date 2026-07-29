"""Per-attempt identity record (Issue #6 decision §5; review items 2, 5).

A frozen Pydantic v2 record binding a typed specification fingerprint, a run
identity, an attempt identity, a UTC creation timestamp, and optional resume
lineage. Each record is the unit serialized to ``run-identity.json``.

The fingerprint is stored as the full reconstructable
:class:`SpecificationFingerprintRecord` (envelope + canonical-config digest +
immutable inputs + public digest), not a bare string, so the sidecar round-trips
the typed structure. Run/attempt IDs are validated against strict path-safe
regexes; timestamps must be timezone-aware UTC.

Two independent versions exist (Issue #6 §versioning):
- the specification-fingerprint envelope version (owned by
  :mod:`expertforge.identity.fingerprint`);
- the identity-record schema version (owned here).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from expertforge.identity.fingerprint import SpecificationFingerprintRecord
from expertforge.identity.ids import validate_attempt_id, validate_run_id
from expertforge.identity.lineage import ResumeLineage

__all__ = ["IDENTITY_SCHEMA_VERSION", "AttemptIdentityRecord"]

# The identity-record (sidecar) schema version. Bumped only on incompatible
# sidecar changes (Issue #6 §versioning).
IDENTITY_SCHEMA_VERSION: int = 1


class AttemptIdentityRecord(BaseModel):
    """One attempt's identity, emitted before training begins."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
        populate_by_name=True,
    )

    identity_schema_version: int = Field(default=IDENTITY_SCHEMA_VERSION)
    specification_fingerprint: SpecificationFingerprintRecord
    run_id: str = Field(..., min_length=1)
    attempt_id: str = Field(..., min_length=1)
    created_at_utc: datetime
    lineage: ResumeLineage | None = Field(default=None)

    @field_validator("identity_schema_version")
    @classmethod
    def _validate_schema_version(cls, v: int) -> int:
        if v != IDENTITY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported identity_schema_version {v}; this version "
                f"of ExpertForge understands version {IDENTITY_SCHEMA_VERSION}."
            )
        return v

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, v: str) -> str:
        validate_run_id(v)
        return v

    @field_validator("attempt_id")
    @classmethod
    def _validate_attempt_id(cls, v: str) -> str:
        validate_attempt_id(v)
        return v

    @field_validator("created_at_utc")
    @classmethod
    def _validate_created_at(cls, v: datetime) -> datetime:
        # Reject naive datetimes AND non-UTC-aware datetimes. The contract is
        # UTC-only; a non-UTC offset is a caller error to surface, not silently
        # normalize (silent normalization hides misuse at the boundary).
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError(f"created_at_utc must be timezone-aware; got naive {v!r}.")
        if v.tzinfo.utcoffset(v) != timedelta(0):
            raise ValueError(
                f"created_at_utc must be UTC (offset 0); got offset {v.tzinfo.utcoffset(v)!r}."
            )
        return v

    def fingerprint_digest_str(self) -> str:
        """The public ``spec-v1-sha256-<hex>`` digest of this record's fingerprint."""
        return self.specification_fingerprint.digest_str

    def to_deterministic_json(self) -> bytes:
        """Compact, sorted-key, UTF-8, non-finite-prohibiting JSON of the record."""
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

        Non-strict on load (the source is JSON where datetimes/envelopes appear
        serialized); field constraints and version checks still apply. The
        embedded fingerprint's stored digest is verified against its envelope.
        """
        version = data.get("identity_schema_version")
        if version != IDENTITY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported identity_schema_version {version!r}; this version "
                f"of ExpertForge understands version {IDENTITY_SCHEMA_VERSION}."
            )
        record = cls.model_validate(data, strict=False)
        record.specification_fingerprint.verify_digest()
        return record
