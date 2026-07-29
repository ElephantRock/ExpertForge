"""Identity-bound provenance record (Issue #7 decision: identity binding).

``ProvenanceRecord`` consumes an :class:`AttemptIdentityRecord` and copies
``run_id``, ``attempt_id``, ``specification_fingerprint``, ``immutable_inputs``,
and ``start_time_utc`` (from ``created_at_utc`` — no second clock sample). It
must not accept an independently supplied duplicate immutable-input list.

The full provenance record is never hashed into the specification fingerprint.
Machine, hardware, platform, repository URL, branch, commit SHA, dependency
observations, topology, and timestamps are provenance-only facts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from expertforge.identity.fingerprint import ImmutableInput, SpecificationFingerprintRecord
from expertforge.identity.ids import validate_attempt_id, validate_run_id
from expertforge.identity.record import AttemptIdentityRecord

__all__ = ["PROVENANCE_SCHEMA_VERSION", "ProvenanceRecord"]

# The provenance sidecar schema version. Bumped only on incompatible sidecar
# changes (Issue #7 §version boundaries).
PROVENANCE_SCHEMA_VERSION: int = 1


class ProvenanceRecord(BaseModel):
    """The identity-bound provenance record.

    Constructed via :meth:`from_identity` (the normal path) or directly (e.g.
    when loading from a sidecar). ``start_time_utc`` must be timezone-aware UTC.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
        populate_by_name=True,
    )

    provenance_schema_version: int = Field(default=PROVENANCE_SCHEMA_VERSION)
    run_id: str = Field(..., min_length=1)
    attempt_id: str = Field(..., min_length=1)
    specification_fingerprint: SpecificationFingerprintRecord
    immutable_inputs: tuple[ImmutableInput, ...] = Field(default_factory=tuple)
    start_time_utc: datetime
    # Optional nested sections, populated by software/hardware capture (later
    # items in this issue). Nullable until those are wired in.
    source: dict[str, Any] | None = Field(default=None)
    software: dict[str, Any] | None = Field(default=None)
    hardware: dict[str, Any] | None = Field(default=None)
    topology: dict[str, Any] | None = Field(default=None)

    @field_validator("provenance_schema_version")
    @classmethod
    def _validate_schema_version(cls, v: int) -> int:
        if v != PROVENANCE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported provenance_schema_version {v}; this version "
                f"of ExpertForge understands version {PROVENANCE_SCHEMA_VERSION}."
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

    @field_validator("start_time_utc")
    @classmethod
    def _validate_start_time(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError(f"start_time_utc must be timezone-aware; got naive {v!r}.")
        if v.tzinfo.utcoffset(v) != timedelta(0):
            raise ValueError(
                f"start_time_utc must be UTC (offset 0); got offset {v.tzinfo.utcoffset(v)!r}."
            )
        return v

    @model_validator(mode="after")
    def _immutable_inputs_match_fingerprint(self) -> ProvenanceRecord:
        # The immutable inputs must exactly match those embedded in the
        # specification fingerprint (no independently supplied duplicate list).
        if self.immutable_inputs != self.specification_fingerprint.immutable_inputs:
            raise ValueError(
                "immutable_inputs must match the specification fingerprint's "
                "immutable_inputs; do not supply a duplicate list."
            )
        return self

    @classmethod
    def from_identity(
        cls,
        identity: AttemptIdentityRecord,
        *,
        source: dict[str, Any] | None = None,
        software: dict[str, Any] | None = None,
        hardware: dict[str, Any] | None = None,
        topology: dict[str, Any] | None = None,
    ) -> ProvenanceRecord:
        """Build a provenance record bound to ``identity``.

        Copies the identity's run/attempt/fingerprint/immutable-inputs and uses
        ``identity.created_at_utc`` as ``start_time_utc`` (no second clock
        sample). Does not accept an independently supplied immutable-input list.
        """
        return cls(
            run_id=identity.run_id,
            attempt_id=identity.attempt_id,
            specification_fingerprint=identity.specification_fingerprint,
            immutable_inputs=identity.specification_fingerprint.immutable_inputs,
            start_time_utc=identity.created_at_utc,
            source=source,
            software=software,
            hardware=hardware,
            topology=topology,
        )

    def to_deterministic_json(self) -> bytes:
        """Compact, sorted-key, UTF-8, non-finite-prohibiting JSON of the record."""
        import json

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ProvenanceRecord:
        """Validate and construct a record from a parsed sidecar mapping.

        Unknown ``provenance_schema_version`` is rejected. Non-strict on load
        (serialized JSON); field constraints still apply.
        """
        version = data.get("provenance_schema_version")
        if version != PROVENANCE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported provenance_schema_version {version!r}; this version "
                f"of ExpertForge understands version {PROVENANCE_SCHEMA_VERSION}."
            )
        return cls.model_validate(data, strict=False)
