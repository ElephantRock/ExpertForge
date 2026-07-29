"""Identity-bound provenance record with typed frozen nested schema
(Issue #7 review items 3, 4).

All nested sections are frozen, extra-forbid Pydantic models — not mutable
dicts. The record binds to an :class:`AttemptIdentityRecord` and verifies that
``source.snapshot`` exists in the immutable inputs and matches the captured
source evidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from expertforge.identity.fingerprint import (
    ImmutableInput,
    SpecificationFingerprintRecord,
)
from expertforge.identity.ids import validate_attempt_id, validate_run_id
from expertforge.identity.record import AttemptIdentityRecord

__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "AcceleratorInfo",
    "CPUInfo",
    "CompletenessInfo",
    "DeviceInfo",
    "LockfileDigest",
    "MemoryInfo",
    "PlatformInfo",
    "ProvenanceRecord",
    "SoftwareEnvironment",
    "SourceState",
    "SourceStateSummary",
    "TopologyInfo",
]

PROVENANCE_SCHEMA_VERSION: int = 1
SOURCE_SNAPSHOT_INPUT_NAME = "source.snapshot"


def _section_config() -> ConfigDict:
    return ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)


# ---------------------------------------------------------------------------
# Typed nested sections
# ---------------------------------------------------------------------------


class CompletenessInfo(BaseModel):
    """Capture completeness + stable warning codes."""

    model_config = _section_config()

    status: str = Field(default="complete")  # complete | partial | error
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    limitations: tuple[str, ...] = Field(default_factory=tuple)


class SourceStateSummary(BaseModel):
    """The provenance-only source-state facts (not behavioral identity)."""

    model_config = _section_config()

    commit_sha: str = Field(..., min_length=1)
    branch: str = Field(default="HEAD")
    remote_url: str | None = Field(default=None)  # sanitized
    is_clean: bool
    is_canonical: bool
    tree_digest: str = Field(..., min_length=1)
    input_digest: str = Field(..., min_length=1)  # == source.snapshot digest


class CPUInfo(BaseModel):
    model_config = _section_config()

    status: str = Field(default="available")
    count: int | None = Field(default=None, ge=1)
    architecture: str | None = Field(default=None)


class MemoryInfo(BaseModel):
    model_config = _section_config()

    status: str = Field(default="available")
    total_bytes: int | None = Field(default=None, ge=0)


class PlatformInfo(BaseModel):
    model_config = _section_config()

    status: str = Field(default="available")
    system: str = Field(default="unknown")
    machine: str = Field(default="unknown")
    cpu: CPUInfo = Field(default_factory=CPUInfo)
    memory: MemoryInfo = Field(default_factory=MemoryInfo)


class LockfileDigest(BaseModel):
    model_config = _section_config()

    status: str  # available | unavailable | not_applicable | error
    algorithm: str | None = Field(default=None)
    digest: str | None = Field(default=None)
    reason: str | None = Field(default=None)


class SoftwareEnvironment(BaseModel):
    model_config = _section_config()

    python: dict[str, str] = Field(default_factory=dict)
    platform: PlatformInfo = Field(default_factory=PlatformInfo)
    dependencies: dict[str, str] = Field(default_factory=dict)
    lockfile: LockfileDigest = Field(default_factory=lambda: LockfileDigest(status="unavailable"))


class DeviceInfo(BaseModel):
    """One accelerator device with stable ordinal and typed memory."""

    model_config = _section_config()

    ordinal: int = Field(..., ge=0)
    model: str
    memory_total_mib: int | None = Field(default=None, ge=0)
    driver_version: str | None = Field(default=None)


class AcceleratorInfo(BaseModel):
    model_config = _section_config()

    status: str  # available | unavailable | not_applicable | error | redacted
    framework: str | None = Field(default=None)
    framework_version: str | None = Field(default=None)
    device_count: int | None = Field(default=None, ge=0)
    devices: tuple[DeviceInfo, ...] = Field(default_factory=tuple)
    reason: str | None = Field(default=None)


class TopologyInfo(BaseModel):
    model_config = _section_config()

    status: str  # available | not_applicable | error
    rank: int | None = Field(default=None, ge=0)
    local_rank: int | None = Field(default=None, ge=0)
    world_size: int | None = Field(default=None, ge=1)
    node_count: int | None = Field(default=None, ge=1)
    backend: str | None = Field(default=None)
    reason: str | None = Field(default=None)


# Alias for the source section — wraps the summary with evidence context.
class SourceState(BaseModel):
    model_config = _section_config()

    summary: SourceStateSummary
    completeness: CompletenessInfo = Field(default_factory=CompletenessInfo)


# ---------------------------------------------------------------------------
# Top-level record
# ---------------------------------------------------------------------------


class ProvenanceRecord(BaseModel):
    """The identity-bound provenance record with typed nested sections."""

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
    source: SourceState | None = Field(default=None)
    software: SoftwareEnvironment | None = Field(default=None)
    hardware: AcceleratorInfo | None = Field(default=None)
    topology: TopologyInfo | None = Field(default=None)
    completeness: CompletenessInfo = Field(default_factory=CompletenessInfo)

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
    def _validate_binding(self) -> ProvenanceRecord:
        # immutable_inputs must match the fingerprint's tuple exactly.
        if self.immutable_inputs != self.specification_fingerprint.immutable_inputs:
            raise ValueError(
                "immutable_inputs must match the specification fingerprint's "
                "immutable_inputs; do not supply a duplicate list."
            )
        # If source state is present, the source.snapshot input must exist and
        # its digest must match the captured source input_digest.
        if self.source is not None:
            snap_input = next(
                (ii for ii in self.immutable_inputs if ii.name == SOURCE_SNAPSHOT_INPUT_NAME),
                None,
            )
            if snap_input is None:
                raise ValueError(
                    "source state is present but no 'source.snapshot' immutable "
                    "input exists in the specification fingerprint."
                )
            if snap_input.digest != self.source.summary.input_digest:
                raise ValueError(
                    f"source.snapshot input digest {snap_input.digest!r} does not "
                    f"match captured source input_digest "
                    f"{self.source.summary.input_digest!r}."
                )
        return self

    @classmethod
    def from_identity(
        cls,
        identity: AttemptIdentityRecord,
        *,
        source: SourceState | None = None,
        software: SoftwareEnvironment | None = None,
        hardware: AcceleratorInfo | None = None,
        topology: TopologyInfo | None = None,
        completeness: CompletenessInfo | None = None,
    ) -> ProvenanceRecord:
        """Build a provenance record bound to ``identity``."""
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
            completeness=completeness or CompletenessInfo(),
        )

    def to_deterministic_json(self) -> bytes:
        """Compact, sorted-key, UTF-8, non-finite-prohibiting JSON."""
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ProvenanceRecord:
        """Validate and construct a record from a parsed sidecar mapping."""
        version = data.get("provenance_schema_version")
        if version != PROVENANCE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported provenance_schema_version {version!r}; this version "
                f"of ExpertForge understands version {PROVENANCE_SCHEMA_VERSION}."
            )
        return cls.model_validate(data, strict=False)
