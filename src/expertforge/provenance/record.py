"""Identity-bound provenance record with typed frozen nested schema
(Issue #7 review items 3, 4).

All nested sections are frozen, extra-forbid Pydantic models — not mutable
dicts. The record binds to an :class:`AttemptIdentityRecord` and verifies that
``source.snapshot`` exists in the immutable inputs and matches the captured
source evidence.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from expertforge.identity.fingerprint import (
    ImmutableInput,
    SpecificationFingerprintRecord,
)
from expertforge.identity.ids import validate_attempt_id, validate_run_id
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.provenance.source_snapshot import SourceSnapshot

__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "AcceleratorInfo",
    "CPUInfo",
    "CompletenessInfo",
    "DependencyObservation",
    "DeviceInfo",
    "HardwareAggregate",
    "LockfileDigest",
    "MemoryInfo",
    "PlatformInfo",
    "ProvenanceRecord",
    "PythonInfo",
    "SoftwareEnvironment",
    "TopologyInfo",
]

PROVENANCE_SCHEMA_VERSION: int = 1
SOURCE_SNAPSHOT_INPUT_NAME = "source.snapshot"
# 64 lowercase hexadecimal characters (SHA-256). Shared by the cross-field
# validators so a tampered sidecar cannot smuggle in a malformed digest.
_DIGEST_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _section_config() -> ConfigDict:
    return ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)


# ---------------------------------------------------------------------------
# Typed nested sections with Literal status domains + cross-field validators
# ---------------------------------------------------------------------------

# Shared status domains.
ObservationStatus = Literal["available", "unavailable", "not_applicable", "error", "redacted"]
CompletenessStatus = Literal["complete", "partial", "error"]


class CompletenessInfo(BaseModel):
    """Capture completeness + stable warning codes."""

    model_config = _section_config()

    status: CompletenessStatus = Field(default="complete")
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    limitations: tuple[str, ...] = Field(default_factory=tuple)


class CPUInfo(BaseModel):
    model_config = _section_config()

    # Default to 'unavailable' so the bare ``CPUInfo()`` default_factory used by
    # ``PlatformInfo`` is internally consistent (status='available' requires a
    # concrete count).
    status: ObservationStatus = Field(default="unavailable")
    count: int | None = Field(default=None, ge=1)
    architecture: str | None = Field(default=None)

    @model_validator(mode="after")
    def _check_count_status_consistency(self) -> CPUInfo:
        """Exhaustive status↔count consistency.

        - ``status="available"``: ``count`` must be present (≥1).
        - any other status (``unavailable``/``error``/``not_applicable``/
          ``redacted``): ``count`` must be ``None``.
        """
        if self.status == "available":
            if self.count is None:
                raise ValueError("CPUInfo status='available' requires count.")
        else:
            if self.count is not None:
                raise ValueError(f"CPUInfo status={self.status!r} forbids a non-None count.")
        return self


class MemoryInfo(BaseModel):
    model_config = _section_config()

    # Default to 'unavailable' so the bare ``MemoryInfo()`` default_factory used
    # by ``PlatformInfo`` is internally consistent (status='available' requires
    # a concrete total_bytes).
    status: ObservationStatus = Field(default="unavailable")
    total_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_total_status_consistency(self) -> MemoryInfo:
        """Exhaustive status↔total_bytes consistency.

        - ``status="available"``: ``total_bytes`` must be present (≥0).
        - any other status (``unavailable``/``error``/``not_applicable``/
          ``redacted``): ``total_bytes`` must be ``None``.
        """
        if self.status == "available":
            if self.total_bytes is None:
                raise ValueError("MemoryInfo status='available' requires total_bytes.")
        else:
            if self.total_bytes is not None:
                raise ValueError(
                    f"MemoryInfo status={self.status!r} forbids a non-None total_bytes."
                )
        return self


class PlatformInfo(BaseModel):
    model_config = _section_config()

    status: ObservationStatus = Field(default="available")
    system: str = Field(default="unknown")
    machine: str = Field(default="unknown")
    cpu: CPUInfo = Field(default_factory=CPUInfo)
    memory: MemoryInfo = Field(default_factory=MemoryInfo)


class LockfileDigest(BaseModel):
    model_config = _section_config()

    status: ObservationStatus
    algorithm: str | None = Field(default=None)
    digest: str | None = Field(default=None)
    reason: str | None = Field(default=None)

    @model_validator(mode="after")
    def _check_available_has_digest(self) -> LockfileDigest:
        """Exhaustive status↔(algorithm, digest) consistency.

        - ``status="available"``: ``algorithm`` must be exactly ``"sha256"`` and
          ``digest`` must be a valid 64-lowercase-hex SHA-256.
        - any other status (``unavailable``/``error``/``not_applicable``/
          ``redacted``): ``algorithm`` AND ``digest`` must both be ``None``.
        """
        if self.status == "available":
            if self.algorithm != "sha256":
                raise ValueError(
                    f"LockfileDigest status='available' requires algorithm='sha256'; "
                    f"got {self.algorithm!r}."
                )
            if not self.digest or not _DIGEST_HEX_PATTERN.fullmatch(self.digest):
                raise ValueError(
                    "LockfileDigest status='available' requires a valid 64-lowercase-hex "
                    f"digest; got {self.digest!r}."
                )
        else:
            if self.algorithm is not None or self.digest is not None:
                raise ValueError(
                    f"LockfileDigest status={self.status!r} requires algorithm=None and "
                    f"digest=None; got algorithm={self.algorithm!r}, digest={self.digest!r}."
                )
        return self


class PythonInfo(BaseModel):
    """Typed Python environment info (frozen, not a mutable dict)."""

    model_config = _section_config()

    version: str = Field(..., min_length=1)
    implementation: str = Field(..., min_length=1)
    build: str | None = Field(default=None)


class DependencyObservation(BaseModel):
    """One installed distribution: normalized name → version."""

    model_config = _section_config()

    name: str = Field(..., min_length=1)  # PEP 503 normalized
    version: str = Field(..., min_length=1)


class SoftwareEnvironment(BaseModel):
    model_config = _section_config()

    python: PythonInfo
    platform: PlatformInfo = Field(default_factory=PlatformInfo)
    dependencies: tuple[DependencyObservation, ...] = Field(default_factory=tuple)
    # Recorded when the SAME normalized dependency name is observed at
    # DIFFERENT versions across distributions. First observation wins; the
    # conflict is surfaced here rather than silently lost.
    dependency_conflicts: tuple[str, ...] = Field(default_factory=tuple)
    lockfile: LockfileDigest = Field(default_factory=lambda: LockfileDigest(status="unavailable"))

    @field_validator("dependencies")
    @classmethod
    def _deps_sorted_unique(
        cls, v: tuple[DependencyObservation, ...]
    ) -> tuple[DependencyObservation, ...]:
        names = [d.name for d in v]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Duplicate dependency names: {sorted({n for n in names if names.count(n) > 1})}"
            )
        if names != sorted(names):
            raise ValueError("Dependencies must be sorted by name.")
        return v

    @field_validator("dependency_conflicts")
    @classmethod
    def _conflicts_sorted_unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError(
                f"Duplicate dependency_conflicts: {sorted({c for c in v if v.count(c) > 1})}"
            )
        if list(v) != sorted(v):
            raise ValueError("dependency_conflicts must be sorted.")
        return v


class DeviceInfo(BaseModel):
    """One accelerator device with stable ordinal and typed memory."""

    model_config = _section_config()

    ordinal: int = Field(..., ge=0)
    model: str
    memory_total_mib: int | None = Field(default=None, ge=0)
    driver_version: str | None = Field(default=None)


class AcceleratorInfo(BaseModel):
    model_config = _section_config()

    status: ObservationStatus
    framework: str | None = Field(default=None)
    framework_version: str | None = Field(default=None)
    runtime_version: str | None = Field(default=None)
    precision_status: ObservationStatus = Field(default="not_applicable")
    device_count: int | None = Field(default=None, ge=0)
    devices: tuple[DeviceInfo, ...] = Field(default_factory=tuple)
    reason: str | None = Field(default=None)

    @model_validator(mode="after")
    def _check_consistency(self) -> AcceleratorInfo:
        """Exhaustive status↔(devices, device_count, framework/runtime) consistency.

        - ``status="available"``: ``device_count`` > 0, ``len(devices) ==
          device_count``, and device ordinals are unique.
        - any non-available status: ``devices`` must be empty,
          ``device_count`` must be None or 0, and ``framework``,
          ``framework_version``, and ``runtime_version`` must all be ``None``
          (descriptive metadata is only meaningful when devices were observed).
        """
        if self.status == "available":
            if self.device_count is None or self.device_count == 0:
                raise ValueError("AcceleratorInfo status='available' requires device_count > 0.")
            if len(self.devices) != self.device_count:
                raise ValueError(
                    f"device_count ({self.device_count}) != len(devices) ({len(self.devices)})."
                )
            ordinals = [d.ordinal for d in self.devices]
            if len(set(ordinals)) != len(ordinals):
                raise ValueError(f"Duplicate device ordinals: {ordinals}")
        else:
            if self.devices:
                raise ValueError(
                    f"AcceleratorInfo status={self.status!r} forbids non-empty devices."
                )
            if self.device_count not in (None, 0):
                raise ValueError(
                    f"AcceleratorInfo status={self.status!r} forbids a non-zero "
                    f"device_count; got {self.device_count!r}."
                )
            if (
                self.framework is not None
                or self.framework_version is not None
                or self.runtime_version is not None
            ):
                raise ValueError(
                    f"AcceleratorInfo status={self.status!r} forbids framework/"
                    f"framework_version/runtime_version; got framework="
                    f"{self.framework!r}, framework_version={self.framework_version!r}, "
                    f"runtime_version={self.runtime_version!r}."
                )
        return self


class TopologyInfo(BaseModel):
    model_config = _section_config()

    status: Literal["available", "not_applicable", "error"]
    rank: int | None = Field(default=None, ge=0)
    local_rank: int | None = Field(default=None, ge=0)
    world_size: int | None = Field(default=None, ge=1)
    node_count: int | None = Field(default=None, ge=1)
    backend: str | None = Field(default=None)
    reason: str | None = Field(default=None)
    # Invalid numeric topology env values observed at capture time (e.g.
    # ``RANK=not-an-int``) that could not be parsed. Stored durably so a
    # misconfigured launcher is visible on round-trip rather than silently
    # discarded. Stable, sorted, unique codes.
    topology_warnings: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("topology_warnings")
    @classmethod
    def _topology_warnings_sorted_unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if list(v) != sorted(v):
            raise ValueError("topology_warnings must be sorted.")
        if len(set(v)) != len(v):
            raise ValueError("topology_warnings must be unique.")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> TopologyInfo:
        """Exhaustive status↔field consistency.

        - ``status="available"``: ``rank`` and ``world_size`` present,
          ``rank < world_size``, and ``local_rank < world_size`` when present.
        - ``status="error"`` or ``"not_applicable"``: ALL of ``rank``,
          ``world_size``, ``local_rank``, ``node_count``, and ``backend`` must
          be ``None``. A record that claims error/not_applicable yet carries
          ANY concrete topology value is internally inconsistent (and a likely
          sign of tampering or a logic bug at capture time). ``node_count`` and
          ``backend`` are NOT allowed here — only meaningful alongside a real
          distributed topology (status='available').
        """
        if self.status == "available":
            if self.rank is None or self.world_size is None:
                raise ValueError("TopologyInfo status='available' requires rank and world_size.")
            if self.rank >= self.world_size:
                raise ValueError("rank must be < world_size.")
            # local_rank < world_size when both are present.
            if self.local_rank is not None and self.local_rank >= self.world_size:
                raise ValueError("local_rank must be < world_size.")
        else:
            # error or not_applicable: rank, world_size, local_rank, node_count,
            # AND backend must ALL be absent.
            present = {
                name: val
                for name, val in (
                    ("rank", self.rank),
                    ("world_size", self.world_size),
                    ("local_rank", self.local_rank),
                    ("node_count", self.node_count),
                    ("backend", self.backend),
                )
                if val is not None
            }
            if present:
                raise ValueError(
                    f"TopologyInfo status={self.status!r} requires rank, world_size, "
                    "local_rank, node_count, and backend to ALL be None; got "
                    f"concrete values {sorted(present)!r}."
                )
        return self


class HardwareAggregate(BaseModel):
    """Frozen aggregate of accelerator + topology capture.

    ``capture_hardware()`` returns this typed model (not a bare dict) so the
    combined result is validated, frozen, and round-trips through JSON.
    ``topology_warnings`` records invalid numeric topology env values that were
    observed but could not be parsed (rather than silently discarding them).
    """

    model_config = _section_config()

    accelerator: AcceleratorInfo
    topology: TopologyInfo
    topology_warnings: tuple[str, ...] = Field(default_factory=tuple)


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
    source: SourceSnapshot
    software: SoftwareEnvironment
    hardware: AcceleratorInfo
    topology: TopologyInfo
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
        # source.snapshot input must exist and match the captured snapshot's
        # input_digest. Source is mandatory, so this always runs.
        snap_input = next(
            (ii for ii in self.immutable_inputs if ii.name == SOURCE_SNAPSHOT_INPUT_NAME),
            None,
        )
        if snap_input is None:
            raise ValueError(
                "No 'source.snapshot' immutable input exists in the specification "
                "fingerprint; every provenance record requires source binding."
            )
        if snap_input.digest != self.source.input_digest:
            raise ValueError(
                f"source.snapshot input digest {snap_input.digest!r} does not "
                f"match captured source input_digest "
                f"{self.source.input_digest!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_completeness_consistency(self) -> ProvenanceRecord:
        """Verify ``self.completeness`` matches the derived whole-record value.

        Catches tampered sidecars (a record whose stored ``completeness`` does
        not reflect the actual section states) and explicit overrides supplied
        to :meth:`from_identity` that disagree with the captured sections. The
        stored status must equal the recomputed status, the stored warning set
        must equal the recomputed warning set, AND the stored limitation set
        must equal the recomputed limitation set.
        """
        derived_status, derived_warnings, derived_limitations = (
            ProvenanceRecord._derive_completeness(
                self.source, self.software, self.hardware, self.topology
            )
        )
        if self.completeness.status != derived_status:
            raise ValueError(
                f"Stored completeness status {self.completeness.status!r} does not "
                f"match the derived whole-record status {derived_status!r}. "
                "Completeness is derived from all mandatory sections; do not "
                "supply a manual override that disagrees with them."
            )
        if self.completeness.warnings != derived_warnings:
            raise ValueError(
                f"Stored completeness warnings {self.completeness.warnings!r} do "
                f"not match the derived whole-record warnings {derived_warnings!r}."
            )
        if self.completeness.limitations != derived_limitations:
            raise ValueError(
                f"Stored completeness limitations {self.completeness.limitations!r} do "
                f"not match the derived whole-record limitations {derived_limitations!r}."
            )
        return self

    @staticmethod
    def _derive_completeness(
        source: SourceSnapshot,
        software: SoftwareEnvironment,
        hardware: AcceleratorInfo,
        topology: TopologyInfo,
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        """Derive top-level completeness + warnings + limitations from sections.

        Returns ``(status, warnings, limitations)``. ``error``/``partial`` states
        in any section propagate to the top-level status with typed warnings and
        typed limitations. Pure static derivation so it can run before the frozen
        record is constructed.

        Limitations are propagated from source evidence (unreadable files,
        submodule/gitlink inspection failures, dirty submodules, the always-on
        standard limitations) so the whole-record completeness honestly reports
        what could not be captured — never silently ``complete``.

        Invalid numeric topology env values are stored durably on
        ``TopologyInfo.topology_warnings`` (captured by
        :func:`capture_hardware`); they are surfaced here so a misconfigured
        launcher is visible at the whole-record level, not silently dropped.
        """
        warnings: list[str] = []
        limitations: list[str] = []
        has_partial = False
        has_error = False

        # Source evidence.
        if source.evidence is not None:
            if source.evidence.completeness == "partial":
                has_partial = True
                warnings.extend(source.evidence.warnings)
            elif source.evidence.completeness == "error":
                has_error = True
                warnings.extend(source.evidence.warnings)
            # Propagate the section's own limitations to the whole record so a
            # hidden incompleteness (unreadable files, submodule failures, dirty
            # submodule content not snapshotted) is visible at the top level.
            limitations.extend(source.evidence.limitations)
        if not source.is_clean:
            warnings.append("non_canonical_dirty_source")
            limitations.append("non_canonical_dirty_source")
        warnings.extend(w.code for w in source.remote_warnings)

        # Software: dependency version conflicts (same normalized name at
        # different versions) are honest partial-completeness signals.
        if software.dependency_conflicts:
            has_partial = True
            warnings.append("dependency_version_conflict")
            limitations.append("dependency_version_conflict")

        # Software: platform CPU/memory error/redacted states propagate.
        cpu = software.platform.cpu
        if cpu.status == "error":
            has_error = True
            warnings.append("cpu_error")
            limitations.append("cpu_error")
        elif cpu.status == "redacted":
            has_partial = True
            warnings.append("cpu_redacted")
            limitations.append("cpu_redacted")
        elif cpu.status == "unavailable":
            limitations.append("cpu_unavailable")
        memory = software.platform.memory
        if memory.status == "error":
            has_error = True
            warnings.append("memory_error")
            limitations.append("memory_error")
        elif memory.status == "redacted":
            has_partial = True
            warnings.append("memory_redacted")
            limitations.append("memory_redacted")
        elif memory.status == "unavailable":
            limitations.append("memory_unavailable")
        if software.platform.status == "error":
            has_error = True
            warnings.append("platform_error")
            limitations.append("platform_error")
        elif software.platform.status == "redacted":
            has_partial = True
            warnings.append("platform_redacted")
            limitations.append("platform_redacted")

        # Software lockfile.
        if software.lockfile.status == "error":
            has_error = True
            warnings.append("lockfile_error")
            limitations.append("lockfile_error")
        elif software.lockfile.status == "unavailable":
            limitations.append("lockfile_unavailable")

        # Hardware.
        if hardware.status == "error":
            has_error = True
            warnings.append("accelerator_error")
            limitations.append("accelerator_error")
        elif hardware.status == "redacted":
            has_partial = True
            warnings.append("accelerator_redacted")
            limitations.append("accelerator_redacted")
        elif hardware.status == "unavailable":
            limitations.append("accelerator_unavailable")

        # Topology.
        if topology.status == "error":
            has_error = True
            warnings.append("topology_error")
            limitations.append("topology_error")
        elif topology.status == "not_applicable":
            limitations.append("topology_not_applicable")
        # Invalid numeric topology env values surface as warnings (do not by
        # themselves force partial unless topology itself is in error).
        warnings.extend(topology.topology_warnings)
        limitations.extend(topology.topology_warnings)

        status = "error" if has_error else ("partial" if has_partial or warnings else "complete")
        # Deduplicate + sort warnings and limitations for determinism.
        return status, tuple(sorted(set(warnings))), tuple(sorted(set(limitations)))

    @classmethod
    def from_identity(
        cls,
        identity: AttemptIdentityRecord,
        *,
        source: SourceSnapshot,
        software: SoftwareEnvironment | None = None,
        hardware: AcceleratorInfo | None = None,
        topology: TopologyInfo | None = None,
        completeness: CompletenessInfo | None = None,
    ) -> ProvenanceRecord:
        """Build a provenance record bound to ``identity``.

        ``source`` (the complete typed SourceSnapshot) is mandatory.
        Software/hardware/topology default to typed unavailable values when
        not supplied. Completeness is derived from all sections unless
        explicitly overridden (and the override must match the derived value).
        """
        resolved_software = software or SoftwareEnvironment(
            python=PythonInfo(version="unknown", implementation="unknown"),
            platform=PlatformInfo(
                cpu=CPUInfo(status="unavailable"),
                memory=MemoryInfo(status="unavailable"),
            ),
            lockfile=LockfileDigest(status="unavailable"),
        )
        resolved_hw = hardware or AcceleratorInfo(status="unavailable")
        resolved_topo = topology or TopologyInfo(status="not_applicable")

        # Derive completeness BEFORE constructing the frozen record so the
        # completeness-consistency model_validator sees a matching value on the
        # first pass. An explicit override must itself match the derived value
        # (the validator rejects overrides that disagree with the sections).
        if completeness is None:
            derived_status, derived_warnings, derived_limitations = cls._derive_completeness(
                source, resolved_software, resolved_hw, resolved_topo
            )
            resolved_completeness = CompletenessInfo(
                status=cast("CompletenessStatus", derived_status),
                warnings=derived_warnings,
                limitations=derived_limitations,
            )
        else:
            resolved_completeness = completeness

        return cls(
            run_id=identity.run_id,
            attempt_id=identity.attempt_id,
            specification_fingerprint=identity.specification_fingerprint,
            immutable_inputs=identity.specification_fingerprint.immutable_inputs,
            start_time_utc=identity.created_at_utc,
            source=source,
            software=resolved_software,
            hardware=resolved_hw,
            topology=resolved_topo,
            completeness=resolved_completeness,
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
