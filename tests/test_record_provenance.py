"""Tests for the identity-bound ProvenanceRecord (Issue #7 decision: identity binding).

ProvenanceRecord consumes an AttemptIdentityRecord and copies run_id,
attempt_id, specification_fingerprint, immutable_inputs, and start_time_utc. It
must not accept an independently supplied duplicate immutable-input list.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import pytest
from pydantic import ValidationError

from expertforge.config.resolve import resolve_config
from expertforge.identity.emit import emit_attempt_identity
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.provenance.record import (
    PROVENANCE_SCHEMA_VERSION,
    AcceleratorInfo,
    CPUInfo,
    DependencyObservation,
    LockfileDigest,
    MemoryInfo,
    PlatformInfo,
    ProvenanceRecord,
    PythonInfo,
    SoftwareEnvironment,
    TopologyInfo,
)
from expertforge.provenance.source_snapshot import (
    SourceSnapshot,
    capture_source_snapshot,
    source_snapshot_immutable_input,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
_FIXED = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _init_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)
    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


def _identity(tmp_path: Path) -> tuple[AttemptIdentityRecord, SourceSnapshot]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _init_repo(repo)
    snap = capture_source_snapshot(repo)
    rec, _ = emit_attempt_identity(
        artifact_root=tmp_path,
        config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
        immutable_inputs=[source_snapshot_immutable_input(snap)],
        clock=lambda: _FIXED,
        entropy=lambda n: bytes(n),
    )
    return rec, snap


class _UnavailableSections(TypedDict):
    """Typed unavailable software/hardware/topology for direct construction."""

    software: SoftwareEnvironment
    hardware: AcceleratorInfo
    topology: TopologyInfo


def _unavailable_sections() -> _UnavailableSections:
    """Typed unavailable software/hardware/topology for direct construction."""
    return {
        "software": SoftwareEnvironment(
            python=PythonInfo(version="0.0.0", implementation="cpython"),
            platform=PlatformInfo(
                cpu=CPUInfo(status="unavailable"),
                memory=MemoryInfo(status="unavailable"),
            ),
            lockfile=LockfileDigest(status="unavailable"),
        ),
        "hardware": AcceleratorInfo(status="unavailable"),
        "topology": TopologyInfo(status="not_applicable"),
    }


# --- identity binding -----------------------------------------------------


class TestProvenanceRecordIdentityBinding:
    def test_from_identity_copies_binding_fields(self, tmp_path: Path) -> None:
        ident, snap = _identity(tmp_path)
        rec = ProvenanceRecord.from_identity(ident, source=snap)
        assert rec.run_id == ident.run_id
        assert rec.attempt_id == ident.attempt_id
        assert rec.specification_fingerprint == ident.specification_fingerprint
        assert rec.immutable_inputs == ident.specification_fingerprint.immutable_inputs

    def test_start_time_copied_from_identity_created_at(self, tmp_path: Path) -> None:
        ident, snap = _identity(tmp_path)
        rec = ProvenanceRecord.from_identity(ident, source=snap)
        # start_time_utc is copied from created_at_utc — no second clock sample.
        assert rec.start_time_utc == ident.created_at_utc

    def test_record_carries_schema_version(self, tmp_path: Path) -> None:
        ident, snap = _identity(tmp_path)
        rec = ProvenanceRecord.from_identity(ident, source=snap)
        assert rec.provenance_schema_version == PROVENANCE_SCHEMA_VERSION

    def test_record_is_frozen(self, tmp_path: Path) -> None:
        ident, snap = _identity(tmp_path)
        rec = ProvenanceRecord.from_identity(ident, source=snap)
        with pytest.raises(ValidationError):
            rec.run_id = "other"  # type: ignore[misc]

    def test_rejects_independently_supplied_immutable_inputs(self, tmp_path: Path) -> None:
        # The binding contract: a direct construction with an immutable-input
        # list that does not match the fingerprint's inputs is rejected.
        ident, snap = _identity(tmp_path)
        from expertforge.identity.fingerprint import ImmutableInput

        dup = (ImmutableInput(name="x", algorithm="sha256", digest="a" * 64),)
        with pytest.raises(ValidationError):
            ProvenanceRecord(
                run_id=ident.run_id,
                attempt_id=ident.attempt_id,
                specification_fingerprint=ident.specification_fingerprint,
                immutable_inputs=dup,  # does NOT match the fingerprint's inputs
                start_time_utc=ident.created_at_utc,
                source=snap,
                **_unavailable_sections(),
            )

    def test_naive_start_time_rejected_on_direct_construction(self, tmp_path: Path) -> None:
        ident, snap = _identity(tmp_path)
        with pytest.raises(ValidationError):
            ProvenanceRecord(
                run_id=ident.run_id,
                attempt_id=ident.attempt_id,
                specification_fingerprint=ident.specification_fingerprint,
                immutable_inputs=ident.specification_fingerprint.immutable_inputs,
                start_time_utc=datetime(2026, 1, 1, 0, 0, 0),  # naive
                source=snap,
                **_unavailable_sections(),
            )

    def test_non_utc_start_time_rejected_on_direct_construction(self, tmp_path: Path) -> None:
        from datetime import timedelta, timezone

        ident, snap = _identity(tmp_path)
        non_utc = timezone(timedelta(hours=2))
        with pytest.raises(ValidationError):
            ProvenanceRecord(
                run_id=ident.run_id,
                attempt_id=ident.attempt_id,
                specification_fingerprint=ident.specification_fingerprint,
                immutable_inputs=ident.specification_fingerprint.immutable_inputs,
                start_time_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=non_utc),
                source=snap,
                **_unavailable_sections(),
            )


# --- completeness derivation from all sections (review item 4) -------------


class TestDeriveCompletenessAllSections:
    """``_derive_completeness`` propagates dependency conflicts, CPU/memory
    platform states, and topology warnings to the whole-record completeness."""

    def test_dependency_conflict_propagates_partial(self, tmp_path: Path) -> None:
        ident, snap = _identity(tmp_path)
        from expertforge.provenance.record import DependencyConflict

        software = SoftwareEnvironment(
            python=PythonInfo(version="3.11", implementation="cpython"),
            platform=PlatformInfo(
                cpu=CPUInfo(status="unavailable"),
                memory=MemoryInfo(status="unavailable"),
            ),
            dependencies=(DependencyObservation(name="numpy", version="1.0.0"),),
            dependency_conflicts=(
                DependencyConflict(name="numpy", observed_versions=("1.0.0", "2.0.0")),
            ),
            lockfile=LockfileDigest(status="unavailable"),
        )
        rec = ProvenanceRecord.from_identity(ident, source=snap, software=software)
        assert "dependency_version_conflict" in rec.completeness.warnings
        assert rec.completeness.status == "partial"

    def test_cpu_error_propagates(self, tmp_path: Path) -> None:
        ident, snap = _identity(tmp_path)
        software = SoftwareEnvironment(
            python=PythonInfo(version="3.11", implementation="cpython"),
            platform=PlatformInfo(
                cpu=CPUInfo(status="error"),
                memory=MemoryInfo(status="unavailable"),
            ),
            lockfile=LockfileDigest(status="unavailable"),
        )
        rec = ProvenanceRecord.from_identity(ident, source=snap, software=software)
        assert "cpu_error" in rec.completeness.warnings
        assert rec.completeness.status == "error"

    def test_memory_redacted_propagates(self, tmp_path: Path) -> None:
        ident, snap = _identity(tmp_path)
        software = SoftwareEnvironment(
            python=PythonInfo(version="3.11", implementation="cpython"),
            platform=PlatformInfo(
                cpu=CPUInfo(status="unavailable"),
                memory=MemoryInfo(status="redacted"),
            ),
            lockfile=LockfileDigest(status="unavailable"),
        )
        rec = ProvenanceRecord.from_identity(ident, source=snap, software=software)
        assert "memory_redacted" in rec.completeness.warnings
        assert rec.completeness.status == "partial"

    def test_platform_error_propagates(self, tmp_path: Path) -> None:
        ident, snap = _identity(tmp_path)
        software = SoftwareEnvironment(
            python=PythonInfo(version="3.11", implementation="cpython"),
            platform=PlatformInfo(
                status="error",
                cpu=CPUInfo(status="unavailable"),
                memory=MemoryInfo(status="unavailable"),
            ),
            lockfile=LockfileDigest(status="unavailable"),
        )
        rec = ProvenanceRecord.from_identity(ident, source=snap, software=software)
        assert "platform_error" in rec.completeness.warnings
        assert rec.completeness.status == "error"

    def test_topology_warnings_propagate_durable(self, tmp_path: Path) -> None:
        # TopologyInfo carries topology_warnings durably (round-trips through
        # JSON) and they appear in the derived whole-record warnings.
        ident, snap = _identity(tmp_path)
        topo = TopologyInfo(
            status="available",
            rank=0,
            world_size=2,
            topology_warnings=("invalid_topology_env_value:RANK",),
        )
        rec = ProvenanceRecord.from_identity(ident, source=snap, topology=topo)
        assert "invalid_topology_env_value:RANK" in rec.completeness.warnings
        # Round-trips through JSON: topology_warnings survive on the model.
        restored = ProvenanceRecord.model_validate_json(rec.model_dump_json())
        assert restored.topology.topology_warnings == ("invalid_topology_env_value:RANK",)
        assert "invalid_topology_env_value:RANK" in restored.completeness.warnings


# --- CPU/Memory cross-field validators (review item 5) ---------------------


class TestCpuMemoryCrossFieldValidators:
    def test_cpu_available_requires_count(self) -> None:
        with pytest.raises(ValidationError):
            CPUInfo(status="available", count=None)

    def test_cpu_available_with_count_accepted(self) -> None:
        cpu = CPUInfo(status="available", count=4)
        assert cpu.count == 4

    def test_cpu_unavailable_rejects_count(self) -> None:
        with pytest.raises(ValidationError):
            CPUInfo(status="unavailable", count=4)

    def test_cpu_error_rejects_count(self) -> None:
        with pytest.raises(ValidationError):
            CPUInfo(status="error", count=4)

    def test_memory_available_requires_total_bytes(self) -> None:
        with pytest.raises(ValidationError):
            MemoryInfo(status="available", total_bytes=None)

    def test_memory_unavailable_rejects_total_bytes(self) -> None:
        with pytest.raises(ValidationError):
            MemoryInfo(status="unavailable", total_bytes=1024)


# --- TopologyInfo model invariants (review item 5) -------------------------


class TestTopologyInfoInvariants:
    def test_error_rejects_local_rank(self) -> None:
        # status='error' must not carry concrete rank/world_size/local_rank.
        with pytest.raises(ValidationError):
            TopologyInfo(status="error", local_rank=0)

    def test_not_applicable_rejects_local_rank(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(status="not_applicable", local_rank=0)

    def test_error_rejects_node_count_and_backend(self) -> None:
        # Review item 5 (tightened): status='error' must not carry node_count
        # OR backend — they are only meaningful alongside a real distributed
        # topology (status='available'). A record claiming error yet carrying
        # these is internally inconsistent / likely tampered.
        with pytest.raises(ValidationError):
            TopologyInfo(status="error", node_count=2, backend="nccl")
        with pytest.raises(ValidationError):
            TopologyInfo(status="error", node_count=2)
        with pytest.raises(ValidationError):
            TopologyInfo(status="error", backend="nccl")
        with pytest.raises(ValidationError):
            TopologyInfo(status="not_applicable", node_count=2, backend="nccl")

    def test_available_rejects_local_rank_ge_world_size(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(status="available", rank=0, world_size=4, local_rank=4)

    def test_topology_warnings_must_be_sorted(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(
                status="available",
                rank=0,
                world_size=2,
                topology_warnings=("z_warn", "a_warn"),
            )


# --- LockfileDigest exhaustive validator (review item 5) -------------------


class TestLockfileDigestExhaustiveValidator:
    """Review item 5: LockfileDigest status↔(algorithm, digest) consistency."""

    def test_available_requires_sha256_algorithm(self) -> None:
        with pytest.raises(ValidationError):
            LockfileDigest(status="available", algorithm="md5", digest="a" * 64)
        with pytest.raises(ValidationError):
            LockfileDigest(status="available", algorithm="sha1", digest="a" * 64)

    def test_available_requires_valid_64_hex_digest(self) -> None:
        with pytest.raises(ValidationError):
            LockfileDigest(status="available", algorithm="sha256", digest="a" * 63)
        with pytest.raises(ValidationError):
            LockfileDigest(status="available", algorithm="sha256", digest="A" * 64)
        with pytest.raises(ValidationError):
            LockfileDigest(status="available", algorithm="sha256", digest=None)

    def test_available_with_sha256_and_valid_digest_accepted(self) -> None:
        lock = LockfileDigest(status="available", algorithm="sha256", digest="a" * 64)
        assert lock.algorithm == "sha256"

    def test_unavailable_requires_none_algorithm_and_digest(self) -> None:
        for status in ("unavailable", "error", "not_applicable", "redacted"):
            with pytest.raises(ValidationError):
                LockfileDigest(status=status, algorithm="sha256", digest=None)
            with pytest.raises(ValidationError):
                LockfileDigest(status=status, algorithm=None, digest="a" * 64)

    def test_unavailable_with_both_none_accepted(self) -> None:
        # unavailable / not_applicable / redacted forbid a reason (their own
        # explanation); error REQUIRES a reason.
        for status in ("unavailable", "not_applicable", "redacted"):
            lock = LockfileDigest(status=status, algorithm=None, digest=None)
            assert lock.algorithm is None
            assert lock.digest is None
        lock_err = LockfileDigest(status="error", algorithm=None, digest=None, reason="io_error")
        assert lock_err.algorithm is None
        assert lock_err.digest is None


# --- AcceleratorInfo exhaustive validator (review item 5) ------------------


class TestAcceleratorInfoExhaustiveValidator:
    """Review item 5: non-available AcceleratorInfo must have empty devices,
    None/zero device_count, and None framework/runtime/framework_version."""

    def test_unavailable_rejects_framework(self) -> None:
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="unavailable", framework="cuda")

    def test_unavailable_rejects_runtime_version(self) -> None:
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="unavailable", runtime_version="12.0")

    def test_unavailable_rejects_framework_version(self) -> None:
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="unavailable", framework_version="2.0")

    def test_error_rejects_nonzero_device_count(self) -> None:
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="error", device_count=1, reason="io_error")

    def test_error_allows_zero_or_none_device_count(self) -> None:
        a = AcceleratorInfo(status="error", device_count=0, reason="io_error")
        assert a.device_count == 0
        b = AcceleratorInfo(status="error", reason="io_error")
        assert b.device_count is None

    def test_unavailable_rejects_non_empty_devices(self) -> None:
        from expertforge.provenance.record import DeviceInfo

        with pytest.raises(ValidationError):
            AcceleratorInfo(
                status="unavailable",
                devices=(DeviceInfo(ordinal=0, model="A100"),),
            )

    def test_available_rejects_empty_devices(self) -> None:
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="available", device_count=0, devices=())


# --- CPUInfo / MemoryInfo exhaustive validator (review item 5) --------------


class TestCpuMemoryExhaustiveValidator:
    """Review item 5: CPUInfo and MemoryInfo forbid count/total_bytes on ALL
    non-available statuses (unavailable/error/not_applicable/redacted)."""

    def test_cpu_not_applicable_rejects_count(self) -> None:
        with pytest.raises(ValidationError):
            CPUInfo(status="not_applicable", count=4)

    def test_cpu_redacted_rejects_count(self) -> None:
        with pytest.raises(ValidationError):
            CPUInfo(status="redacted", count=4)

    def test_cpu_available_requires_count_ge_one(self) -> None:
        with pytest.raises(ValidationError):
            CPUInfo(status="available", count=0)

    def test_cpu_non_available_with_none_count_accepted(self) -> None:
        for status in ("unavailable", "error", "not_applicable", "redacted"):
            cpu = CPUInfo(status=status, count=None)
            assert cpu.count is None

    def test_memory_not_applicable_rejects_total_bytes(self) -> None:
        with pytest.raises(ValidationError):
            MemoryInfo(status="not_applicable", total_bytes=1024)

    def test_memory_redacted_rejects_total_bytes(self) -> None:
        with pytest.raises(ValidationError):
            MemoryInfo(status="redacted", total_bytes=1024)

    def test_memory_available_with_zero_total_bytes_accepted(self) -> None:
        # total_bytes >= 0 is valid; zero is a legitimate (if unusual) value.
        mem = MemoryInfo(status="available", total_bytes=0)
        assert mem.total_bytes == 0

    def test_memory_non_available_with_none_total_bytes_accepted(self) -> None:
        for status in ("unavailable", "error", "not_applicable", "redacted"):
            mem = MemoryInfo(status=status, total_bytes=None)
            assert mem.total_bytes is None


# --- TopologyInfo tightened validator (review item 5) ----------------------


class TestTopologyInfoTightenedValidator:
    """Review item 5: status='error'/'not_applicable' now requires ALL of rank,
    world_size, local_rank, node_count, AND backend to be None."""

    def test_error_rejects_rank(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(status="error", rank=0)

    def test_error_rejects_world_size(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(status="error", world_size=2)

    def test_not_applicable_rejects_node_count(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(status="not_applicable", node_count=1)

    def test_not_applicable_rejects_backend(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(status="not_applicable", backend="nccl")

    def test_error_with_only_reason_accepted(self) -> None:
        topo = TopologyInfo(status="error", reason="rank_and_world_size_required_together")
        assert topo.reason == "rank_and_world_size_required_together"
        assert topo.rank is None
        assert topo.backend is None

    def test_available_allows_node_count_and_backend(self) -> None:
        topo = TopologyInfo(status="available", rank=0, world_size=2, node_count=2, backend="nccl")
        assert topo.node_count == 2
        assert topo.backend == "nccl"


# --- CompletenessInfo limitations derivation/check (review item 4) ----------


class TestCompletenessLimitationsDerivation:
    """Review item 4: CompletenessInfo.limitations is now DERIVED from the
    sections and CHECKED by the consistency validator (not just status/warnings)."""

    def test_accelerator_error_propagates_limitation(self, tmp_path: Path) -> None:
        ident, snap = _identity(tmp_path)
        rec = ProvenanceRecord.from_identity(
            ident,
            source=snap,
            hardware=AcceleratorInfo(status="error", reason="io_error"),
        )
        assert "accelerator_error" in rec.completeness.limitations

    def test_default_sections_carry_unavailable_limitations(self, tmp_path: Path) -> None:
        # Default unavailable sections surface as honest limitations.
        ident, snap = _identity(tmp_path)
        rec = ProvenanceRecord.from_identity(ident, source=snap)
        limitations = rec.completeness.limitations
        assert "cpu_unavailable" in limitations
        assert "memory_unavailable" in limitations
        assert "lockfile_unavailable" in limitations
        assert "accelerator_unavailable" in limitations
        assert "topology_not_applicable" in limitations

    def test_tampered_limitations_rejected(self, tmp_path: Path) -> None:
        # A record whose stored limitations disagree with the derived set is
        # rejected by the completeness-consistency validator.
        ident, snap = _identity(tmp_path)
        rec = ProvenanceRecord.from_identity(
            ident,
            source=snap,
            hardware=AcceleratorInfo(status="error", reason="io_error"),
        )
        tampered = rec.model_dump()
        # Drop the accelerator_error limitation while keeping the rest.
        tampered["completeness"]["limitations"] = tuple(
            lim for lim in rec.completeness.limitations if lim != "accelerator_error"
        )
        with pytest.raises(ValidationError):
            ProvenanceRecord.model_validate(tampered)

    def test_topology_warnings_appear_in_limitations(self, tmp_path: Path) -> None:
        ident, snap = _identity(tmp_path)
        topo = TopologyInfo(
            status="available",
            rank=0,
            world_size=2,
            topology_warnings=("invalid_topology_env_value:RANK",),
        )
        rec = ProvenanceRecord.from_identity(ident, source=snap, topology=topo)
        assert "invalid_topology_env_value:RANK" in rec.completeness.limitations


# --- stable reason/warning domains (review item 3) -------------------------


class TestLockfileDigestReasonDomain:
    """Review item 3: ``LockfileDigest.reason`` is a Literal domain
    (``"io_error"`` | ``"not_found"`` | ``None``). Free-text reasons are
    rejected. Cross-field: available forbids a reason; not_applicable/redacted
    forbid a reason."""

    def test_io_error_reason_accepted(self) -> None:
        lock = LockfileDigest(status="error", reason="io_error")
        assert lock.reason == "io_error"

    def test_not_found_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LockfileDigest(status="error", reason="not_found")  # type: ignore[arg-type]

    def test_unknown_reason_rejected(self) -> None:
        bad_reason: str = "disk_full"
        with pytest.raises(ValidationError):
            LockfileDigest(status="error", reason=bad_reason)  # type: ignore[arg-type]

    def test_available_forbids_reason(self) -> None:
        with pytest.raises(ValidationError):
            LockfileDigest(
                status="available", algorithm="sha256", digest="a" * 64, reason="io_error"
            )

    def test_not_applicable_forbids_reason(self) -> None:
        with pytest.raises(ValidationError):
            LockfileDigest(status="not_applicable", reason="io_error")

    def test_redacted_forbids_reason(self) -> None:
        with pytest.raises(ValidationError):
            LockfileDigest(status="redacted", reason="io_error")

    def test_unavailable_none_reason_accepted(self) -> None:
        lock = LockfileDigest(status="unavailable", reason=None)
        assert lock.reason is None


class TestAcceleratorInfoReasonDomain:
    """Review item 3: ``AcceleratorInfo.reason`` is a Literal domain
    (``"io_error"`` | ``"timeout"`` | ``"decode_error"`` |
    ``"duplicate_device_ordinals"`` | ``"not_found"`` | ``None``).
    Cross-field: available forbids a reason; ``precision_status="available"``
    with a non-available status is rejected."""

    def test_known_reasons_accepted(self) -> None:
        assert AcceleratorInfo(status="error", reason="io_error").reason == "io_error"
        assert AcceleratorInfo(status="error", reason="timeout").reason == "timeout"
        assert AcceleratorInfo(status="error", reason="decode_error").reason == "decode_error"

    def test_duplicate_device_ordinals_reason_accepted(self) -> None:
        accel = AcceleratorInfo(status="error", reason="duplicate_device_ordinals")
        assert accel.reason == "duplicate_device_ordinals"

    def test_unknown_reason_rejected(self) -> None:
        bad_reason: str = "out_of_memory"
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="error", reason=bad_reason)  # type: ignore[arg-type]

    def test_available_forbids_reason(self) -> None:
        from expertforge.provenance.record import DeviceInfo

        with pytest.raises(ValidationError):
            AcceleratorInfo(
                status="available",
                device_count=1,
                devices=(DeviceInfo(ordinal=0, model="A100"),),
                reason="io_error",
            )

    def test_precision_available_requires_status_available(self) -> None:
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="unavailable", precision_status="available")

    def test_precision_available_with_error_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="error", precision_status="available", reason="io_error")

    def test_precision_not_applicable_with_unavailable_accepted(self) -> None:
        accel = AcceleratorInfo(status="unavailable", precision_status="not_applicable")
        assert accel.precision_status == "not_applicable"


class TestTopologyInfoReasonDomain:
    """Review item 3: ``TopologyInfo.reason`` is a Literal domain.
    Cross-field: available forbids a reason; not_applicable forbids a reason."""

    def test_known_reasons_accepted(self) -> None:
        for r in (
            "rank_and_world_size_required_together",
            "partial_topology_env_without_rank_world_size",
            "rank_must_be_less_than_world_size",
            "partial_explicit_input_without_rank_world_size",
            "node_count_must_be_positive",
            "invalid_local_rank",
        ):
            topo = TopologyInfo(status="error", reason=r)
            assert topo.reason == r

    def test_unknown_reason_rejected(self) -> None:
        bad_reason: str = "custom_reason"
        with pytest.raises(ValidationError):
            TopologyInfo(status="error", reason=bad_reason)  # type: ignore[arg-type]

    def test_available_forbids_reason(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(status="available", rank=0, world_size=2, reason="invalid_local_rank")

    def test_not_applicable_forbids_reason(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(status="not_applicable", reason="invalid_local_rank")


class TestReasonStatusBinding:
    """Review item 3: reason domains are exhaustively bound to status. An error
    status REQUIRES a reason (an error without one is under-constrained); every
    non-available/non-error status FORBIDS a reason (those statuses are their
    own explanation)."""

    # --- LockfileDigest ---------------------------------------------------

    def test_lockfile_error_without_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LockfileDigest(status="error", reason=None)

    def test_lockfile_error_with_reason_accepted(self) -> None:
        lock = LockfileDigest(status="error", reason="io_error")
        assert lock.reason == "io_error"

    def test_lockfile_unavailable_with_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LockfileDigest(
                status="unavailable",
                reason="not_found",  # type: ignore[arg-type]
            )

    def test_lockfile_not_applicable_with_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LockfileDigest(status="not_applicable", reason="io_error")

    def test_lockfile_redacted_with_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LockfileDigest(status="redacted", reason="io_error")

    # --- AcceleratorInfo --------------------------------------------------

    def test_accelerator_error_without_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="error", reason=None)

    def test_accelerator_error_with_reason_accepted(self) -> None:
        assert AcceleratorInfo(status="error", reason="io_error").reason == "io_error"
        assert AcceleratorInfo(status="error", reason="timeout").reason == "timeout"
        assert AcceleratorInfo(status="error", reason="decode_error").reason == "decode_error"
        assert (
            AcceleratorInfo(status="error", reason="duplicate_device_ordinals").reason
            == "duplicate_device_ordinals"
        )

    def test_accelerator_unavailable_with_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="unavailable", reason="io_error")

    def test_accelerator_not_applicable_with_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="not_applicable", reason="io_error")

    def test_accelerator_redacted_with_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="redacted", reason="io_error")

    # --- TopologyInfo -----------------------------------------------------

    def test_topology_error_without_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(status="error", reason=None)

    def test_topology_error_with_reason_accepted(self) -> None:
        topo = TopologyInfo(status="error", reason="node_count_must_be_positive")
        assert topo.reason == "node_count_must_be_positive"

    def test_topology_not_applicable_with_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(status="not_applicable", reason="invalid_local_rank")


class TestTopologyInfoWarningPattern:
    """Review item 3: ``TopologyInfo.topology_warnings`` entries must match
    ``invalid_topology_env_value:<FIELD>`` where FIELD is one of the allowed
    topology env var names."""

    def test_valid_fields_accepted(self) -> None:
        for field in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "NODE_RANK",
            "NNODES",
        ):
            topo = TopologyInfo(
                status="available",
                rank=0,
                world_size=2,
                topology_warnings=(f"invalid_topology_env_value:{field}",),
            )
            assert topo.topology_warnings == (f"invalid_topology_env_value:{field}",)

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(
                status="available",
                rank=0,
                world_size=2,
                topology_warnings=("invalid_topology_env_value:UNKNOWN",),
            )

    def test_lowercase_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(
                status="available",
                rank=0,
                world_size=2,
                topology_warnings=("invalid_topology_env_value:rank",),
            )

    def test_free_text_warning_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(
                status="available",
                rank=0,
                world_size=2,
                topology_warnings=("custom_warning",),
            )

    def test_wrong_prefix_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(
                status="available",
                rank=0,
                world_size=2,
                topology_warnings=("invalid_env_value:RANK",),
            )


class TestDependencyConflictModel:
    """Review item 3: ``SoftwareEnvironment.dependency_conflicts`` is now a
    tuple of typed ``DependencyConflict`` models (name: PEP 503 normalized,
    observed_versions: sorted unique, >= 2 versions)."""

    def test_dependency_conflict_accepted(self) -> None:
        from expertforge.provenance.record import (
            DependencyConflict,
            PythonInfo,
            SoftwareEnvironment,
        )

        env = SoftwareEnvironment(
            python=PythonInfo(version="3.11", implementation="cpython"),
            dependency_conflicts=(
                DependencyConflict(name="numpy", observed_versions=("1.0.0", "2.0.0")),
            ),
        )
        assert env.dependency_conflicts[0].name == "numpy"
        assert env.dependency_conflicts[0].observed_versions == ("1.0.0", "2.0.0")

    def test_dependency_conflict_normalizes_name(self) -> None:
        from expertforge.provenance.record import DependencyConflict

        conflict = DependencyConflict(name="My_Awesome.Pkg", observed_versions=("1.0", "2.0"))
        assert conflict.name == "my-awesome-pkg"

    def test_dependency_conflict_single_version_rejected(self) -> None:
        from expertforge.provenance.record import DependencyConflict

        with pytest.raises(ValidationError):
            DependencyConflict(name="numpy", observed_versions=("1.0.0",))

    def test_dependency_conflict_unsorted_versions_rejected(self) -> None:
        from expertforge.provenance.record import DependencyConflict

        with pytest.raises(ValidationError):
            DependencyConflict(name="numpy", observed_versions=("2.0.0", "1.0.0"))

    def test_dependency_conflict_duplicate_versions_rejected(self) -> None:
        from expertforge.provenance.record import DependencyConflict

        with pytest.raises(ValidationError):
            DependencyConflict(name="numpy", observed_versions=("1.0.0", "1.0.0"))

    def test_dependency_conflict_invalid_name_rejected(self) -> None:
        from expertforge.provenance.record import DependencyConflict

        with pytest.raises(ValidationError):
            DependencyConflict(name="-leading-dash", observed_versions=("1.0.0", "2.0.0"))

    def test_dependency_conflicts_round_trip(self) -> None:
        from expertforge.provenance.record import (
            DependencyConflict,
            PythonInfo,
            SoftwareEnvironment,
        )

        env = SoftwareEnvironment(
            python=PythonInfo(version="3.11", implementation="cpython"),
            dependency_conflicts=(
                DependencyConflict(name="alpha", observed_versions=("1.0", "2.0")),
                DependencyConflict(name="beta", observed_versions=("0.9", "1.0", "1.1")),
            ),
        )
        restored = SoftwareEnvironment.model_validate_json(env.model_dump_json())
        assert restored.dependency_conflicts == env.dependency_conflicts
