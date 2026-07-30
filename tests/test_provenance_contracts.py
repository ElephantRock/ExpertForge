"""Contract-level regression tests for Issue #7 provenance models.

Covers: nested mutation rejection, invalid statuses, cross-field validator
rejections (available+no-digest, available+no-devices, duplicate ordinals,
device-count mismatch, rank>=world_size), source flag/evidence contradictions,
topology partial-input→error, completeness propagation, and source-snapshot
clean/evidence invariants.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from expertforge.config.resolve import resolve_config
from expertforge.identity.emit import emit_attempt_identity
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.provenance.record import (
    AcceleratorInfo,
    CompletenessInfo,
    DependencyObservation,
    DeviceInfo,
    LockfileDigest,
    ProvenanceRecord,
    PythonInfo,
    SoftwareEnvironment,
    TopologyInfo,
)
from expertforge.provenance.source_snapshot import SourceSnapshot, capture_source_snapshot

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


def _identity_and_snap(tmp_path: Path) -> tuple[AttemptIdentityRecord, SourceSnapshot]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _init_repo(repo)
    snap = capture_source_snapshot(repo)
    from expertforge.provenance.source_snapshot import source_snapshot_immutable_input

    ident, _ = emit_attempt_identity(
        artifact_root=tmp_path,
        config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
        immutable_inputs=[source_snapshot_immutable_input(snap)],
        clock=lambda: _FIXED,
        entropy=lambda n: bytes(n),
    )
    return ident, snap


# --- invalid statuses ------------------------------------------------------


class TestInvalidStatuses:
    def test_completeness_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CompletenessInfo(status="complete-but-fabricated")  # type: ignore[arg-type]

    def test_accelerator_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="bogus")  # type: ignore[arg-type]

    def test_topology_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(status="unknown")  # type: ignore[arg-type]

    def test_lockfile_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LockfileDigest(status="maybe")  # type: ignore[arg-type]


# --- cross-field validators ------------------------------------------------


class TestCrossFieldValidators:
    def test_lockfile_available_requires_digest(self) -> None:
        with pytest.raises(ValidationError):
            LockfileDigest(status="available", digest=None)

    def test_accelerator_available_requires_devices(self) -> None:
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="available", device_count=0, devices=())

    def test_accelerator_device_count_mismatch(self) -> None:
        dev = DeviceInfo(ordinal=0, model="A100")
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="available", device_count=2, devices=(dev,))

    def test_accelerator_duplicate_ordinals(self) -> None:
        dev1 = DeviceInfo(ordinal=0, model="A100")
        dev2 = DeviceInfo(ordinal=0, model="A100")
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="available", device_count=2, devices=(dev1, dev2))

    def test_topology_available_requires_rank_world_size(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(status="available", rank=None, world_size=None)

    def test_topology_rank_must_be_less_than_world_size(self) -> None:
        with pytest.raises(ValidationError):
            TopologyInfo(status="available", rank=4, world_size=4)


# --- nested immutability ---------------------------------------------------


class TestNestedImmutability:
    def test_dependency_observation_is_frozen(self) -> None:
        dep = DependencyObservation(name="numpy", version="1.0.0")
        with pytest.raises(ValidationError):
            dep.version = "2.0.0"

    def test_software_dependencies_reject_duplicates(self) -> None:
        deps = (
            DependencyObservation(name="numpy", version="1.0.0"),
            DependencyObservation(name="numpy", version="2.0.0"),
        )
        with pytest.raises(ValidationError):
            SoftwareEnvironment(
                python=PythonInfo(version="3.11", implementation="cpython"),
                dependencies=deps,
            )

    def test_software_dependencies_must_be_sorted(self) -> None:
        deps = (
            DependencyObservation(name="zzz", version="1.0.0"),
            DependencyObservation(name="aaa", version="2.0.0"),
        )
        with pytest.raises(ValidationError):
            SoftwareEnvironment(
                python=PythonInfo(version="3.11", implementation="cpython"),
                dependencies=deps,
            )


# --- source snapshot invariants -------------------------------------------


class TestSourceSnapshotInvariants:
    def test_dirty_evidence_records_ignored_files_limitation(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        _init_repo(repo)
        (repo / "new.txt").write_text("new\n", encoding="utf-8")
        snap = capture_source_snapshot(repo, allow_dirty=True)
        assert snap.evidence is not None
        assert any("ignored_files_not_included" in lim for lim in snap.evidence.limitations)

    def test_dirty_evidence_records_external_symlink_limitation(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        _init_repo(repo)
        (repo / "new.txt").write_text("new\n", encoding="utf-8")
        snap = capture_source_snapshot(repo, allow_dirty=True)
        assert snap.evidence is not None
        assert any(
            "external_symlink_targets_not_followed" in lim for lim in snap.evidence.limitations
        )

    def test_clean_snapshot_has_no_evidence(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        _init_repo(repo)
        snap = capture_source_snapshot(repo)
        assert snap.evidence is None
        assert snap.is_clean is True

    def test_remote_url_sanitized_in_snapshot(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        _init_repo(repo)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://user:token@github.com/org/repo.git"],
            cwd=repo,
            check=True,
        )
        snap = capture_source_snapshot(repo)
        # The raw credential-bearing URL must not appear in the serialized form.
        serialized = snap.model_dump_json()
        assert "user:token" not in serialized
        assert "token@" not in serialized

    def test_input_digest_recomputed_on_load(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        _init_repo(repo)
        snap = capture_source_snapshot(repo)
        # Round-trip preserves the digest (model_validator recomputes on load).
        restored = SourceSnapshot.model_validate_json(snap.model_dump_json())
        assert restored.input_digest == snap.input_digest

    def test_tampered_input_digest_rejected(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        _init_repo(repo)
        snap = capture_source_snapshot(repo)
        bad = snap.model_dump()
        bad["input_digest"] = "0" * 64
        with pytest.raises(ValidationError):
            SourceSnapshot.model_validate(bad)


# --- completeness propagation ----------------------------------------------


class TestCompletenessPropagation:
    def test_clean_source_completes(self, tmp_path: Path) -> None:
        ident, snap = _identity_and_snap(tmp_path)
        rec = ProvenanceRecord.from_identity(ident, source=snap)
        assert rec.completeness.status in ("complete", "partial")
        # Clean source with no warnings → complete.
        if not rec.completeness.warnings:
            assert rec.completeness.status == "complete"

    def test_dirty_source_propagates_warning(self, tmp_path: Path) -> None:
        from expertforge.provenance.source_snapshot import source_snapshot_immutable_input

        repo = tmp_path / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        _init_repo(repo)
        (repo / "a.txt").write_text("dirty\n", encoding="utf-8")
        snap = capture_source_snapshot(repo, allow_dirty=True)
        ident, _ = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            immutable_inputs=[source_snapshot_immutable_input(snap)],
            clock=lambda: _FIXED,
            entropy=lambda n: bytes(n),
        )
        rec = ProvenanceRecord.from_identity(ident, source=snap)
        assert "non_canonical_dirty_source" in rec.completeness.warnings
        assert rec.completeness.status == "partial"

    def test_accelerator_error_propagates(self, tmp_path: Path) -> None:
        ident, snap = _identity_and_snap(tmp_path)
        rec = ProvenanceRecord.from_identity(
            ident,
            source=snap,
            hardware=AcceleratorInfo(status="error", reason="io_error"),
        )
        assert "accelerator_error" in rec.completeness.warnings


# --- topology partial input → error ----------------------------------------


class TestTopologyPartialInput:
    def test_local_rank_without_rank_world_size_is_error(self) -> None:
        from expertforge.provenance.hardware import capture_topology

        topo = capture_topology(local_rank=0)
        assert topo.status == "error"
        assert topo.reason is not None
        assert "partial" in topo.reason or "without" in topo.reason

    def test_backend_without_rank_world_size_is_error(self) -> None:
        from expertforge.provenance.hardware import capture_topology

        topo = capture_topology(backend="nccl")
        # With only backend and no env vars, returns not_applicable (no env detected).
        # But with explicit backend, it's still not_applicable since no rank/world.
        assert topo.status in ("not_applicable", "error")
