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
    LockfileDigest,
    MemoryInfo,
    PlatformInfo,
    ProvenanceRecord,
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
