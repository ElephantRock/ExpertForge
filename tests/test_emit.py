"""Tests for the identity emit orchestrator and the cross-cutting versioning
policy (Issue #6 decision §5, §versioning).

The orchestrator wires fingerprint → spec-prefix → run_id → attempt_id →
record → sidecar, so a caller can emit a complete, durable, reconstructable
identity for a resolved configuration in one step.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from expertforge.config.resolve import resolve_config
from expertforge.identity.emit import (
    IdentityEmitError,
    emit_attempt_identity,
)
from expertforge.identity.record import AttemptIdentityRecord

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class TestEmitAttemptIdentity:
    def test_emit_writes_sidecar_and_returns_record(self, tmp_path: Path) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        record, path = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=env,
            clock=lambda: fixed,
            entropy=lambda n: bytes(n),
        )
        assert isinstance(record, AttemptIdentityRecord)
        assert path.exists()
        # Sidecar path follows the canonical layout.
        assert (
            path == tmp_path / record.run_id / "attempts" / record.attempt_id / "run-identity.json"
        )

    def test_emit_record_carries_fingerprint_run_attempt(self, tmp_path: Path) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fixed = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        record, _ = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=env,
            clock=lambda: fixed,
            entropy=lambda n: bytes(n),
        )
        assert record.specification_fingerprint.startswith("spec-v1-sha256-")
        assert record.run_id.startswith("run-20260102t030405z-")
        assert record.attempt_id.startswith("attempt-20260102t030405z-")
        assert record.lineage is None  # first attempt

    def test_emit_with_resume_lineage(self, tmp_path: Path) -> None:
        from expertforge.identity.lineage import ResumeLineage

        env = resolve_config(CONFIGS / "smoke.yaml")
        fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        lin = ResumeLineage(
            parent_run_id="run-parent",
            parent_attempt_id="attempt-parent",
            parent_checkpoint_id="ckpt-parent",
        )
        record, _ = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=env,
            lineage=lin,
            clock=lambda: fixed,
            entropy=lambda n: bytes(n),
        )
        assert record.lineage is not None
        assert record.lineage.parent_run_id == "run-parent"

    def test_emit_two_attempts_same_run_share_spec_fingerprint(self, tmp_path: Path) -> None:
        # Two executions of the same spec get distinct run identities but the
        # same specification fingerprint.
        env = resolve_config(CONFIGS / "smoke.yaml")
        fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        r1, _ = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=env,
            clock=lambda: fixed,
            entropy=lambda n: b"\x01" * n,
        )
        r2, _ = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=env,
            clock=lambda: fixed,
            entropy=lambda n: b"\x02" * n,
        )
        assert r1.run_id != r2.run_id
        assert r1.attempt_id != r2.attempt_id
        assert r1.specification_fingerprint == r2.specification_fingerprint

    def test_emit_reconstructs_from_sidecar(self, tmp_path: Path) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        record, path = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=env,
            clock=lambda: fixed,
            entropy=lambda n: bytes(n),
        )
        from expertforge.identity.sidecar import load_identity_sidecar

        loaded = load_identity_sidecar(path, expected_fingerprint=record.specification_fingerprint)
        assert loaded == record

    def test_emit_collision_exhaustion_propagates(self, tmp_path: Path) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        with pytest.raises(IdentityEmitError):
            emit_attempt_identity(
                artifact_root=tmp_path,
                config_envelope=env,
                clock=lambda: fixed,
                entropy=lambda n: bytes(n),  # constant
                exists=lambda _id: True,  # every candidate collides
                max_retries=3,
            )


class TestVersioningIndependence:
    def test_fingerprint_version_independent_of_schema_version(self) -> None:
        from expertforge.identity.fingerprint import FINGERPRINT_VERSION
        from expertforge.identity.record import IDENTITY_SCHEMA_VERSION

        # Both are 1 today but are conceptually independent; the test documents
        # that they live in separate modules and can diverge.
        assert FINGERPRINT_VERSION == 1
        assert IDENTITY_SCHEMA_VERSION == 1
