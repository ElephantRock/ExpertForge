"""Tests for the identity emit orchestrator and the cross-cutting versioning
policy (Issue #6 decision §5, §versioning; review item 1).

The orchestrator wires fingerprint → spec-prefix → run_id → attempt_id →
record → sidecar via explicit allocation modes (INDEPENDENT / RESUME / FORK /
LEGACY), preserving the run-versus-attempt contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from expertforge.config.resolve import ResolutionEnvelope, resolve_config
from expertforge.identity.emit import (
    AllocationMode,
    IdentityEmitError,
    emit_attempt_identity,
)
from expertforge.identity.record import AttemptIdentityRecord

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

_VALID_RUN = "run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb"
_VALID_ATTEMPT = "attempt-20260101t000000z-cccccccccccccccccccc"
_FIXED = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _env() -> ResolutionEnvelope:
    return resolve_config(CONFIGS / "smoke.yaml")


class TestIndependentMode:
    def test_independent_writes_sidecar_and_returns_record(self, tmp_path: Path) -> None:
        record, path = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=_env(),
            clock=lambda: _FIXED,
            entropy=lambda n: bytes(n),
        )
        assert isinstance(record, AttemptIdentityRecord)
        assert path.exists()
        assert (
            path == tmp_path / record.run_id / "attempts" / record.attempt_id / "run-identity.json"
        )

    def test_independent_record_carries_fingerprint_run_attempt(self, tmp_path: Path) -> None:
        record, _ = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=_env(),
            clock=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
            entropy=lambda n: bytes(n),
        )
        assert record.fingerprint_digest_str().startswith("spec-v1-sha256-")
        assert record.run_id.startswith("run-20260102t030405z-")
        assert record.attempt_id.startswith("attempt-20260102t030405z-")
        assert record.lineage is None

    def test_independent_rejects_lineage(self, tmp_path: Path) -> None:
        from expertforge.identity.lineage import ResumeLineage

        lin = ResumeLineage(
            parent_run_id=_VALID_RUN, parent_attempt_id=_VALID_ATTEMPT, parent_checkpoint_id="c"
        )
        with pytest.raises(IdentityEmitError):
            emit_attempt_identity(
                artifact_root=tmp_path,
                config_envelope=_env(),
                mode=AllocationMode.INDEPENDENT,
                lineage=lin,
                clock=lambda: _FIXED,
                entropy=lambda n: bytes(n),
            )

    def test_two_independent_runs_distinct_ids_common_fingerprint(self, tmp_path: Path) -> None:
        r1, _ = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=_env(),
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x01" * n,
        )
        r2, _ = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=_env(),
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x02" * n,
        )
        assert r1.run_id != r2.run_id
        assert r1.attempt_id != r2.attempt_id
        assert r1.fingerprint_digest_str() == r2.fingerprint_digest_str()


class TestResumeMode:
    def test_resume_retains_run_id_new_attempt(self, tmp_path: Path) -> None:
        from expertforge.identity.lineage import ResumeLineage

        # First, an independent run to get a real run_id and spec fingerprint.
        first, _ = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=_env(),
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x01" * n,
        )
        lin = ResumeLineage(
            parent_run_id=first.run_id,
            parent_attempt_id=first.attempt_id,
            parent_checkpoint_id="ckpt-001",
        )
        resumed, path = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=_env(),
            mode=AllocationMode.RESUME,
            lineage=lin,
            parent_specification_fingerprint=first.fingerprint_digest_str(),
            retained_run_id=first.run_id,
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x02" * n,
        )
        # Run ID retained; new attempt ID.
        assert resumed.run_id == first.run_id
        assert resumed.attempt_id != first.attempt_id
        # Same spec fingerprint.
        assert resumed.fingerprint_digest_str() == first.fingerprint_digest_str()
        # Full native lineage recorded.
        assert resumed.lineage is not None
        assert resumed.lineage.parent_run_id == first.run_id
        assert resumed.lineage.parent_attempt_id == first.attempt_id
        assert resumed.lineage.parent_checkpoint_id == "ckpt-001"

    def test_resume_requires_parent_attempt_id(self, tmp_path: Path) -> None:
        from expertforge.identity.lineage import ResumeLineage

        lin = ResumeLineage(
            parent_run_id=_VALID_RUN, parent_attempt_id=None, parent_checkpoint_id="c"
        )
        with pytest.raises(IdentityEmitError):
            emit_attempt_identity(
                artifact_root=tmp_path,
                config_envelope=_env(),
                mode=AllocationMode.RESUME,
                lineage=lin,
                parent_specification_fingerprint="spec-v1-sha256-" + "0" * 64,
                retained_run_id=_VALID_RUN,
                clock=lambda: _FIXED,
                entropy=lambda n: bytes(n),
            )

    def test_resume_rejects_spec_mismatch(self, tmp_path: Path) -> None:
        from expertforge.identity.lineage import ResumeLineage

        lin = ResumeLineage(
            parent_run_id=_VALID_RUN, parent_attempt_id=_VALID_ATTEMPT, parent_checkpoint_id="c"
        )
        with pytest.raises(IdentityEmitError):
            emit_attempt_identity(
                artifact_root=tmp_path,
                config_envelope=_env(),
                mode=AllocationMode.RESUME,
                lineage=lin,
                parent_specification_fingerprint="spec-v1-sha256-" + "0" * 64,  # wrong
                retained_run_id=_VALID_RUN,
                clock=lambda: _FIXED,
                entropy=lambda n: bytes(n),
            )

    def test_resume_lineage_parent_run_must_equal_retained(self, tmp_path: Path) -> None:
        from expertforge.identity.lineage import ResumeLineage

        lin = ResumeLineage(
            parent_run_id=_VALID_RUN, parent_attempt_id=_VALID_ATTEMPT, parent_checkpoint_id="c"
        )
        with pytest.raises(IdentityEmitError):
            emit_attempt_identity(
                artifact_root=tmp_path,
                config_envelope=_env(),
                mode=AllocationMode.RESUME,
                lineage=lin,
                parent_specification_fingerprint="spec-v1-sha256-" + "0" * 64,
                retained_run_id="run-20260101t000000z-dddddddddddd-eeeeeeeeeeeeeeeeeeee",
                clock=lambda: _FIXED,
                entropy=lambda n: bytes(n),
            )


class TestForkMode:
    def test_fork_new_run_id_with_parent_lineage(self, tmp_path: Path) -> None:
        from expertforge.identity.lineage import ResumeLineage

        first, _ = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=_env(),
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x01" * n,
        )
        # Materially changed spec (different seed) -> different fingerprint.
        changed = resolve_config(CONFIGS / "smoke.yaml", ["training.seed=999"])
        lin = ResumeLineage(
            parent_run_id=first.run_id,
            parent_attempt_id=first.attempt_id,
            parent_checkpoint_id="ckpt-001",
        )
        forked, _ = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=changed,
            mode=AllocationMode.FORK,
            lineage=lin,
            parent_specification_fingerprint=first.fingerprint_digest_str(),
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x03" * n,
        )
        # New run ID (fork), new attempt, parent lineage recorded.
        assert forked.run_id != first.run_id
        assert forked.fingerprint_digest_str() != first.fingerprint_digest_str()
        assert forked.lineage is not None
        assert forked.lineage.parent_run_id == first.run_id

    def test_fork_rejects_retained_run_id(self, tmp_path: Path) -> None:
        from expertforge.identity.lineage import ResumeLineage

        lin = ResumeLineage(
            parent_run_id=_VALID_RUN, parent_attempt_id=_VALID_ATTEMPT, parent_checkpoint_id="c"
        )
        with pytest.raises(IdentityEmitError):
            emit_attempt_identity(
                artifact_root=tmp_path,
                config_envelope=_env(),
                mode=AllocationMode.FORK,
                lineage=lin,
                parent_specification_fingerprint="spec-v1-sha256-" + "0" * 64,
                retained_run_id=_VALID_RUN,
                clock=lambda: _FIXED,
                entropy=lambda n: bytes(n),
            )


class TestLegacyMode:
    def test_legacy_allows_missing_parent_attempt_id(self, tmp_path: Path) -> None:
        from expertforge.identity.lineage import ResumeLineage

        first, _ = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=_env(),
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x01" * n,
        )
        lin = ResumeLineage(
            parent_run_id=first.run_id, parent_attempt_id=None, parent_checkpoint_id="legacy-ckpt"
        )
        legacy, _ = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=_env(),
            mode=AllocationMode.LEGACY,
            lineage=lin,
            parent_specification_fingerprint=first.fingerprint_digest_str(),
            retained_run_id=first.run_id,
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x04" * n,
        )
        assert legacy.run_id == first.run_id
        assert legacy.lineage is not None
        assert legacy.lineage.parent_attempt_id is None


class TestReconstructionAndErrors:
    def test_emit_reconstructs_from_sidecar(self, tmp_path: Path) -> None:
        from expertforge.identity.sidecar import load_identity_sidecar

        record, path = emit_attempt_identity(
            artifact_root=tmp_path,
            config_envelope=_env(),
            clock=lambda: _FIXED,
            entropy=lambda n: bytes(n),
        )
        loaded = load_identity_sidecar(path, expected_fingerprint=record.fingerprint_digest_str())
        assert loaded == record

    def test_emit_collision_exhaustion_propagates(self, tmp_path: Path) -> None:
        with pytest.raises(IdentityEmitError):
            emit_attempt_identity(
                artifact_root=tmp_path,
                config_envelope=_env(),
                clock=lambda: _FIXED,
                entropy=lambda n: bytes(n),
                exists=lambda _id: True,
                max_retries=3,
            )

    def test_naive_clock_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(IdentityEmitError):
            emit_attempt_identity(
                artifact_root=tmp_path,
                config_envelope=_env(),
                clock=lambda: datetime(2026, 1, 1, 0, 0, 0),  # naive
                entropy=lambda n: bytes(n),
            )


class TestVersioningIndependence:
    def test_fingerprint_version_independent_of_schema_version(self) -> None:
        from expertforge.identity.fingerprint import FINGERPRINT_VERSION
        from expertforge.identity.record import IDENTITY_SCHEMA_VERSION

        assert FINGERPRINT_VERSION == 1
        assert IDENTITY_SCHEMA_VERSION == 1
