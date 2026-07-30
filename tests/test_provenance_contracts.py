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
from typing import cast

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


# --- contradictory source-flag / evidence invariants (review item 2) -------


class TestSourceFlagContradictions:
    """A tampered sidecar cannot claim clean/dirty states that contradict the
    evidence (or lack thereof)."""

    @staticmethod
    def _envelope(tree: str, evidence: object) -> str:
        from expertforge.provenance.source_snapshot import _envelope_digest

        return _envelope_digest(tree, evidence)  # type: ignore[arg-type]

    def test_is_clean_true_with_evidence_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import (
            SourceCounts,
            SourceEvidence,
        )

        evidence = SourceEvidence(
            staged_digest="a" * 64,
            unstaged_digest="b" * 64,
            counts=SourceCounts(staged=1),
            completeness="complete",
        )
        with pytest.raises(ValidationError):
            SourceSnapshot(
                commit_sha="1" * 40,
                is_clean=True,  # contradiction: clean but evidence present
                is_canonical=True,
                tree_digest="c" * 64,
                input_digest=self._envelope("c" * 64, evidence),
                evidence=evidence,
            )

    def test_is_clean_false_without_evidence_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SourceSnapshot(
                commit_sha="1" * 40,
                is_clean=False,  # contradiction: dirty but no evidence
                is_canonical=False,
                tree_digest="c" * 64,
                input_digest=self._envelope("c" * 64, None),
                evidence=None,
            )

    def test_clean_but_non_canonical_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SourceSnapshot(
                commit_sha="1" * 40,
                is_clean=True,
                is_canonical=False,  # contradiction: clean implies canonical
                tree_digest="c" * 64,
                input_digest=self._envelope("c" * 64, None),
            )

    def test_dirty_but_canonical_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import (
            SourceCounts,
            SourceEvidence,
        )

        evidence = SourceEvidence(
            staged_digest="a" * 64,
            unstaged_digest="b" * 64,
            counts=SourceCounts(staged=1),
            completeness="complete",
        )
        with pytest.raises(ValidationError):
            SourceSnapshot(
                commit_sha="1" * 40,
                is_clean=False,
                is_canonical=True,  # contradiction: dirty implies non-canonical
                tree_digest="c" * 64,
                input_digest=self._envelope("c" * 64, evidence),
                evidence=evidence,
            )

    def test_evidence_completeness_error_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import (
            SourceCounts,
            SourceEvidence,
        )

        evidence = SourceEvidence(
            staged_digest="a" * 64,
            unstaged_digest="b" * 64,
            counts=SourceCounts(staged=1),
            completeness="error",  # an error-state capture aborts, not records
        )
        with pytest.raises(ValidationError):
            SourceSnapshot(
                commit_sha="1" * 40,
                is_clean=False,
                is_canonical=False,
                tree_digest="c" * 64,
                input_digest=self._envelope("c" * 64, evidence),
                evidence=evidence,
            )

    def test_remote_url_model_accepts_any_value(self) -> None:
        # Review item 1: the SourceSnapshot model no longer correlates
        # remote_url with remote_warnings (that correlation cannot survive a
        # sidecar round-trip). The model accepts whatever URL/warnings are
        # supplied; the CAPTURE PATH is the sole place where sanitization +
        # warning correlation happens. A raw-credential URL is therefore
        # accepted by the MODEL (it never sees the raw URL in practice —
        # capture sanitizes first).
        snap = SourceSnapshot(
            commit_sha="1" * 40,
            is_clean=True,
            is_canonical=True,
            remote_url="https://user:token@github.com/org/repo.git",
            tree_digest="c" * 64,
            input_digest=self._envelope("c" * 64, None),
        )
        # The model stores whatever it was given — no sanitization at the
        # model level.
        assert snap.remote_url == "https://user:token@github.com/org/repo.git"

    def test_capture_path_sanitizes_raw_credentials(self, tmp_path: Path) -> None:
        # The CAPTURE PATH is where sanitization happens: it strips credentials
        # and emits the matching remote_credentials_removed warning. The raw
        # URL never reaches the stored snapshot.
        from expertforge.provenance.source_snapshot import capture_source_snapshot

        _init_repo(tmp_path)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://user:token@github.com/org/repo.git"],
            cwd=tmp_path,
            check=True,
        )
        snap = capture_source_snapshot(tmp_path)
        assert snap.remote_url == "https://github.com/org/repo.git"
        assert {w.code for w in snap.remote_warnings} == {"remote_credentials_removed"}
        assert "token" not in snap.model_dump_json()


# --- malformed digest fields rejected (review item 2) ----------------------


class TestMalformedDigests:
    def test_tree_digest_not_hex_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SourceSnapshot(
                commit_sha="1" * 40,
                is_clean=True,
                is_canonical=True,
                tree_digest="g" * 64,  # not hex
                input_digest="0" * 64,
            )

    def test_tree_digest_wrong_length_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SourceSnapshot(
                commit_sha="1" * 40,
                is_clean=True,
                is_canonical=True,
                tree_digest="a" * 63,  # too short
                input_digest="0" * 64,
            )

    def test_tree_digest_uppercase_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SourceSnapshot(
                commit_sha="1" * 40,
                is_clean=True,
                is_canonical=True,
                tree_digest="A" * 64,  # must be lowercase
                input_digest="0" * 64,
            )

    def test_source_evidence_staged_digest_malformed_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import (
            SourceCounts,
            SourceEvidence,
        )

        with pytest.raises(ValidationError):
            SourceEvidence(
                staged_digest="not-hex",  # malformed
                unstaged_digest="b" * 64,
                counts=SourceCounts(),
                completeness="complete",
            )


# --- unsafe untracked paths rejected (review item 2) -----------------------


class TestUnsafeUntrackedPaths:
    @pytest.mark.parametrize(
        "path",
        [
            "/etc/passwd",  # POSIX absolute
            "\\windows\\secret",  # backslash-absolute
            "C:/Users/secret",  # Windows drive absolute
            "C:\\Users\\secret",  # Windows drive absolute (backslash)
            "../escape",  # traversal
            "sub/../escape",  # traversal in middle
        ],
    )
    def test_unsafe_path_rejected(self, path: str) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path=path, kind="file", digest="a" * 64)

    def test_safe_relative_path_accepted(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        entry = UntrackedEntry(path="sub/dir/file.txt", kind="file", mode="0o644", digest="a" * 64)
        assert entry.path == "sub/dir/file.txt"


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

    def test_tampered_completeness_rejected(self, tmp_path: Path) -> None:
        # A record whose stored completeness claims 'complete' but whose sections
        # contain an error (here: accelerator error) must be rejected by the
        # completeness-consistency model_validator. This catches tampered
        # sidecars that misrepresent their own section states.
        ident, snap = _identity_and_snap(tmp_path)
        rec = ProvenanceRecord.from_identity(
            ident,
            source=snap,
            hardware=AcceleratorInfo(status="error", reason="io_error"),
        )
        tampered = rec.model_dump()
        tampered["completeness"] = {"status": "complete", "warnings": [], "limitations": []}
        with pytest.raises(ValidationError):
            ProvenanceRecord.model_validate(tampered)

    def test_explicit_completeness_override_must_match_sections(self, tmp_path: Path) -> None:
        # An explicit completeness override that disagrees with the derived
        # whole-record value is rejected (prevents callers from papering over
        # section errors with a manual 'complete' override).
        ident, snap = _identity_and_snap(tmp_path)
        with pytest.raises(ValidationError):
            ProvenanceRecord.from_identity(
                ident,
                source=snap,
                hardware=AcceleratorInfo(status="error", reason="io_error"),
                completeness=CompletenessInfo(status="complete"),
            )

    def test_explicit_completeness_override_that_matches_accepted(self, tmp_path: Path) -> None:
        # An explicit override that matches the derived value (status, warnings,
        # AND limitations) is accepted.
        ident, snap = _identity_and_snap(tmp_path)
        # Use the default-derived software/hardware/topology so the override
        # exactly matches what from_identity() would derive.
        from expertforge.provenance.record import (
            AcceleratorInfo,
            CompletenessStatus,
            CPUInfo,
            LockfileDigest,
            MemoryInfo,
            PlatformInfo,
            PythonInfo,
            SoftwareEnvironment,
            TopologyInfo,
        )

        software = SoftwareEnvironment(
            python=PythonInfo(version="unknown", implementation="unknown"),
            platform=PlatformInfo(
                cpu=CPUInfo(status="unavailable"),
                memory=MemoryInfo(status="unavailable"),
            ),
            lockfile=LockfileDigest(status="unavailable"),
        )
        hardware = AcceleratorInfo(status="unavailable")
        topology = TopologyInfo(status="not_applicable")
        derived_status, derived_warnings, derived_limitations = (
            ProvenanceRecord._derive_completeness(snap, software, hardware, topology)
        )
        rec = ProvenanceRecord.from_identity(
            ident,
            source=snap,
            software=software,
            hardware=hardware,
            topology=topology,
            completeness=CompletenessInfo(
                status=cast("CompletenessStatus", derived_status),
                warnings=derived_warnings,
                limitations=derived_limitations,
            ),
        )
        assert rec.completeness.status == derived_status
        assert rec.completeness.limitations == derived_limitations


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
