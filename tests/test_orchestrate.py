"""Tests for the source→identity→provenance orchestration API (Issue #7 review item 1).

One API performs the full normative runtime order:
  capture source snapshot
  → create ImmutableInput("source.snapshot")
  → emit identity containing that input
  → capture typed provenance against the identity
  → write run-provenance.json before training
  → return

The emitted specification fingerprint must contain exactly one source.snapshot
input with the captured digest. The provenance record copies that tuple from
the identity. Missing/duplicate/changed/independently-supplied source inputs
are rejected.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from expertforge.provenance.orchestrate import (
    ProvenanceOrchestrationError,
    prepare_run,
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


class TestPrepareRun:
    def test_prepare_run_emits_identity_with_source_snapshot(self, tmp_path: Path) -> None:
        from expertforge.config.resolve import resolve_config

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        identity, provenance, path = prepare_run(
            artifact_root=tmp_path / "runs",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            clock=lambda: _FIXED,
            entropy=lambda n: bytes(n),
        )
        # Identity fingerprint contains exactly one source.snapshot input.
        inputs = identity.specification_fingerprint.immutable_inputs
        source_inputs = [ii for ii in inputs if ii.name == "source.snapshot"]
        assert len(source_inputs) == 1
        # Provenance record copies that tuple from the identity.
        assert provenance.immutable_inputs == inputs
        # Sidecar written before return.
        assert path.exists()
        assert path.name == "run-provenance.json"

    def test_provenance_source_state_matches_snapshot(self, tmp_path: Path) -> None:
        from expertforge.config.resolve import resolve_config
        from expertforge.provenance.source_snapshot import capture_source_snapshot

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        identity, provenance, _ = prepare_run(
            artifact_root=tmp_path / "runs",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            clock=lambda: _FIXED,
            entropy=lambda n: bytes(n),
        )
        snap = capture_source_snapshot(repo)
        assert provenance.source is not None
        assert provenance.source.input_digest == snap.input_digest
        source_input = next(
            ii
            for ii in identity.specification_fingerprint.immutable_inputs
            if ii.name == "source.snapshot"
        )
        assert source_input.digest == snap.input_digest

    def test_prepare_run_rejects_dirty_without_allow(self, tmp_path: Path) -> None:
        from expertforge.config.resolve import resolve_config

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "a.txt").write_text("dirty\n", encoding="utf-8")
        with pytest.raises(ProvenanceOrchestrationError):
            prepare_run(
                artifact_root=tmp_path / "runs",
                config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
                repo=repo,
                clock=lambda: _FIXED,
                entropy=lambda n: bytes(n),
            )

    def test_prepare_run_allows_dirty_with_flag(self, tmp_path: Path) -> None:
        from expertforge.config.resolve import resolve_config

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "a.txt").write_text("dirty\n", encoding="utf-8")
        identity, provenance, path = prepare_run(
            artifact_root=tmp_path / "runs",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            allow_dirty=True,
            clock=lambda: _FIXED,
            entropy=lambda n: bytes(n),
        )
        assert provenance.source is not None
        assert provenance.source.is_clean is False
        assert path.exists()

    def test_two_preparations_same_source_same_fingerprint(self, tmp_path: Path) -> None:
        from expertforge.config.resolve import resolve_config

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        id1, _, _ = prepare_run(
            artifact_root=tmp_path / "runs1",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x01" * n,
        )
        id2, _, _ = prepare_run(
            artifact_root=tmp_path / "runs2",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x02" * n,
        )
        assert id1.specification_fingerprint.digest_str == id2.specification_fingerprint.digest_str

    def test_changed_source_changes_fingerprint(self, tmp_path: Path) -> None:
        from expertforge.config.resolve import resolve_config

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        id1, _, _ = prepare_run(
            artifact_root=tmp_path / "runs1",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            clock=lambda: _FIXED,
            entropy=lambda n: bytes(n),
        )
        (repo / "a.txt").write_text("changed\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "change"], cwd=repo, check=True)
        id2, _, _ = prepare_run(
            artifact_root=tmp_path / "runs2",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            clock=lambda: _FIXED,
            entropy=lambda n: bytes(n),
        )
        assert id1.specification_fingerprint.digest_str != id2.specification_fingerprint.digest_str

    def test_prepare_run_derives_completeness_from_all_sections(self, tmp_path: Path) -> None:
        # prepare_run() must NOT supply a source-only completeness override;
        # it lets from_identity() derive whole-record completeness. A record
        # with a clean source but an accelerator error must be 'partial' with
        # the accelerator warning — a source-only override would have missed it.
        from expertforge.config.resolve import resolve_config
        from expertforge.provenance.record import AcceleratorInfo

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        _, provenance, _ = prepare_run(
            artifact_root=tmp_path / "runs",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            hardware=AcceleratorInfo(status="error", reason="io_error"),
            clock=lambda: _FIXED,
            entropy=lambda n: bytes(n),
        )
        # The accelerator error propagates to whole-record completeness. A
        # source-only override would have reported 'complete' here; the
        # whole-record derivation surfaces the error.
        assert "accelerator_error" in provenance.completeness.warnings
        assert provenance.completeness.status == "error"


# --- RESUME / FORK / LEGACY via prepare_run --------------------------------


class TestPrepareRunAllocationModes:
    """End-to-end regression: prepare_run() supports RESUME/FORK/LEGACY."""

    def test_resume_via_prepare_run_retains_run_id(self, tmp_path: Path) -> None:
        from expertforge.config.resolve import resolve_config
        from expertforge.identity.emit import AllocationMode
        from expertforge.identity.lineage import ResumeLineage

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        # First independent preparation.
        first, first_prov, _ = prepare_run(
            artifact_root=tmp_path / "runs1",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x01" * n,
        )
        # Resume: same source → same spec fingerprint, retained run ID.
        lin = ResumeLineage(
            parent_run_id=first.run_id,
            parent_attempt_id=first.attempt_id,
            parent_checkpoint_id="ckpt-001",
        )
        resumed, resumed_prov, resumed_path = prepare_run(
            artifact_root=tmp_path / "runs2",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            mode=AllocationMode.RESUME,
            lineage=lin,
            parent_specification_fingerprint=first.fingerprint_digest_str(),
            retained_run_id=first.run_id,
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x02" * n,
        )
        assert resumed.run_id == first.run_id
        assert resumed.attempt_id != first.attempt_id
        # Same source → same spec fingerprint.
        assert resumed.fingerprint_digest_str() == first.fingerprint_digest_str()
        assert resumed.lineage is not None
        assert resumed.lineage.parent_run_id == first.run_id
        assert resumed.lineage.parent_checkpoint_id == "ckpt-001"
        # Same source → same provenance source binding.
        assert resumed_prov.source.input_digest == first_prov.source.input_digest
        assert resumed_path.exists()

    def test_fork_via_prepare_run_new_run_with_parent_lineage(self, tmp_path: Path) -> None:
        from expertforge.config.resolve import resolve_config
        from expertforge.identity.emit import AllocationMode
        from expertforge.identity.lineage import ResumeLineage

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        first, first_prov, _ = prepare_run(
            artifact_root=tmp_path / "runs1",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x01" * n,
        )
        # FORK: change the source materially, allocate a new run with lineage.
        (repo / "a.txt").write_text("forked\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "fork"], cwd=repo, check=True)
        lin = ResumeLineage(
            parent_run_id=first.run_id,
            parent_attempt_id=first.attempt_id,
            parent_checkpoint_id="ckpt-001",
        )
        forked, forked_prov, forked_path = prepare_run(
            artifact_root=tmp_path / "runs2",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            mode=AllocationMode.FORK,
            lineage=lin,
            parent_specification_fingerprint=first.fingerprint_digest_str(),
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x02" * n,
        )
        # New run ID, materially different spec fingerprint.
        assert forked.run_id != first.run_id
        assert forked.fingerprint_digest_str() != first.fingerprint_digest_str()
        assert forked.lineage is not None
        assert forked.lineage.parent_run_id == first.run_id
        assert forked_prov.source.input_digest != first_prov.source.input_digest
        assert forked_path.exists()

    def test_legacy_via_prepare_run_allows_missing_parent_attempt(self, tmp_path: Path) -> None:
        from expertforge.config.resolve import resolve_config
        from expertforge.identity.emit import AllocationMode
        from expertforge.identity.lineage import ResumeLineage

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        first, first_prov, _ = prepare_run(
            artifact_root=tmp_path / "runs1",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x01" * n,
        )
        # LEGACY: parent_attempt_id may be None for imported/legacy provenance.
        lin = ResumeLineage(
            parent_run_id=first.run_id,
            parent_attempt_id=None,  # legacy: parent attempt not reconstructable
            parent_checkpoint_id="imported-ckpt",
        )
        legacy, legacy_prov, legacy_path = prepare_run(
            artifact_root=tmp_path / "runs2",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            mode=AllocationMode.LEGACY,
            lineage=lin,
            parent_specification_fingerprint=first.fingerprint_digest_str(),
            retained_run_id=first.run_id,
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x02" * n,
        )
        assert legacy.run_id == first.run_id
        assert legacy.attempt_id != first.attempt_id
        assert legacy.fingerprint_digest_str() == first.fingerprint_digest_str()
        assert legacy.lineage is not None
        assert legacy.lineage.parent_attempt_id is None
        assert legacy_prov.source.input_digest == first_prov.source.input_digest
        assert legacy_path.exists()
