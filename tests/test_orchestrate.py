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
        assert provenance.source.summary.input_digest == snap.input_digest
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
        assert provenance.source.summary.is_clean is False
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
