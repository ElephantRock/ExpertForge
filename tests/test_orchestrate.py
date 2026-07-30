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
from expertforge.provenance.record import ProvenanceRecord

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

    def test_prepare_run_propagates_topology_warnings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # prepare_run() uses capture_hardware() (not capture_accelerator() +
        # capture_topology() separately) so topology_warnings are NOT lost
        # between the two calls. An unparseable RANK env value must surface in
        # the provenance record's completeness warnings.
        from expertforge.config.resolve import resolve_config

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        monkeypatch.setenv("RANK", "not-an-int")
        monkeypatch.setenv("WORLD_SIZE", "2")
        _, provenance, _ = prepare_run(
            artifact_root=tmp_path / "runs",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            clock=lambda: _FIXED,
            entropy=lambda n: bytes(n),
        )
        # The topology warning is stored durably on TopologyInfo AND surfaces
        # in the whole-record completeness warnings.
        assert "invalid_topology_env_value:RANK" in provenance.topology.topology_warnings
        assert "invalid_topology_env_value:RANK" in provenance.completeness.warnings

    def test_prepare_run_preserves_topology_warnings_when_topology_supplied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Review item 4: when topology is EXPLICITLY supplied, prepare_run() must
        # still call capture_hardware() and copy the aggregate's
        # topology_warnings onto the supplied TopologyInfo — they must not
        # disappear just because an override was provided.
        from expertforge.config.resolve import resolve_config
        from expertforge.provenance.record import TopologyInfo

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        monkeypatch.setenv("RANK", "not-an-int")
        monkeypatch.setenv("WORLD_SIZE", "2")
        explicit_topology = TopologyInfo(status="available", rank=0, world_size=2)
        _, provenance, _ = prepare_run(
            artifact_root=tmp_path / "runs",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            topology=explicit_topology,
            clock=lambda: _FIXED,
            entropy=lambda n: bytes(n),
        )
        # The explicit override's fields win, but the captured topology_warning
        # is preserved (not dropped).
        assert provenance.topology.rank == 0
        assert provenance.topology.world_size == 2
        assert "invalid_topology_env_value:RANK" in provenance.topology.topology_warnings
        assert "invalid_topology_env_value:RANK" in provenance.completeness.warnings

    def test_prepare_run_preserves_topology_warnings_when_hardware_supplied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Review item 4: when hardware is EXPLICITLY supplied (but topology is
        # not), prepare_run() must still call capture_hardware() and the
        # aggregate's topology_warnings must reach the derived TopologyInfo.
        from expertforge.config.resolve import resolve_config
        from expertforge.provenance.record import AcceleratorInfo

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        monkeypatch.setenv("RANK", "not-an-int")
        monkeypatch.setenv("WORLD_SIZE", "2")
        _, provenance, _ = prepare_run(
            artifact_root=tmp_path / "runs",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            hardware=AcceleratorInfo(status="unavailable"),
            clock=lambda: _FIXED,
            entropy=lambda n: bytes(n),
        )
        assert "invalid_topology_env_value:RANK" in provenance.topology.topology_warnings
        assert "invalid_topology_env_value:RANK" in provenance.completeness.warnings


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


# --- sanitized remote round-trip through sidecar (review item 1) -------------


class TestSanitizedRemoteSidecarRoundTrip:
    """Review item 1: a sanitized remote URL plus its retained warning codes
    MUST survive a full ``capture_source_snapshot → prepare_run → sidecar
    write → parse_provenance_sidecar`` round-trip. The previous design re-derived
    required warnings from the already-sanitized URL on load, which rejected
    the retained warnings as fabricated. The capture path now sanitizes the
    raw URL and supplies BOTH the sanitized value and the warning codes to the
    constructor; the model no longer correlates the two."""

    @staticmethod
    def _set_origin(repo: Path, url: str) -> None:
        # Set (or replace) the origin remote URL on a repo.
        subprocess.run(["git", "remote", "remove", "origin"], cwd=repo, check=False)
        subprocess.run(["git", "remote", "add", "origin", url], cwd=repo, check=True)

    def _round_trip(
        self, tmp_path: Path, remote_url: str
    ) -> tuple[ProvenanceRecord, ProvenanceRecord]:
        from expertforge.config.resolve import resolve_config
        from expertforge.provenance.sidecar import parse_provenance_sidecar

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        if remote_url:  # empty string → leave no origin configured
            self._set_origin(repo, remote_url)
        _, provenance, path = prepare_run(
            artifact_root=tmp_path / "runs",
            config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
            repo=repo,
            clock=lambda: _FIXED,
            entropy=lambda n: bytes(n),
        )
        loaded = parse_provenance_sidecar(path)
        return provenance, loaded

    def test_credential_https_round_trips(self, tmp_path: Path) -> None:
        provenance, loaded = self._round_trip(
            tmp_path, "https://user:token@github.com/org/repo.git"
        )
        # The raw credentials are NEVER stored.
        assert "token" not in provenance.source.model_dump_json()
        assert provenance.source.remote_url == "https://github.com/org/repo.git"
        codes = {w.code for w in provenance.source.remote_warnings}
        assert codes == {"remote_credentials_removed"}
        # The loaded record matches: the retained warning survives the round-trip.
        assert loaded == provenance
        assert {w.code for w in loaded.source.remote_warnings} == {"remote_credentials_removed"}
        assert loaded.source.remote_url == "https://github.com/org/repo.git"

    def test_git_at_host_https_round_trips(self, tmp_path: Path) -> None:
        # Regression for the credential-classifier bug:
        # ``https://git@host/repo.git`` has scheme + userinfo, so its ``@`` IS
        # credential-bearing and is flagged + sanitized.
        provenance, loaded = self._round_trip(tmp_path, "https://git@host/repo.git")
        assert provenance.source.remote_url == "https://host/repo.git"
        codes = {w.code for w in provenance.source.remote_warnings}
        assert codes == {"remote_credentials_removed"}
        assert loaded == provenance
        assert loaded.source.remote_url == "https://host/repo.git"

    def test_query_round_trips(self, tmp_path: Path) -> None:
        provenance, loaded = self._round_trip(
            tmp_path, "https://github.com/org/repo.git?signed=xyz"
        )
        assert provenance.source.remote_url == "https://github.com/org/repo.git"
        codes = {w.code for w in provenance.source.remote_warnings}
        assert codes == {"remote_query_fragment_removed"}
        assert loaded == provenance

    def test_fragment_round_trips(self, tmp_path: Path) -> None:
        provenance, loaded = self._round_trip(tmp_path, "https://github.com/org/repo.git#frag")
        assert provenance.source.remote_url == "https://github.com/org/repo.git"
        codes = {w.code for w in provenance.source.remote_warnings}
        assert codes == {"remote_query_fragment_removed"}
        assert loaded == provenance

    def test_local_file_round_trips(self, tmp_path: Path) -> None:
        provenance, loaded = self._round_trip(tmp_path, "/home/secret/repos/ExpertForge")
        # A local-path URL sanitizes to None; the warning is retained.
        assert provenance.source.remote_url is None
        codes = {w.code for w in provenance.source.remote_warnings}
        assert codes == {"remote_local_or_unsupported_removed"}
        assert loaded == provenance
        assert loaded.source.remote_url is None

    def test_no_origin_round_trips(self, tmp_path: Path) -> None:
        provenance, loaded = self._round_trip(tmp_path, "")  # empty → no origin set
        assert provenance.source.remote_url is None
        assert provenance.source.remote_warnings == ()
        assert loaded == provenance

    def test_scp_style_round_trips(self, tmp_path: Path) -> None:
        # SCP-style ``git@host:path`` has no scheme; the ``git`` is a protocol
        # user, not a credential. It is preserved verbatim with NO warning.
        provenance, loaded = self._round_trip(
            tmp_path, "git@github.com:ElephantRock/ExpertForge.git"
        )
        assert provenance.source.remote_url == "git@github.com:ElephantRock/ExpertForge.git"
        assert provenance.source.remote_warnings == ()
        assert loaded == provenance
