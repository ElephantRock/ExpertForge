"""Portable CPU-only integration tests for the Issue #10 artifact contract
(design comment 5138481783 ratified by amendment 5138634449 §K/L).

Covers the required portable integration matrix:

1. full publish → locate → verify → list → inspect lifecycle;
2. crash recovery: bundle rename succeeded before registry append (orphan) and
   crash during write (no partial canonical dir, no registry record);
3. telemetry stream registration through the public Issue #9 loader (complete
   accepted; incomplete/corrupt rejected);
4. orphan detection and reconciliation;
5. target-exists idempotency vs. immutable-metadata conflict;
6. source-path mutation after register_existing not changing canonical bytes;
7. credential-bearing external locations rejected;
8. git check-ignore evidence for generated run/artifact paths.

Markers: permanent #13 CPU-integration tier; CPU-only; no network; no
credentials; no GPU; no ExpertOS dependency.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests._artifact_fixtures import make_store

from expertforge.artifacts.models import (
    ArtifactConflictError,
    ArtifactRecord,
    ParentReference,
)
from expertforge.artifacts.store import ArtifactStore

pytestmark = pytest.mark.integration

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _init_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)
    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


def _prepare_run(tmp_path: Path) -> tuple[Path, Path]:
    """Initialize a git repo and return (repo, artifact_root) for prepare_run."""
    from expertforge.config.resolve import resolve_config
    from expertforge.provenance.orchestrate import prepare_run

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    root = tmp_path / "runs"
    prepare_run(
        artifact_root=root,
        config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
        repo=repo,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        entropy=lambda n: bytes(n),
    )
    return repo, root


# ---------------------------------------------------------------------------
# 1. Full publish → locate → verify → list → inspect lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_publish_locate_verify_list_inspect(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        rec = store.publish(
            b'{"answer":42}',
            category="report",
            format="json",
            format_version=1,
            producing_component="eval",
        )
        # locate
        content_path = store.locate(rec.artifact_id)
        assert content_path is not None
        assert content_path.read_bytes() == b'{"answer":42}'
        # verify (typed result)
        result = store.verify(rec.artifact_id)
        assert result.status is True
        assert result.diagnostic_code == "verified"
        assert result.observed_digest == "sha256:" + hashlib.sha256(b'{"answer":42}').hexdigest()
        # list
        listed = store.list_artifacts()
        assert len(listed) == 1
        # inspect returns the registry record
        inspected = store.inspect(rec.artifact_id)
        assert isinstance(inspected, ArtifactRecord)
        assert inspected.artifact_id == rec.artifact_id
        assert inspected.byte_size == len(b'{"answer":42}')


# ---------------------------------------------------------------------------
# 2. Crash recovery
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    def test_orphan_bundle_after_crash_reconciled(self, tmp_path: Path) -> None:
        """A crash after bundle rename but before registry append leaves a valid
        unindexed bundle. Reconciliation appends its entry after strict
        validation (amendment C/K)."""
        store = make_store(tmp_path)
        # Publish normally, then wipe the registry to simulate the crash window.
        rec = store.publish(
            b"payload-bytes",
            category="report",
            format="text",
            format_version=1,
            producing_component="training",
        )
        store.registry_path.unlink()
        # The bundle exists but is unindexed.
        bundle = store.attempt_dir / "artifacts" / "report" / rec.artifact_id
        assert bundle.exists()
        assert store.list_artifacts() == []
        # Reconciliation finds and indexes the orphan.
        recon = store.reconcile_orphans()
        assert rec.artifact_id in recon.orphan_artifact_ids
        assert rec.artifact_id in recon.appended_artifact_ids
        assert len(store.list_artifacts()) == 1
        assert store.verify(rec.artifact_id).status is True

    def test_crash_during_write_leaves_no_canonical_dir(self, tmp_path: Path) -> None:
        """A crash during write must not leave a partial canonical directory or
        a registry record."""
        store = make_store(tmp_path)
        # Simulate a crashed temp bundle by creating one directly (no rename,
        # no registry record).
        cdir = store.attempt_dir / "artifacts" / "report"
        cdir.mkdir(parents=True)
        crashed_tmp = cdir / ".tmp-bundle-crashed"
        crashed_tmp.mkdir()
        (crashed_tmp / "content").write_bytes(b"partial")
        # No canonical bundles, no registry entries.
        assert store.list_artifacts() == []
        # reconcile_orphans must skip temp bundles and report zero appends.
        recon = store.reconcile_orphans()
        assert recon.appended_artifact_ids == []

    def test_corrupt_orphan_skipped(self, tmp_path: Path) -> None:
        """An orphan whose content digest does not match its metadata is skipped,
        not indexed (amendment K: strict validation)."""
        store = make_store(tmp_path)
        # Publish, then tamper the content and wipe the registry.
        rec = store.publish(
            b"good",
            category="report",
            format="text",
            format_version=1,
            producing_component="training",
        )
        (store.attempt_dir / "artifacts" / "report" / rec.artifact_id / "content").write_bytes(
            b"TAMPERED"
        )
        store.registry_path.unlink()
        recon = store.reconcile_orphans()
        assert rec.artifact_id in recon.orphan_artifact_ids
        assert rec.artifact_id not in recon.appended_artifact_ids
        assert any(aid == rec.artifact_id for aid, _ in recon.skipped)


# ---------------------------------------------------------------------------
# 3. Telemetry stream registration through the public loader
# ---------------------------------------------------------------------------


class TestTelemetryRegistration:
    def _write_stream(self, store: ArtifactStore) -> Path:
        from expertforge.telemetry.models import (
            MetricObservation,
            ProcessContext,
            ProgressPosition,
        )
        from expertforge.telemetry.writer import TelemetryWriter

        ctx = ProcessContext(rank=0, world_size=1, local_rank=0)

        class _Clk:
            def __init__(self) -> None:
                self._mono = 0

            def wall(self) -> datetime:
                return datetime(2026, 1, 1, tzinfo=UTC)

            def mono(self) -> int:
                v = self._mono
                self._mono += 1_000_000
                return v

        clk = _Clk()
        w = TelemetryWriter(
            artifact_root=store.artifact_root,
            identity=store.identity,
            process_context=ctx,
            console_stream=None,
            console_enabled=False,
            wall_clock=clk.wall,
            monotonic_clock=clk.mono,
        )
        w.emit_event(component="training", severity="INFO", event_name="training.update")
        w.emit_metric(
            component="training",
            progress=ProgressPosition(step=1, update=1, processed_tokens=128),
            observations=[
                MetricObservation(
                    namespace="training",
                    name="loss",
                    unit="dimensionless",
                    aggregation="gauge",
                    window="point",
                    value_status="finite",
                    value=0.5,
                )
            ],
        )
        w.close(outcome="normal")
        return w.path

    def test_complete_telemetry_registered_as_artifact(self, tmp_path: Path) -> None:
        from expertforge.telemetry.models import ProcessContext

        store = make_store(tmp_path)
        stream_path = self._write_stream(store)
        ctx = ProcessContext(rank=0, world_size=1, local_rank=0)
        # Register the verified stream through the enforced telemetry path: the
        # public loader verifies completeness, then the bytes are copied into a
        # canonical telemetry bundle (item #9).
        rec = store.register_telemetry(
            stream_path,
            process_context=ctx,
            producing_component="logging",
        )
        assert rec.category == "telemetry"
        assert rec.format == "jsonl"
        content = store.locate(rec.artifact_id)
        assert content is not None
        assert content.read_bytes() == stream_path.read_bytes()

    def test_incomplete_telemetry_rejected_for_canonical_registration(self, tmp_path: Path) -> None:
        from expertforge.artifacts.store import ArtifactStoreError
        from expertforge.telemetry.models import ProcessContext

        store = make_store(tmp_path)
        stream_path = self._write_stream(store)
        ctx = ProcessContext(rank=0, world_size=1, local_rank=0)
        # Truncate the stream to simulate an incomplete stream.
        raw = stream_path.read_bytes()
        stream_path.write_bytes(raw[:-5])
        # register_telemetry enforces canonical verification end-to-end: an
        # incomplete stream is rejected with a typed store error (item #9).
        with pytest.raises(ArtifactStoreError):
            store.register_telemetry(
                stream_path,
                process_context=ctx,
                producing_component="logging",
            )


# ---------------------------------------------------------------------------
# 4. Orphan detection and reconciliation (cross-cutting, see TestCrashRecovery)
# 5. Target-exists idempotency vs. immutable-metadata conflict
# ---------------------------------------------------------------------------


class TestIdempotencyVsConflict:
    def test_idempotent_republish_returns_existing(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        r1 = store.publish(
            b"same",
            category="report",
            format="text",
            format_version=1,
            producing_component="training",
        )
        r2 = store.publish(
            b"same",
            category="report",
            format="text",
            format_version=1,
            producing_component="training",
        )
        assert r1.artifact_id == r2.artifact_id
        # Only one registry entry.
        assert len(store.list_artifacts()) == 1

    def test_immutable_metadata_conflict(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        # Different producing_component -> different artifact_id, so no conflict
        # at the same path. Conflict arises only when the same artifact_id
        # appears with mismatched immutable metadata. We construct that by
        # tampering the bundle's artifact.json after publication.
        rec = store.publish(
            b"payload",
            category="report",
            format="text",
            format_version=1,
            producing_component="training",
        )
        meta_path = store.attempt_dir / "artifacts" / "report" / rec.artifact_id / "artifact.json"
        meta = json.loads(meta_path.read_bytes())
        meta["producing_component"] = "tampered"
        meta_path.write_text(json.dumps(meta))
        # Re-publishing identical inputs now sees mismatched immutable metadata.
        with pytest.raises(ArtifactConflictError):
            store.publish(
                b"payload",
                category="report",
                format="text",
                format_version=1,
                producing_component="training",
            )


# ---------------------------------------------------------------------------
# 6. Source-path mutation independence
# ---------------------------------------------------------------------------


class TestSourceMutationIndependence:
    def test_source_mutation_does_not_change_canonical(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        src = tmp_path / "src.bin"
        src.write_bytes(b"v1")
        rec = store.register_existing(
            src,
            category="checkpoint",
            format="binary",
            format_version=1,
            producing_component="ckpt",
        )
        canonical = store.locate(rec.artifact_id)
        assert canonical is not None
        # Mutate and re-save the source.
        src.write_bytes(b"v2-completely-different")
        # Canonical content is unchanged.
        assert canonical.read_bytes() == b"v1"
        assert store.verify(rec.artifact_id).status is True


# ---------------------------------------------------------------------------
# 7. Credential-bearing external locations rejected
# ---------------------------------------------------------------------------


class TestExternalCredentialRejection:
    def test_userinfo_rejected(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(ArtifactConflictError):
            store.register_external(
                category="report",
                format="json",
                format_version=1,
                producing_component="eval",
                location_type="uri",
                location="https://alice:secret@example.com/r.json",
                expected_digest="sha256:" + "a" * 64,
                expected_byte_size=10,
            )

    def test_token_query_rejected(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(ArtifactConflictError):
            store.register_external(
                category="report",
                format="json",
                format_version=1,
                producing_component="eval",
                location_type="uri",
                location="https://example.com/r.json?token=abc",
                expected_digest="sha256:" + "a" * 64,
                expected_byte_size=10,
            )


# ---------------------------------------------------------------------------
# 8. git check-ignore evidence for generated run/artifact paths
# ---------------------------------------------------------------------------


class TestGitIgnore:
    def test_generated_run_paths_are_gitignored(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        # Copy the project .gitignore so check-ignore reflects policy.
        project_gitignore = Path(__file__).resolve().parents[2] / ".gitignore"
        (repo / ".gitignore").write_text(project_gitignore.read_text(), encoding="utf-8")
        store = make_store(tmp_path)
        rec = store.publish(
            b"ignored-content",
            category="report",
            format="text",
            format_version=1,
            producing_component="training",
        )
        bundle = store.attempt_dir / "artifacts" / "report" / rec.artifact_id
        rel = bundle.relative_to(tmp_path).as_posix()
        result = subprocess.run(
            ["git", "check-ignore", rel],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        # check-ignore prints the path and exits 0 when ignored.
        assert result.returncode == 0, (
            f"generated artifact path {rel} is NOT gitignored: "
            f"rc={result.returncode} out={result.stdout!r} err={result.stderr!r}"
        )

    def test_registry_path_is_gitignored(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        project_gitignore = Path(__file__).resolve().parents[2] / ".gitignore"
        (repo / ".gitignore").write_text(project_gitignore.read_text(), encoding="utf-8")
        store = make_store(tmp_path)
        store.publish(
            b"x",
            category="report",
            format="text",
            format_version=1,
            producing_component="training",
        )
        rel = store.registry_path.relative_to(tmp_path).as_posix()
        result = subprocess.run(
            ["git", "check-ignore", rel],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"registry path {rel} is NOT gitignored: {result.stderr!r}"


# ---------------------------------------------------------------------------
# Cross-attempt parent reference end-to-end
# ---------------------------------------------------------------------------


class TestCrossAttemptParent:
    def test_checkpoint_with_cross_attempt_parent(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        parent = ParentReference(
            run_id="run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb",
            attempt_id="attempt-20260101t000000z-dddddddddddddddddddd",
            artifact_id="artifact-v1-sha256-" + "e" * 64,
        )
        rec = store.publish(
            b"checkpoint-bytes",
            category="checkpoint",
            format="binary",
            format_version=1,
            producing_component="ckpt",
            parent=parent,
        )
        assert rec.parent == parent
        # Persisted in artifact.json and registry.
        meta = json.loads(
            (
                store.attempt_dir / "artifacts" / "checkpoint" / rec.artifact_id / "artifact.json"
            ).read_bytes()
        )
        assert meta["parent"]["run_id"] == parent.run_id
        inspected = store.inspect(rec.artifact_id)
        assert isinstance(inspected, ArtifactRecord)
        assert inspected.parent == parent


# ---------------------------------------------------------------------------
# Deterministic checkpoint-archive boundary as ordinary files (amendment I/L)
# ---------------------------------------------------------------------------


class TestCheckpointArchiveBoundary:
    def test_tar_bytes_published_as_checkpoint(self, tmp_path: Path) -> None:
        # Issue #11 produces tar archives; Issue #10 hashes and publishes the
        # bytes without understanding training-framework objects. Here we
        # supply ordinary bytes under format=tar.
        store = make_store(tmp_path)
        archive = tmp_path / "ckpt.tar"
        archive.write_bytes(b"FAKE-TAR-CONTENT")
        rec = store.register_existing(
            archive,
            category="checkpoint",
            format="tar",
            format_version=1,
            producing_component="ckpt",
        )
        assert rec.format == "tar"
        assert store.verify(rec.artifact_id).status is True
