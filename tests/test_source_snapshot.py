"""Tests for the source snapshot (Issue #7 decision: source identity).

The source-code *content* snapshot is behavioral identity (an #6
``ImmutableInput``); the rest of provenance is not. For a clean repo the
snapshot digest is derived from deterministic tree material
(``git ls-tree -r -z --full-tree HEAD``), not branch names, paths, timestamps,
or commit history alone.

Dirty execution:
- rejected by default;
- permitted only with explicit ``allow_dirty=True``;
- always marked non-canonical;
- uses a deterministic hash of tracked changes + untracked files;
- does not persist a raw patch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from expertforge.provenance.source_snapshot import (
    SOURCE_SNAPSHOT_VERSION,
    DirtySourceError,
    SourceSnapshot,
    SourceUnavailableError,
    capture_source_snapshot,
)


def _init_repo(repo: Path) -> None:
    """Initialize a throwaway git repo with one committed file."""
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)
    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


# --- clean source ---------------------------------------------------------


class TestCleanSourceSnapshot:
    def test_clean_snapshot_is_canonical(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        snap = capture_source_snapshot(tmp_path)
        assert isinstance(snap, SourceSnapshot)
        assert snap.is_clean is True
        assert snap.is_canonical is True
        assert snap.commit_sha
        assert snap.content_digest  # 64 hex
        assert len(snap.content_digest) == 64
        int(snap.content_digest, 16)

    def test_clean_snapshot_carries_version(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        snap = capture_source_snapshot(tmp_path)
        assert snap.snapshot_version == SOURCE_SNAPSHOT_VERSION

    def test_identical_clean_trees_same_digest(self, tmp_path: Path) -> None:
        # Two repos with identical committed content -> same content digest,
        # independent of commit SHA / branch / paths.
        repo_a = tmp_path / "a"
        repo_b = tmp_path / "b"
        repo_a.mkdir()
        repo_b.mkdir()
        _init_repo(repo_a)
        _init_repo(repo_b)
        a = capture_source_snapshot(repo_a)
        b = capture_source_snapshot(repo_b)
        assert a.content_digest == b.content_digest

    def test_different_content_different_digest(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        before = capture_source_snapshot(tmp_path)
        # Change committed content (not dirty — commit it).
        (tmp_path / "a.txt").write_text("goodbye\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "change"], cwd=tmp_path, check=True)
        after = capture_source_snapshot(tmp_path)
        assert before.content_digest != after.content_digest


# --- dirty source ---------------------------------------------------------


class TestDirtySourceSnapshot:
    def test_dirty_rejected_by_default(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")  # dirty
        with pytest.raises(DirtySourceError):
            capture_source_snapshot(tmp_path)

    def test_dirty_allowed_with_flag_marked_non_canonical(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
        snap = capture_source_snapshot(tmp_path, allow_dirty=True)
        assert snap.is_clean is False
        assert snap.is_canonical is False
        # Dirty hash evidence present.
        assert snap.dirty_digest is not None
        assert len(snap.dirty_digest) == 64

    def test_dirty_does_not_persist_raw_patch(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
        snap = capture_source_snapshot(tmp_path, allow_dirty=True)
        # No patch content is recorded; only digests.
        assert not hasattr(snap, "patch") or snap.patch is None
        # The serialized form must not contain the dirty file contents.
        serialized = snap.model_dump_json()
        assert "changed" not in serialized

    def test_dirty_digest_deterministic_for_same_changes(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
        a = capture_source_snapshot(tmp_path, allow_dirty=True)
        b = capture_source_snapshot(tmp_path, allow_dirty=True)
        assert a.dirty_digest == b.dirty_digest


# --- git unavailable / not a repo ----------------------------------------


class TestSourceDegradesWhenGitUnavailable:
    def test_not_a_repo_degrades_explicitly(self, tmp_path: Path) -> None:
        # tmp_path is not a git repo.
        with pytest.raises(SourceUnavailableError):
            capture_source_snapshot(tmp_path)


# --- immutability ---------------------------------------------------------


class TestSourceSnapshotImmutability:
    def test_snapshot_is_frozen(self, tmp_path: Path) -> None:
        from pydantic import ValidationError

        _init_repo(tmp_path)
        snap = capture_source_snapshot(tmp_path)
        with pytest.raises(ValidationError):
            snap.commit_sha = "other"  # type: ignore[misc]
