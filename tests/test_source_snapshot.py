"""Tests for the versioned canonical source snapshot (Issue #7 review item 2).

The behavioral-identity digest is the SHA-256 of a **versioned envelope**
(schema, version, tree_digest, evidence), not raw ``sha256(git ls-tree)``.
Dirty evidence includes staged/unstaged/untracked digests, file kinds, symlink
targets (not followed), submodule status, counts, and completeness.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from expertforge.identity.fingerprint import ImmutableInput
from expertforge.provenance.source_snapshot import (
    SOURCE_SNAPSHOT_VERSION,
    DirtySourceError,
    SourceUnavailableError,
    capture_source_snapshot,
    source_snapshot_immutable_input,
)


def _init_repo(repo: Path) -> None:
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
        assert snap.is_clean is True
        assert snap.is_canonical is True
        assert snap.commit_sha
        assert snap.input_digest
        assert len(snap.input_digest) == 64
        int(snap.input_digest, 16)

    def test_clean_snapshot_carries_version(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        snap = capture_source_snapshot(tmp_path)
        assert snap.snapshot_version == SOURCE_SNAPSHOT_VERSION

    def test_clean_snapshot_has_no_evidence(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        snap = capture_source_snapshot(tmp_path)
        assert snap.evidence is None

    def test_clean_envelope_digest_differs_from_raw_tree_digest(self, tmp_path: Path) -> None:
        # The input_digest is the versioned envelope digest, not the raw tree hash.
        _init_repo(tmp_path)
        snap = capture_source_snapshot(tmp_path)
        assert snap.input_digest != snap.tree_digest

    def test_identical_clean_trees_same_input_digest(self, tmp_path: Path) -> None:
        repo_a, repo_b = tmp_path / "a", tmp_path / "b"
        repo_a.mkdir()
        repo_b.mkdir()
        _init_repo(repo_a)
        _init_repo(repo_b)
        a = capture_source_snapshot(repo_a)
        b = capture_source_snapshot(repo_b)
        assert a.input_digest == b.input_digest

    def test_different_content_different_input_digest(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        before = capture_source_snapshot(tmp_path)
        (tmp_path / "a.txt").write_text("goodbye\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "change"], cwd=tmp_path, check=True)
        after = capture_source_snapshot(tmp_path)
        assert before.input_digest != after.input_digest

    def test_captures_branch_and_remote(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        snap = capture_source_snapshot(tmp_path)
        assert snap.branch  # non-empty
        # remote_url may be None (no origin set), which is acceptable.


# --- dirty source ---------------------------------------------------------


class TestDirtySourceSnapshot:
    def test_dirty_rejected_by_default(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
        with pytest.raises(DirtySourceError):
            capture_source_snapshot(tmp_path)

    def test_dirty_allowed_with_flag_marked_non_canonical(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
        snap = capture_source_snapshot(tmp_path, allow_dirty=True)
        assert snap.is_clean is False
        assert snap.is_canonical is False
        assert snap.evidence is not None
        assert snap.evidence.staged_digest
        assert snap.evidence.unstaged_digest

    def test_dirty_does_not_persist_raw_patch(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
        snap = capture_source_snapshot(tmp_path, allow_dirty=True)
        serialized = snap.model_dump_json()
        assert "changed" not in serialized

    def test_dirty_evidence_includes_counts(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
        (tmp_path / "new.txt").write_text("new\n", encoding="utf-8")
        snap = capture_source_snapshot(tmp_path, allow_dirty=True)
        assert snap.evidence is not None
        assert snap.evidence.counts.untracked >= 1
        assert snap.evidence.completeness == "complete"

    def test_dirty_digest_deterministic_for_same_changes(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
        a = capture_source_snapshot(tmp_path, allow_dirty=True)
        b = capture_source_snapshot(tmp_path, allow_dirty=True)
        assert a.input_digest == b.input_digest


# --- symlink no-follow ----------------------------------------------------


class TestSymlinkNoFollow:
    @pytest.mark.skipif(
        os.name == "nt",
        reason="symlink creation requires elevated privileges on Windows",
    )
    def test_untracked_symlink_target_recorded_not_followed(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        # Create a symlink pointing outside the repo.
        target = tmp_path / "external_target.txt"
        target.write_text("secret\n", encoding="utf-8")
        link = tmp_path / "link.txt"
        os.symlink(target, link)  # untracked symlink
        snap = capture_source_snapshot(tmp_path, allow_dirty=True)
        assert snap.evidence is not None
        # Find the symlink in the untracked manifest.
        symlinks = [e for e in snap.evidence.untracked if e.get("kind") == "symlink"]
        assert len(symlinks) == 1
        # The symlink's digest is the target PATH string digest, not the
        # target FILE CONTENT digest. "secret" must not appear.
        serialized = snap.evidence.model_dump_json()
        assert "secret" not in serialized


# --- rename-aware parsing -------------------------------------------------


class TestRenameParsing:
    def test_rename_does_not_crash_parser(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        (tmp_path / "old.txt").write_text("rename me\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "add"], cwd=tmp_path, check=True)
        # Rename via git mv (creates a rename in the index).
        subprocess.run(["git", "mv", "old.txt", "new.txt"], cwd=tmp_path, check=True)
        snap = capture_source_snapshot(tmp_path, allow_dirty=True)
        assert snap.evidence is not None


# --- git unavailable ------------------------------------------------------


class TestSourceDegradesWhenGitUnavailable:
    def test_not_a_repo_degrades_explicitly(self, tmp_path: Path) -> None:
        with pytest.raises(SourceUnavailableError):
            capture_source_snapshot(tmp_path)


# --- immutable input adapter ----------------------------------------------


class TestSourceImmutableInput:
    def test_returns_real_immutable_input(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        snap = capture_source_snapshot(tmp_path)
        ii = source_snapshot_immutable_input(snap)
        assert isinstance(ii, ImmutableInput)
        assert ii.name == "source.snapshot"
        assert ii.algorithm == "sha256"
        assert ii.digest == snap.input_digest


# --- immutability ---------------------------------------------------------


class TestSourceSnapshotImmutability:
    def test_snapshot_is_frozen(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        snap = capture_source_snapshot(tmp_path)
        with pytest.raises(ValidationError):
            snap.commit_sha = "other"  # type: ignore[misc]
