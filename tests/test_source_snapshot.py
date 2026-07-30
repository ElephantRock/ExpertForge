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
from typing import Any, cast

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
        symlinks = [e for e in snap.evidence.untracked if e.kind == "symlink"]
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


# --- dirty submodule detection via gitlink intersection (review item 1) ----


def _add_submodule(parent: Path, sub_url: str, sub_path: str) -> None:
    """Add a submodule to ``parent`` using the file:// protocol allowlist."""
    subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "submodule", "add", "-q", sub_url, sub_path],
        cwd=parent,
        check=True,
    )


class TestDirtySubmoduleGitlinkDetection:
    """Real dirty submodules are detected by intersecting tracked gitlink paths
    (mode 160000 from ``git ls-files --stage``) with non-clean porcelain
    entries. Git porcelain does NOT add trailing slashes for submodules."""

    def test_dirty_submodule_content_marks_evidence_partial(self, tmp_path: Path) -> None:
        # Build a real parent repo with an initialized submodule.
        sub = tmp_path / "sub"
        sub.mkdir()
        _init_repo(sub)
        parent = tmp_path / "parent"
        parent.mkdir()
        _init_repo(parent)
        _add_submodule(parent, str(sub.resolve()).replace("\\", "/"), "vendor/sub")
        subprocess.run(["git", "commit", "-q", "-m", "add submodule"], cwd=parent, check=True)
        # Now dirty the submodule's content. This produces a non-clean porcelain
        # entry whose path is exactly the tracked gitlink path (NO trailing slash).
        (parent / "vendor" / "sub" / "a.txt").write_text("dirty submodule\n", encoding="utf-8")
        snap = capture_source_snapshot(parent, allow_dirty=True)
        assert snap.evidence is not None
        # The dirty submodule content is NOT snapshotted → evidence must be partial.
        assert snap.evidence.completeness == "partial"
        assert any("dirty_submodules_not_snapshotted" in lim for lim in snap.evidence.limitations)
        assert "dirty_submodule_content_not_captured" in snap.evidence.warnings

    def test_tracked_gitlink_paths_reads_mode_160000(self, tmp_path: Path) -> None:
        # The helper returns ONLY mode 160000 entries, not regular blobs/trees.
        from expertforge.provenance.source_snapshot import _tracked_gitlink_paths

        sub = tmp_path / "sub"
        sub.mkdir()
        _init_repo(sub)
        parent = tmp_path / "parent"
        parent.mkdir()
        _init_repo(parent)
        _add_submodule(parent, str(sub.resolve()).replace("\\", "/"), "vendor/sub")
        subprocess.run(["git", "commit", "-q", "-m", "add submodule"], cwd=parent, check=True)
        gitlinks = _tracked_gitlink_paths(parent)
        # The gitlink path is the submodule path (no trailing slash).
        assert gitlinks == {"vendor/sub"}

    def test_dirty_gitlink_intersection_logic_with_synthetic_data(self) -> None:
        # Synthetic unit test of the intersection logic so the regression holds
        # even where real submodules are unavailable. A porcelain entry whose
        # path matches a tracked gitlink with a non-trivial XY → dirty gitlink.
        gitlinks = {"vendor/sub", "third_party/lib"}
        # (xy, path, orig_path)
        porcelain = [
            (" M", "vendor/sub", None),  # dirty gitlink (worktree change)
            ("M ", "src/file.py", None),  # regular modified file
            ("??", "new.txt", None),  # untracked file
        ]
        dirty_gitlink_paths: set[str] = set()
        for xy, path, _orig in porcelain:
            x, y = xy[0], xy[1]
            if path in gitlinks and (x not in (" ", "?") or y not in (" ", "?")):
                dirty_gitlink_paths.add(path)
        assert dirty_gitlink_paths == {"vendor/sub"}

    def test_no_trailing_slash_required_for_submodule_detection(self, tmp_path: Path) -> None:
        # Explicitly assert that git porcelain emits the bare gitlink path for a
        # dirty submodule (the trailing-slash heuristic from before this fix
        # would have MISSED this entry).
        sub = tmp_path / "sub"
        sub.mkdir()
        _init_repo(sub)
        parent = tmp_path / "parent"
        parent.mkdir()
        _init_repo(parent)
        _add_submodule(parent, str(sub.resolve()).replace("\\", "/"), "vendor/sub")
        subprocess.run(["git", "commit", "-q", "-m", "add submodule"], cwd=parent, check=True)
        (parent / "vendor" / "sub" / "a.txt").write_text("dirty\n", encoding="utf-8")
        out = subprocess.run(
            ["git", "status", "--porcelain", "-z", "--untracked-files=all"],
            cwd=parent,
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8")
        # The porcelain entry must be " M vendor/sub" with NO trailing slash.
        assert "vendor/sub" in out
        assert "vendor/sub/" not in out

    def test_untracked_content_inside_submodule_marks_partial(self, tmp_path: Path) -> None:
        # Review item 1: untracked content inside an INITIALIZED submodule must
        # also surface as a dirty/partial evidence (not just modified tracked
        # content). Git porcelain flags a submodule with untracked content using
        # a "+" in the worktree column (" m" or "?? " variants); the gitlink
        # intersection must still detect it.
        sub = tmp_path / "sub"
        sub.mkdir()
        _init_repo(sub)
        parent = tmp_path / "parent"
        parent.mkdir()
        _init_repo(parent)
        _add_submodule(parent, str(sub.resolve()).replace("\\", "/"), "vendor/sub")
        subprocess.run(["git", "commit", "-q", "-m", "add submodule"], cwd=parent, check=True)
        # Add an UNTRACKED file inside the submodule (no tracked file modified).
        (parent / "vendor" / "sub" / "brand_new_untracked.txt").write_text(
            "untracked inside submodule\n", encoding="utf-8"
        )
        snap = capture_source_snapshot(parent, allow_dirty=True)
        assert snap.evidence is not None
        # The submodule's working tree content differs from what this snapshot
        # captures → evidence must be partial with the dirty-submodule limitation.
        assert snap.evidence.completeness == "partial"
        assert any("dirty_submodules_not_snapshotted" in lim for lim in snap.evidence.limitations)
        assert "dirty_submodule_content_not_captured" in snap.evidence.warnings


class TestGitlinkDiscoveryFailsClosed:
    """Review item 1: gitlink discovery must FAIL CLOSED. If
    ``git ls-files --stage`` fails, the evidence is marked partial with a
    ``gitlink_discovery_failed`` limitation rather than silently continuing with
    an empty gitlink set (which would hide uninspected dirty submodules)."""

    def test_tracked_gitlink_paths_raises_on_git_failure(self, tmp_path: Path) -> None:
        # Not a git repo → ``git ls-files --stage`` fails → the helper now
        # RAISES SourceUnavailableError (it used to return an empty set).
        from expertforge.provenance.source_snapshot import _tracked_gitlink_paths

        with pytest.raises(SourceUnavailableError):
            _tracked_gitlink_paths(tmp_path)

    def test_gitlink_discovery_failure_marks_evidence_partial(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force ``git ls-files --stage`` to fail mid-capture by patching the
        # underlying git runner so it raises ONLY for the ls-files call. The
        # resulting dirty evidence must be partial with gitlink_discovery_failed.
        import expertforge.provenance.source_snapshot as ss

        _init_repo(tmp_path)
        (tmp_path / "a.txt").write_text("dirty\n", encoding="utf-8")
        real_run_git = ss._run_git

        def _failing_ls_files(repo: Path, args: list[str]) -> bytes:
            if args[:2] == ["ls-files", "--stage"]:
                raise SourceUnavailableError("forced ls-files failure for test")
            return real_run_git(repo, args)

        monkeypatch.setattr(ss, "_run_git", _failing_ls_files)
        snap = capture_source_snapshot(tmp_path, allow_dirty=True)
        assert snap.evidence is not None
        assert snap.evidence.completeness == "partial"
        assert "gitlink_discovery_failed" in snap.evidence.limitations
        assert "gitlink_discovery_failed" in snap.evidence.warnings


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


# --- SourceEvidence canonical model invariants (review item 2) -------------


def _make_evidence(**overrides: object) -> object:
    """Build a minimal valid SourceEvidence; callers override fields.

    The dict splat loses per-field typing; the cast is the accepted pattern for
    test fixtures that intentionally exercise the validator with arbitrary
    override shapes.
    """
    from expertforge.provenance.source_snapshot import (
        SourceCounts,
        SourceEvidence,
    )

    base: dict[str, object] = dict(
        staged_digest="a" * 64,
        unstaged_digest="b" * 64,
        counts=SourceCounts(),
        completeness="complete",
    )
    base.update(overrides)
    return SourceEvidence(**cast("dict[str, Any]", base))


class TestSourceEvidenceInvariants:
    def test_untracked_must_be_sorted_by_path(self) -> None:
        from expertforge.provenance.source_snapshot import (
            SourceCounts,
            SourceEvidence,
            UntrackedEntry,
        )

        # Out-of-order paths.
        untracked = (
            UntrackedEntry(path="z.txt", kind="file", mode="0o644", digest="0" * 64),
            UntrackedEntry(path="a.txt", kind="file", mode="0o644", digest="0" * 64),
        )
        with pytest.raises(ValidationError):
            SourceEvidence(
                staged_digest="a" * 64,
                unstaged_digest="b" * 64,
                untracked=untracked,
                counts=SourceCounts(untracked=2),
                completeness="partial",
            )

    def test_untracked_duplicate_paths_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import (
            SourceCounts,
            SourceEvidence,
            UntrackedEntry,
        )

        untracked = (
            UntrackedEntry(path="a.txt", kind="file", mode="0o644", digest="0" * 64),
            UntrackedEntry(path="a.txt", kind="file", mode="0o644", digest="0" * 64),
        )
        with pytest.raises(ValidationError):
            SourceEvidence(
                staged_digest="a" * 64,
                unstaged_digest="b" * 64,
                untracked=untracked,
                counts=SourceCounts(untracked=2),
                completeness="partial",
            )

    def test_submodule_status_must_be_sorted_by_name(self) -> None:
        from expertforge.provenance.source_snapshot import (
            SourceCounts,
            SourceEvidence,
            SubmoduleEntry,
        )

        with pytest.raises(ValidationError):
            SourceEvidence(
                staged_digest="a" * 64,
                unstaged_digest="b" * 64,
                submodule_status=(
                    SubmoduleEntry(name="z_sub", commit="1" * 40, state="initialized"),
                    SubmoduleEntry(name="a_sub", commit="2" * 40, state="initialized"),
                ),
                counts=SourceCounts(submodules=2),
                completeness="complete",
            )

    def test_counts_untracked_must_match_len_untracked(self) -> None:
        from expertforge.provenance.source_snapshot import (
            SourceCounts,
            SourceEvidence,
            UntrackedEntry,
        )

        with pytest.raises(ValidationError):
            SourceEvidence(
                staged_digest="a" * 64,
                unstaged_digest="b" * 64,
                untracked=(
                    UntrackedEntry(path="a.txt", kind="file", mode="0o644", digest="0" * 64),
                ),
                counts=SourceCounts(untracked=2),  # mismatch: 1 entry, claims 2
                completeness="partial",
            )

    def test_counts_submodules_must_match_len_submodule_status(self) -> None:
        from expertforge.provenance.source_snapshot import (
            SourceCounts,
            SourceEvidence,
            SubmoduleEntry,
        )

        with pytest.raises(ValidationError):
            SourceEvidence(
                staged_digest="a" * 64,
                unstaged_digest="b" * 64,
                submodule_status=(SubmoduleEntry(name="s", commit="1" * 40, state="initialized"),),
                counts=SourceCounts(submodules=2),  # mismatch
                completeness="complete",
            )

    def test_complete_rejects_unreadable_untracked_entry(self) -> None:
        from expertforge.provenance.source_snapshot import (
            SourceCounts,
            SourceEvidence,
            UntrackedEntry,
        )

        with pytest.raises(ValidationError):
            SourceEvidence(
                staged_digest="a" * 64,
                unstaged_digest="b" * 64,
                untracked=(UntrackedEntry(path="a.txt", kind="file", digest=None),),
                counts=SourceCounts(untracked=1),
                completeness="complete",  # contradiction: digest=None means partial
            )

    def test_complete_rejects_warnings(self) -> None:
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="complete",
                warnings=("some_warning",),
            )

    def test_complete_rejects_non_standard_limitations(self) -> None:
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="complete",
                limitations=("some_hidden_incompleteness",),
            )

    def test_complete_allows_standard_limitations(self) -> None:
        # The two always-present standard limitations are allowed with complete.
        evidence = _make_evidence(
            completeness="complete",
            limitations=(
                "external_symlink_targets_not_followed",
                "ignored_files_not_included",
            ),
        )
        assert evidence is not None

    def test_warnings_must_be_sorted(self) -> None:
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="partial",
                warnings=("z_warn", "a_warn"),
            )

    def test_warnings_must_be_unique(self) -> None:
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="partial",
                warnings=("a_warn", "a_warn"),
            )

    def test_limitations_must_be_sorted(self) -> None:
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="partial",
                limitations=("z_lim", "a_lim"),
            )


# --- typed warning/limitation code vocabularies (review item 2) ------------


class TestEvidenceCodeVocabularies:
    """Review item 2: SourceEvidence.warnings and .limitations use CLOSED code
    vocabularies. Free-text / unrecognized codes are rejected so the contract is
    stable and tamper-evident."""

    def test_unknown_warning_code_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="partial",
                warnings=("made_up_warning",),
            )

    def test_known_warning_codes_accepted(self) -> None:
        evidence = _make_evidence(
            completeness="partial",
            warnings=(
                "dirty_submodule_content_not_captured",
                "gitlink_discovery_failed",
            ),
            limitations=(
                "dirty_submodules_not_snapshotted:1",
                "external_symlink_targets_not_followed",
                "gitlink_discovery_failed",
                "ignored_files_not_included",
            ),
        )
        assert evidence is not None

    def test_unknown_limitation_code_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="partial",
                limitations=("fabricated_limitation",),
            )

    def test_limitation_count_suffix_must_be_positive_int(self) -> None:
        # ``unreadable_untracked_files:N`` with a non-numeric or non-positive
        # suffix is rejected. A zero count is nonsensical (zero unreadable files
        # is not a real limitation).
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="partial",
                limitations=("unreadable_untracked_files:not-a-number",),
            )
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="partial",
                limitations=("dirty_submodules_not_snapshotted:-1",),
            )
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="partial",
                limitations=("unreadable_untracked_files:0",),
            )
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="partial",
                limitations=("dirty_submodules_not_snapshotted:0",),
            )

    def test_warning_codes_are_unique_and_sorted_still_enforced(self) -> None:
        # The closed vocabulary does not relax the sorted/unique requirement.
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="partial",
                warnings=(
                    "unreadable_untracked_content",
                    "submodule_inspection_failed",
                ),  # out of sorted order
            )


# --- SubmoduleEntry validators (review item 2) -----------------------------


class TestSubmoduleEntryValidators:
    """Review item 2: SubmoduleEntry.commit is 40 lowercase hex or "unknown";
    SubmoduleEntry.name is a canonical repo-relative POSIX path."""

    def test_commit_sha1_accepted(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        entry = SubmoduleEntry(name="vendor/sub", commit="1" * 40, state="initialized")
        assert entry.commit == "1" * 40

    def test_commit_unknown_accepted(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        entry = SubmoduleEntry(name="vendor/sub", commit="unknown", state="uninitialized")
        assert entry.commit == "unknown"

    def test_commit_truncated_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        with pytest.raises(ValidationError):
            SubmoduleEntry(name="vendor/sub", commit="1" * 39, state="initialized")

    def test_commit_uppercase_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        with pytest.raises(ValidationError):
            SubmoduleEntry(name="vendor/sub", commit="A" * 40, state="initialized")

    def test_commit_sha256_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        # 64 hex chars is a SHA-256, not a SHA-1 submodule commit.
        with pytest.raises(ValidationError):
            SubmoduleEntry(name="vendor/sub", commit="1" * 64, state="initialized")

    def test_commit_non_hex_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        with pytest.raises(ValidationError):
            SubmoduleEntry(name="vendor/sub", commit="z" * 40, state="initialized")

    def test_name_canonical_rel_path(self) -> None:
        # Backslashes are normalized; the name is stored in canonical POSIX form.
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        entry = SubmoduleEntry(name="vendor\\deep\\sub", commit="1" * 40, state="initialized")
        assert entry.name == "vendor/deep/sub"

    def test_name_dot_component_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        with pytest.raises(ValidationError):
            SubmoduleEntry(name="./vendor/sub", commit="1" * 40, state="initialized")

    def test_name_traversal_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        with pytest.raises(ValidationError):
            SubmoduleEntry(name="../escape/sub", commit="1" * 40, state="initialized")

    def test_name_absolute_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        with pytest.raises(ValidationError):
            SubmoduleEntry(name="/abs/sub", commit="1" * 40, state="initialized")


# --- UntrackedEntry cross-field consistency (review item 2) -----------------


class TestUntrackedEntryKindConsistency:
    """Review item 2: UntrackedEntry cross-field consistency between kind and
    mode/digest."""

    def test_file_requires_mode(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path="a.txt", kind="file", digest="0" * 64)  # no mode

    def test_file_with_mode_accepted(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        entry = UntrackedEntry(path="a.txt", kind="file", mode="0o644", digest="0" * 64)
        assert entry.mode == "0o644"

    def test_symlink_requires_mode(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path="a.txt", kind="symlink", digest="0" * 64)  # no mode

    def test_symlink_with_mode_and_digest_accepted(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        entry = UntrackedEntry(path="link", kind="symlink", mode="0o777", digest="0" * 64)
        assert entry.kind == "symlink"

    def test_unknown_rejects_mode(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path="a.txt", kind="unknown", mode="0o644")

    def test_unknown_rejects_digest(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path="a.txt", kind="unknown", digest="0" * 64)

    def test_unknown_with_none_mode_and_digest_accepted(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        entry = UntrackedEntry(path="a.txt", kind="unknown", mode=None, digest=None)
        assert entry.kind == "unknown"

    def test_dot_path_component_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path="./a.txt", kind="file", mode="0o644", digest="0" * 64)
        with pytest.raises(ValidationError):
            UntrackedEntry(path="a/./b.txt", kind="file", mode="0o644", digest="0" * 64)


# --- canonical UntrackedEntry.path validator (review item 2) ---------------


class TestCanonicalUntrackedPath:
    def test_backslash_normalized_to_forward_slash(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        entry = UntrackedEntry(
            path="sub\\dir\\file.txt", kind="file", mode="0o644", digest="0" * 64
        )
        assert entry.path == "sub/dir/file.txt"

    def test_empty_path_component_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path="a//b", kind="file", digest="0" * 64)
        with pytest.raises(ValidationError):
            UntrackedEntry(path="a/", kind="file", digest="0" * 64)

    def test_leading_slash_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path="/abs/path", kind="file", digest="0" * 64)

    def test_traversal_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path="sub/../escape", kind="file", digest="0" * 64)


# --- remote URL sanitization correlation (review item 3) -------------------


def _envelope_digest_for(tree: str, evidence: object) -> str:
    from expertforge.provenance.source_snapshot import _envelope_digest

    return _envelope_digest(tree, evidence)  # type: ignore[arg-type]


class TestRemoteUrlSanitizationClassifier:
    """The classifier (:func:`_classify_remote_url_sanitization`) is the SOLE
    place where URL↔warning correlation is performed — it sanitizes the raw URL
    and derives the matching warning codes. The capture path
    (:func:`capture_source_snapshot`) calls it and supplies BOTH the sanitized
    URL and the codes to the :class:`SourceSnapshot` constructor. The model
    itself does NOT correlate ``remote_url`` with ``remote_warnings``: that
    correlation cannot survive a sidecar round-trip (the stored URL is already
    sanitized, so re-deriving required warnings on load would reject the
    retained warnings as fabricated)."""

    def test_query_removal_emits_warning_and_sanitizes(self) -> None:
        from expertforge.provenance.source_snapshot import (
            _classify_remote_url_sanitization,
        )

        sanitized, codes = _classify_remote_url_sanitization(
            "https://github.com/org/repo.git?signed=xyz"
        )
        assert sanitized == "https://github.com/org/repo.git"
        assert codes == ("remote_query_fragment_removed",)

    def test_fragment_removal_emits_warning_and_sanitizes(self) -> None:
        from expertforge.provenance.source_snapshot import (
            _classify_remote_url_sanitization,
        )

        sanitized, codes = _classify_remote_url_sanitization("https://github.com/org/repo.git#frag")
        assert sanitized == "https://github.com/org/repo.git"
        assert codes == ("remote_query_fragment_removed",)

    def test_local_url_removal_emits_warning_and_returns_none(self) -> None:
        from expertforge.provenance.source_snapshot import (
            _classify_remote_url_sanitization,
        )

        sanitized, codes = _classify_remote_url_sanitization("/home/secret/repos/ExpertForge")
        assert sanitized is None
        assert codes == ("remote_local_or_unsupported_removed",)

    def test_scp_style_remote_not_flagged_as_credentials(self) -> None:
        # SCP-style ``git@github.com:org/repo.git`` has NO scheme; the ``git``
        # part is a protocol user, NOT a credential. It must NOT trigger
        # ``remote_credentials_removed`` and must be preserved verbatim.
        from expertforge.provenance.source_snapshot import (
            _classify_remote_url_sanitization,
        )

        sanitized, codes = _classify_remote_url_sanitization(
            "git@github.com:ElephantRock/ExpertForge.git"
        )
        assert sanitized == "git@github.com:ElephantRock/ExpertForge.git"
        assert "remote_credentials_removed" not in codes

    def test_https_with_userinfo_flagged_as_credentials(self) -> None:
        # ``https://user:token@host`` DOES carry credentials and is flagged.
        from expertforge.provenance.source_snapshot import (
            _classify_remote_url_sanitization,
        )

        sanitized, codes = _classify_remote_url_sanitization(
            "https://user:token@github.com/org/repo.git"
        )
        assert sanitized == "https://github.com/org/repo.git"
        assert "remote_credentials_removed" in codes

    def test_https_git_at_host_flagged_as_credentials(self) -> None:
        # Regression for the credential-classifier bug:
        # ``https://git@host/repo.git`` has scheme + userinfo, so its ``@`` IS
        # credential-bearing. The SCP-style exemption (``git@host:path``,
        # scheme-less) must NOT apply when ``://`` is present.
        from expertforge.provenance.source_snapshot import (
            _classify_remote_url_sanitization,
        )

        sanitized, codes = _classify_remote_url_sanitization("https://git@host/repo.git")
        assert sanitized == "https://host/repo.git"
        assert "remote_credentials_removed" in codes

    def test_empty_url_returns_none_and_no_warnings(self) -> None:
        from expertforge.provenance.source_snapshot import (
            _classify_remote_url_sanitization,
        )

        sanitized, codes = _classify_remote_url_sanitization("")
        assert sanitized is None
        assert codes == ()


class TestSourceSnapshotAcceptsSanitizedRemotes:
    """Review item 1: the :class:`SourceSnapshot` model accepts whatever
    ``remote_url`` and ``remote_warnings`` are supplied. The codes are
    Literal-validated, so a tampered sidecar cannot smuggle in unknown warning
    text — but the model does NOT re-derive redactions from the (already-
    sanitized) URL on load. The capture path / classifier is the sole source
    of truth for the correlation."""

    def test_clean_url_no_warnings_accepted(self) -> None:
        from expertforge.provenance.source_snapshot import SourceSnapshot

        tree = "c" * 64
        snap = SourceSnapshot(
            commit_sha="1" * 40,
            is_clean=True,
            is_canonical=True,
            remote_url="https://github.com/org/repo.git",
            remote_warnings=(),
            tree_digest=tree,
            input_digest=_envelope_digest_for(tree, None),
        )
        assert snap.remote_url == "https://github.com/org/repo.git"
        assert snap.remote_warnings == ()

    def test_none_url_no_warnings_accepted(self) -> None:
        from expertforge.provenance.source_snapshot import SourceSnapshot

        tree = "c" * 64
        snap = SourceSnapshot(
            commit_sha="1" * 40,
            is_clean=True,
            is_canonical=True,
            remote_url=None,
            remote_warnings=(),
            tree_digest=tree,
            input_digest=_envelope_digest_for(tree, None),
        )
        assert snap.remote_url is None
        assert snap.remote_warnings == ()

    def test_sanitized_url_with_matching_warning_accepted(self) -> None:
        # The model accepts the sanitized URL + the retained warning code that
        # the capture path produced. No re-derivation on load.
        from expertforge.provenance.source_snapshot import (
            RemoteWarning,
            SourceSnapshot,
        )

        tree = "c" * 64
        snap = SourceSnapshot(
            commit_sha="1" * 40,
            is_clean=True,
            is_canonical=True,
            remote_url="https://github.com/org/repo.git",
            remote_warnings=(RemoteWarning(code="remote_query_fragment_removed"),),
            tree_digest=tree,
            input_digest=_envelope_digest_for(tree, None),
        )
        assert snap.remote_url == "https://github.com/org/repo.git"
        assert {w.code for w in snap.remote_warnings} == {"remote_query_fragment_removed"}

    def test_none_url_with_warnings_accepted_by_model(self) -> None:
        # The model no longer correlates: a None URL with warnings is accepted
        # at the model level. (The capture path would never produce this
        # combination, but a tampered/edited sidecar is not rejected by the
        # model — the codes are still Literal-validated.)
        from expertforge.provenance.source_snapshot import (
            RemoteWarning,
            SourceSnapshot,
        )

        tree = "c" * 64
        snap = SourceSnapshot(
            commit_sha="1" * 40,
            is_clean=True,
            is_canonical=True,
            remote_url=None,
            remote_warnings=(RemoteWarning(code="remote_credentials_removed"),),
            tree_digest=tree,
            input_digest=_envelope_digest_for(tree, None),
        )
        assert snap.remote_url is None
        assert {w.code for w in snap.remote_warnings} == {"remote_credentials_removed"}

    def test_scp_style_remote_preserved_verbatim(self) -> None:
        from expertforge.provenance.source_snapshot import SourceSnapshot

        tree = "c" * 64
        snap = SourceSnapshot(
            commit_sha="1" * 40,
            is_clean=True,
            is_canonical=True,
            remote_url="git@github.com:ElephantRock/ExpertForge.git",
            remote_warnings=(),
            tree_digest=tree,
            input_digest=_envelope_digest_for(tree, None),
        )
        assert snap.remote_url == "git@github.com:ElephantRock/ExpertForge.git"
        assert snap.remote_warnings == ()

    def test_remote_warnings_round_trip_through_json(self) -> None:
        # The codes are Literal-validated and durable: they survive a JSON
        # round-trip on a snapshot whose URL is already sanitized.
        from expertforge.provenance.source_snapshot import (
            RemoteWarning,
            SourceSnapshot,
        )

        tree = "c" * 64
        snap = SourceSnapshot(
            commit_sha="1" * 40,
            is_clean=True,
            is_canonical=True,
            remote_url="https://github.com/org/repo.git",
            remote_warnings=(
                RemoteWarning(code="remote_credentials_removed"),
                RemoteWarning(code="remote_query_fragment_removed"),
            ),
            tree_digest=tree,
            input_digest=_envelope_digest_for(tree, None),
        )
        restored = SourceSnapshot.model_validate_json(snap.model_dump_json())
        assert restored.remote_url == "https://github.com/org/repo.git"
        assert {w.code for w in restored.remote_warnings} == {
            "remote_credentials_removed",
            "remote_query_fragment_removed",
        }


# --- SourceEvidence tightened behavioral invariants (review item 2) ---------


class TestEvidencePartialRequiresExplanation:
    """Review item 2: ``completeness="partial"`` requires at least one warning
    OR one non-standard limitation (beyond the two always-present standard
    limitations). A partial record with neither is rejected — an explanation
    is mandatory."""

    def test_partial_with_warning_accepted(self) -> None:
        evidence = _make_evidence(
            completeness="partial",
            warnings=("unreadable_untracked_content",),
            limitations=("unreadable_untracked_files:1",),
        )
        assert evidence is not None

    def test_partial_with_non_standard_limitation_accepted(self) -> None:
        evidence = _make_evidence(
            completeness="partial",
            limitations=(
                "dirty_submodules_not_snapshotted:1",
                "external_symlink_targets_not_followed",
                "ignored_files_not_included",
            ),
            warnings=("dirty_submodule_content_not_captured",),
        )
        assert evidence is not None

    def test_partial_with_only_standard_limitations_rejected(self) -> None:
        # The two standard limitations do NOT explain partial-ness by
        # themselves — an explanation is mandatory.
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="partial",
                limitations=(
                    "external_symlink_targets_not_followed",
                    "ignored_files_not_included",
                ),
            )

    def test_partial_with_no_warnings_no_limitations_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_evidence(completeness="partial")


class TestEvidenceDirtySubmoduleCorrelation:
    """Review item 2: the ``dirty_submodule_content_not_captured`` warning and
    the ``dirty_submodules_not_snapshotted:N`` limitation describe the SAME
    condition; one cannot appear without the other."""

    def test_warning_without_limitation_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="partial",
                warnings=("dirty_submodule_content_not_captured",),
            )

    def test_limitation_without_warning_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_evidence(
                completeness="partial",
                limitations=(
                    "dirty_submodules_not_snapshotted:1",
                    "external_symlink_targets_not_followed",
                    "ignored_files_not_included",
                ),
            )

    def test_warning_and_limitation_together_accepted(self) -> None:
        evidence = _make_evidence(
            completeness="partial",
            warnings=("dirty_submodule_content_not_captured",),
            limitations=(
                "dirty_submodules_not_snapshotted:2",
                "external_symlink_targets_not_followed",
                "ignored_files_not_included",
            ),
        )
        assert evidence is not None


class TestUntrackedEntryModeValidator:
    """Review item 2: ``UntrackedEntry.mode`` is validated as a canonical octal
    mode string (``^0o[0-7]{3,4}$``), not an arbitrary string."""

    def test_canonical_octal_mode_3_digits_accepted(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        entry = UntrackedEntry(path="a.txt", kind="file", mode="0o644", digest="0" * 64)
        assert entry.mode == "0o644"

    def test_canonical_octal_mode_4_digits_accepted(self) -> None:
        # 4-digit octal mode covers the setuid/setgid/sticky bits + rwxrwxrwx
        # (e.g. setuid 0o4755). The capture path emits oct(stat.S_IMODE(...))
        # which masks off the file-type bits, leaving at most 4 octal digits.
        from expertforge.provenance.source_snapshot import UntrackedEntry

        entry = UntrackedEntry(path="a.txt", kind="file", mode="0o4755", digest="0" * 64)
        assert entry.mode == "0o4755"

    def test_decimal_mode_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path="a.txt", kind="file", mode="644", digest="0" * 64)

    def test_non_octal_digit_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path="a.txt", kind="file", mode="0o844", digest="0" * 64)

    def test_missing_0o_prefix_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path="a.txt", kind="file", mode="644", digest="0" * 64)

    def test_too_many_digits_rejected(self) -> None:
        # 5+ octal digits (e.g. the full st_mode with file-type bits) is
        # rejected; only S_IMODE (3-4 octal digits) is canonical.
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path="a.txt", kind="file", mode="0o100644", digest="0" * 64)

    def test_too_few_digits_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path="a.txt", kind="file", mode="0o64", digest="0" * 64)


class TestUntrackedEntryOtherKindRejectsDigest:
    """Review item 2: ``kind="other"`` (a special/non-regular/non-symlink entry)
    must have ``digest=None`` (no content digest is meaningful for a non-file/
    non-symlink kind)."""

    def test_other_with_digest_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        with pytest.raises(ValidationError):
            UntrackedEntry(path="a.fifo", kind="other", mode="0o444", digest="0" * 64)

    def test_other_with_none_digest_accepted(self) -> None:
        from expertforge.provenance.source_snapshot import UntrackedEntry

        entry = UntrackedEntry(path="a.fifo", kind="other", mode="0o444", digest=None)
        assert entry.kind == "other"
        assert entry.digest is None


class TestSubmoduleEntryCommitStateCorrelation:
    """Review item 2: ``commit="unknown"`` is only valid with
    ``state="uninitialized"``. For initialized/changed/conflicted, commit must
    be a valid 40-hex SHA-1."""

    def test_unknown_with_initialized_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        with pytest.raises(ValidationError):
            SubmoduleEntry(name="vendor/sub", commit="unknown", state="initialized")

    def test_unknown_with_changed_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        with pytest.raises(ValidationError):
            SubmoduleEntry(name="vendor/sub", commit="unknown", state="changed")

    def test_unknown_with_conflicted_rejected(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        with pytest.raises(ValidationError):
            SubmoduleEntry(name="vendor/sub", commit="unknown", state="conflicted")

    def test_unknown_with_uninitialized_accepted(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        entry = SubmoduleEntry(name="vendor/sub", commit="unknown", state="uninitialized")
        assert entry.commit == "unknown"
        assert entry.state == "uninitialized"

    def test_sha1_with_initialized_accepted(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        entry = SubmoduleEntry(name="vendor/sub", commit="1" * 40, state="initialized")
        assert entry.commit == "1" * 40

    def test_sha1_with_changed_accepted(self) -> None:
        from expertforge.provenance.source_snapshot import SubmoduleEntry

        entry = SubmoduleEntry(name="vendor/sub", commit="1" * 40, state="changed")
        assert entry.state == "changed"
