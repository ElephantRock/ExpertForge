"""Source-code content snapshot (Issue #7 decision: source identity).

The source snapshot is **behavioral identity**: it is the one piece of
provenance that feeds an Issue #6 ``ImmutableInput`` (name ``source.snapshot``),
so that materially different code cannot run under the same configuration,
dataset, and tokenizer while retaining the same specification fingerprint.

For a clean repository, the snapshot digest is derived from deterministic tree
material (``git ls-tree -r -z --full-tree HEAD``) — not branch names, local
paths, timestamps, or commit history alone. The commit SHA is still recorded for
attribution.

Dirty execution:
- is rejected by default (:class:`DirtySourceError`);
- requires explicit ``allow_dirty=True``;
- is always marked non-canonical;
- uses a deterministic hash of tracked changes + untracked files;
- does not persist a raw patch.

Git is invoked through bounded, argument-list subprocess calls (no shell). When
git is unavailable or the path is not a repository, capture degrades explicitly.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "SOURCE_SNAPSHOT_VERSION",
    "DirtySourceError",
    "SourceSnapshot",
    "SourceUnavailableError",
    "capture_source_snapshot",
    "source_snapshot_immutable_input",
]

# The source-snapshot digest version. Bumped only when the digest's input
# bytes change (Issue #7 §version boundaries).
SOURCE_SNAPSHOT_VERSION: int = 1


class DirtySourceError(Exception):
    """Raised when the working tree is dirty and ``allow_dirty`` is False."""


class SourceUnavailableError(Exception):
    """Raised when git is unavailable or the path is not a repository."""


def _run_git(repo: Path, args: list[str], *, want_bytes: bool = False) -> bytes:
    """Run a bounded git subprocess in ``repo`` and return its stdout.

    Raises :class:`SourceUnavailableError` if git is missing, the path is not a
    repository, or git reports a non-zero exit. Argument-list form only; no
    shell. Raw stderr is never surfaced (Issue #7 §secret handling).
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise SourceUnavailableError("git executable not found") from e
    if result.returncode != 0:
        raise SourceUnavailableError("git command failed (not a repository?)")
    return result.stdout


def _is_dirty(repo: Path) -> bool:
    """True if the working tree has tracked changes or untracked files."""
    out = _run_git(repo, ["status", "--porcelain", "-z"])
    # `-z` separates entries with NUL; any non-empty output means dirty.
    return any(part.strip() for part in out.split(b"\x00"))


def _clean_tree_digest(repo: Path) -> str:
    """SHA-256 of the deterministic ``git ls-tree -r -z --full-tree HEAD`` bytes."""
    out = _run_git(repo, ["ls-tree", "-r", "-z", "--full-tree", "HEAD"])
    return hashlib.sha256(out).hexdigest()


def _head_sha(repo: Path) -> str:
    """The current HEAD commit SHA (attribution, not fingerprint input)."""
    return _run_git(repo, ["rev-parse", "HEAD"]).decode("utf-8").strip()


def _dirty_digest(repo: Path) -> str:
    """Deterministic digest of tracked changes + untracked files.

    Combines the ``git status --porcelain -z`` manifest (stable, sorted by git)
    with content hashes of each changed/untracked path. No raw patch or file
    contents are persisted — only digests. Stable across repeated captures of
    the same dirty state.
    """
    status = _run_git(repo, ["status", "--porcelain", "-z"])
    # Build a deterministic digest: manifest bytes + per-path staged-blob hash
    # where available, else a hash of the working-tree bytes.
    h = hashlib.sha256()
    h.update(status)
    for part in status.split(b"\x00"):
        part = part.strip()
        if not part:
            continue
        # ``--porcelain`` lines: "<XY> <path>" (path may be quoted for special chars).
        path_bytes = part[3:]  # skip "XY " (2 status chars + space)
        try:
            path = path_bytes.decode("utf-8")
        except UnicodeDecodeError:
            h.update(b"<undecodable-path>")
            continue
        file_path = repo / path
        if file_path.is_file():
            try:
                h.update(hashlib.sha256(file_path.read_bytes()).digest())
            except OSError:
                h.update(b"<unreadable>")
    return h.hexdigest()


class SourceSnapshot(BaseModel):
    """The captured source-code content snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    snapshot_version: int = Field(default=SOURCE_SNAPSHOT_VERSION)
    commit_sha: str = Field(..., min_length=1)
    is_clean: bool
    is_canonical: bool
    # Content digest of the committed tree (behavioral-identity input).
    content_digest: str = Field(..., min_length=1)
    # Digest of tracked changes + untracked files; present only when dirty.
    dirty_digest: str | None = Field(default=None)


def capture_source_snapshot(repo: Path | str, *, allow_dirty: bool = False) -> SourceSnapshot:
    """Capture the source snapshot for ``repo``.

    Raises :class:`DirtySourceError` if the tree is dirty and ``allow_dirty``
    is False. Raises :class:`SourceUnavailableError` if git is unavailable or
    the path is not a repository.
    """
    repo_path = Path(repo)
    dirty = _is_dirty(repo_path)
    if dirty and not allow_dirty:
        raise DirtySourceError(
            "Source working tree is dirty; pass allow_dirty=True to record a "
            "non-canonical snapshot."
        )
    content_digest = _clean_tree_digest(repo_path)
    commit_sha = _head_sha(repo_path)
    return SourceSnapshot(
        commit_sha=commit_sha,
        is_clean=not dirty,
        is_canonical=not dirty,
        content_digest=content_digest,
        dirty_digest=_dirty_digest(repo_path) if dirty else None,
    )


@dataclass(frozen=True)
class _ImmutableInputView:
    """Adapter so callers can construct the #6 ImmutableInput from a snapshot."""

    name: str
    algorithm: str
    digest: str


def source_snapshot_immutable_input(snap: SourceSnapshot) -> _ImmutableInputView:
    """Return the source-snapshot digest as an #6 ImmutableInput view.

    The behavioral-identity digest of a dirty tree is its dirty digest (so that
    different dirty states differ); for a clean tree it is the committed-tree
    content digest.
    """
    return _ImmutableInputView(
        name="source.snapshot",
        algorithm="sha256",
        digest=snap.dirty_digest or snap.content_digest,
    )
