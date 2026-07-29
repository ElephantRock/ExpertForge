"""Versioned canonical source-code content snapshot (Issue #7 decision: source identity).

The source snapshot is **behavioral identity**: it feeds an Issue #6
``ImmutableInput`` (name ``source.snapshot``) so materially different code
cannot run under the same configuration/dataset/tokenizer while retaining the
same specification fingerprint.

The behavioral-identity digest is **not** raw ``sha256(git ls-tree bytes)``. It
is the SHA-256 of a **versioned canonical envelope**::

    {
      "schema": "expertforge.source-snapshot",
      "version": <SOURCE_SNAPSHOT_VERSION>,
      "tree_digest": "<sha256 of git ls-tree -r -z --full-tree HEAD>",
      "evidence": null | { ... dirty evidence ... }
    }

For a clean tree ``evidence`` is ``null``. For a dirty tree (requires
``allow_dirty=True``) the evidence is a structured record with separate
staged/unstacked/untracked/submodule digests, file kinds/modes, symlink-target
digests (never following the link), counts, and completeness/limitations.

Git subprocess calls use bounded argument-list form, a deterministic locale
(``LC_ALL=C``), and a timeout. Raw stderr is never surfaced.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from expertforge.identity.fingerprint import ImmutableInput

__all__ = [
    "SOURCE_SNAPSHOT_VERSION",
    "DirtySourceError",
    "SourceEvidence",
    "SourceSnapshot",
    "SourceUnavailableError",
    "capture_source_snapshot",
    "source_snapshot_immutable_input",
]

SOURCE_SNAPSHOT_VERSION: int = 1
_SNAPSHOT_SCHEMA = "expertforge.source-snapshot"
_GIT_TIMEOUT: float = 10.0
# Deterministic locale environment for git subprocess calls.
_GIT_ENV = {**os.environ, "LC_ALL": "C", "LANG": "C"}


class DirtySourceError(Exception):
    """Raised when the working tree is dirty and ``allow_dirty`` is False."""


class SourceUnavailableError(Exception):
    """Raised when git is unavailable or the path is not a repository."""


# ---------------------------------------------------------------------------
# Git subprocess helpers
# ---------------------------------------------------------------------------


def _run_git(repo: Path, args: list[str]) -> bytes:
    """Run a bounded git subprocess; return stdout bytes.

    Argument-list form only (no shell). Deterministic locale. Bounded timeout.
    Raises :class:`SourceUnavailableError` on any failure; raw stderr is never
    surfaced.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            timeout=_GIT_TIMEOUT,
            check=False,
            env=_GIT_ENV,
        )
    except FileNotFoundError as e:
        raise SourceUnavailableError("git executable not found") from e
    except subprocess.TimeoutExpired as e:
        raise SourceUnavailableError("git command timed out") from e
    except (subprocess.SubprocessError, OSError) as e:
        raise SourceUnavailableError(f"git command failed: {type(e).__name__}") from e
    if result.returncode != 0:
        raise SourceUnavailableError("git command failed (not a repository?)")
    return result.stdout


def _git_text(repo: Path, args: list[str]) -> str:
    return _run_git(repo, args).decode("utf-8", errors="replace").strip()


# ---------------------------------------------------------------------------
# Clean tree digest
# ---------------------------------------------------------------------------


def _clean_tree_digest(repo: Path) -> str:
    """SHA-256 of the deterministic ``git ls-tree -r -z --full-tree HEAD`` bytes."""
    out = _run_git(repo, ["ls-tree", "-r", "-z", "--full-tree", "HEAD"])
    return hashlib.sha256(out).hexdigest()


def _head_sha(repo: Path) -> str:
    return _git_text(repo, ["rev-parse", "HEAD"])


def _branch_ref(repo: Path) -> str:
    """Current branch name, or ``HEAD`` when detached."""
    ref = _git_text(repo, ["rev-parse", "--abbrev-ref", "HEAD"])
    return ref if ref else "HEAD"


def _remote_url(repo: Path) -> str | None:
    """Best-effort ``origin`` remote URL (unsanitized; sanitization is the
    caller's responsibility — the provenance record applies it)."""
    try:
        url = _git_text(repo, ["config", "--get", "remote.origin.url"])
    except SourceUnavailableError:
        return None
    return url if url else None


# ---------------------------------------------------------------------------
# Dirty detection + evidence
# ---------------------------------------------------------------------------


def _is_dirty(repo: Path) -> bool:
    """True if tracked changes or untracked files exist."""
    out = _run_git(repo, ["status", "--porcelain", "-z", "--untracked-files=all"])
    return any(part.strip() for part in out.split(b"\x00"))


def _staged_diff_digest(repo: Path) -> str:
    """Deterministic digest of staged changes (binary, no textconv)."""
    out = _run_git(repo, ["diff", "--cached", "--no-textconv", "--binary"])
    return hashlib.sha256(out).hexdigest()


def _unstaged_diff_digest(repo: Path) -> str:
    """Deterministic digest of unstaged tracked changes (binary, no textconv)."""
    out = _run_git(repo, ["diff", "--no-textconv", "--binary"])
    return hashlib.sha256(out).hexdigest()


def _parse_porcelain_z(raw: bytes) -> list[tuple[str, str, str | None]]:
    """Parse ``git status --porcelain -z`` output into (xy, path, orig_path) triples.

    Handles rename/copy entries correctly: ``XY <new_path>\\0<old_path>\\0``.
    Returns a list of ``(status_xy, path, orig_path_or_None)``.
    """
    entries: list[tuple[str, str, str | None]] = []
    parts = raw.split(b"\x00")
    i = 0
    while i < len(parts):
        part = parts[i]
        if not part.strip():
            i += 1
            continue
        # Status is first 2 chars; path follows after a space.
        xy = part[:2].decode("ascii", errors="replace")
        path_bytes = part[3:]  # skip "XY "
        path = path_bytes.decode("utf-8", errors="replace")
        orig_path: str | None = None
        # Rename/copy: X or Y is 'R' or 'C' → next NUL part is the original path.
        if ("R" in xy or "C" in xy) and i + 1 < len(parts):
            orig_bytes = parts[i + 1]
            if orig_bytes.strip():
                orig_path = orig_bytes.decode("utf-8", errors="replace")
                i += 1  # consume the original-path part
        entries.append((xy, path, orig_path))
        i += 1
    return entries


def _file_kind_and_digest(repo: Path, rel_path: str) -> dict[str, Any]:
    """Record kind/mode/digest for a working-tree path WITHOUT following symlinks.

    - Regular file: ``kind="file"``, content digest via ``os.open(O_NOFOLLOW)``.
    - Symlink: ``kind="symlink"``, digest of the link target string (readlink).
    - Missing/other: ``kind="unknown"``, no digest.
    """
    abs_path = repo / rel_path
    # Use lstat to NOT follow symlinks.
    try:
        st = os.lstat(abs_path)
    except OSError:
        return {"path": rel_path, "kind": "unknown", "mode": None, "digest": None}

    import stat as stat_mod

    mode = stat_mod.S_IMODE(st.st_mode)
    if stat_mod.S_ISLNK(st.st_mode):
        # Symlink: record the target digest, never follow.
        try:
            target = os.readlink(abs_path)
            digest = hashlib.sha256(target.encode("utf-8")).hexdigest()
        except OSError:
            digest = None
        return {"path": rel_path, "kind": "symlink", "mode": oct(mode), "digest": digest}
    if stat_mod.S_ISREG(st.st_mode):
        # Regular file: open with O_NOFOLLOW to prevent following a swapped-in
        # symlink. O_NOFOLLOW is POSIX-only; on Windows symlinks require elevated
        # privileges and are not a practical attack surface here.
        open_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW
        try:
            fd = os.open(abs_path, open_flags)
            try:
                h = hashlib.sha256()
                while True:
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    h.update(chunk)
            finally:
                os.close(fd)
            return {"path": rel_path, "kind": "file", "mode": oct(mode), "digest": h.hexdigest()}
        except OSError:
            return {"path": rel_path, "kind": "file", "mode": oct(mode), "digest": None}
    # Directory or other (shouldn't happen with --untracked-files=all).
    return {"path": rel_path, "kind": "other", "mode": oct(mode), "digest": None}


def _build_dirty_evidence(repo: Path) -> SourceEvidence:
    """Build the structured dirty evidence record.

    Includes:
    - separate staged and unstaged deterministic diff hashes;
    - sorted untracked-file manifest with path/kind/mode/digest;
    - symlink targets (not followed);
    - submodule status;
    - counts and completeness/limitations.
    """
    status_raw = _run_git(repo, ["status", "--porcelain", "-z", "--untracked-files=all"])
    entries = _parse_porcelain_z(status_raw)

    staged_digest = _staged_diff_digest(repo)
    unstaged_digest = _unstaged_diff_digest(repo)

    # Partition entries by status.
    untracked: list[dict[str, Any]] = []
    staged_count = 0
    unstaged_count = 0
    renamed_count = 0
    deleted_count = 0
    for xy, path, _orig_path in entries:
        x, y = xy[0], xy[1]
        if x == "?" and y == "?":
            info = _file_kind_and_digest(repo, path)
            untracked.append(info)
        if x not in ("?", " ", "!", "_"):
            staged_count += 1
        if y not in ("?", " ", "!", "_"):
            unstaged_count += 1
        if "R" in xy or "C" in xy:
            renamed_count += 1
        if "D" in xy:
            deleted_count += 1

    # Sort untracked manifest by path for determinism.
    untracked.sort(key=lambda e: e["path"])

    # Submodule status (best-effort).
    submodule_status: list[dict[str, str]] = []
    try:
        sub_raw = _run_git(repo, ["submodule", "status"])
        for line in sub_raw.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    sha = parts[0].strip().lstrip("-+~")  # status prefix
                    name = parts[1]
                    submodule_status.append({"name": name, "commit": sha[:12]})
    except SourceUnavailableError:
        pass  # submodules unavailable — not an error

    return SourceEvidence(
        staged_digest=staged_digest,
        unstaged_digest=unstaged_digest,
        untracked=tuple(untracked),
        submodule_status=tuple(f"{s['name']}:{s['commit']}" for s in submodule_status),
        counts={
            "staged": staged_count,
            "unstaged": unstaged_count,
            "untracked": len(untracked),
            "renamed_or_copied": renamed_count,
            "deleted": deleted_count,
            "submodules": len(submodule_status),
        },
        completeness="complete",
        limitations=tuple(),
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SourceEvidence(BaseModel):
    """Structured dirty-source evidence (staged/unstaged/untracked/submodule)."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    staged_digest: str = Field(..., min_length=1)
    unstaged_digest: str = Field(..., min_length=1)
    untracked: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    submodule_status: tuple[str, ...] = Field(default_factory=tuple)
    counts: dict[str, int] = Field(default_factory=dict)
    completeness: str = Field(default="complete")
    limitations: tuple[str, ...] = Field(default_factory=tuple)


class SourceSnapshot(BaseModel):
    """The captured source-code content snapshot.

    The ``input_digest`` is the SHA-256 of the versioned canonical envelope —
    this is the value that enters the #6 ``ImmutableInput("source.snapshot")``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    snapshot_version: int = Field(default=SOURCE_SNAPSHOT_VERSION)
    commit_sha: str = Field(..., min_length=1)
    branch: str = Field(default="HEAD")
    remote_url: str | None = Field(default=None)
    is_clean: bool
    is_canonical: bool
    # SHA-256 of the committed tree (from git ls-tree).
    tree_digest: str = Field(..., min_length=1)
    # The versioned envelope digest — the behavioral-identity input.
    input_digest: str = Field(..., min_length=1)
    # Dirty evidence; present only when dirty.
    evidence: SourceEvidence | None = Field(default=None)

    @field_validator("snapshot_version")
    @classmethod
    def _validate_version(cls, v: int) -> int:
        if v != SOURCE_SNAPSHOT_VERSION:
            raise ValueError(
                f"Unsupported source-snapshot version {v}; this version "
                f"supports {SOURCE_SNAPSHOT_VERSION}."
            )
        return v


# ---------------------------------------------------------------------------
# Envelope canonicalization
# ---------------------------------------------------------------------------


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _envelope_digest(tree_digest: str, evidence: SourceEvidence | None) -> str:
    """Compute the versioned canonical envelope digest.

    The envelope includes the schema, version, tree digest, and evidence (or
    null for clean). This is the behavioral-identity input.
    """
    envelope: dict[str, Any] = {
        "schema": _SNAPSHOT_SCHEMA,
        "version": SOURCE_SNAPSHOT_VERSION,
        "tree_digest": tree_digest,
        "evidence": evidence.model_dump() if evidence else None,
    }
    return hashlib.sha256(_canonical_json_bytes(envelope)).hexdigest()


# ---------------------------------------------------------------------------
# Public capture API
# ---------------------------------------------------------------------------


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
    tree_digest = _clean_tree_digest(repo_path)
    commit_sha = _head_sha(repo_path)
    branch = _branch_ref(repo_path)
    remote_url = _remote_url(repo_path)
    evidence = _build_dirty_evidence(repo_path) if dirty else None
    input_digest = _envelope_digest(tree_digest, evidence)
    return SourceSnapshot(
        commit_sha=commit_sha,
        branch=branch,
        remote_url=remote_url,
        is_clean=not dirty,
        is_canonical=not dirty,
        tree_digest=tree_digest,
        input_digest=input_digest,
        evidence=evidence,
    )


def source_snapshot_immutable_input(snap: SourceSnapshot) -> ImmutableInput:
    """Return the source-snapshot input as a real Issue #6 ``ImmutableInput``.

    The behavioral-identity digest of a dirty tree is the versioned envelope
    digest (which includes dirty evidence); for a clean tree it is the clean
    envelope digest.
    """
    # Late import to avoid a circular dependency at module load time.
    from expertforge.identity.fingerprint import ImmutableInput

    return ImmutableInput(
        name="source.snapshot",
        algorithm="sha256",
        digest=snap.input_digest,
    )
