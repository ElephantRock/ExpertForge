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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
# Deterministic locale environment for git subprocess calls. Strip GIT_*
# environment variables that could alter diff behavior (external diff drivers,
# aliases, config overrides) — only carry safe locale settings.
_GIT_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
_GIT_ENV["LC_ALL"] = "C"
_GIT_ENV["LANG"] = "C"


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


def _remote_url(repo: Path) -> tuple[str | None, tuple[str, ...]]:
    """Capture the ``origin`` remote URL, **sanitized at capture time**.

    Returns ``(sanitized_url, warnings)``. If the raw URL contained credentials,
    query parameters, fragments, or was a local path, the sanitized value may
    be ``None`` and typed warnings are emitted. The raw URL is never exposed.
    """
    try:
        raw = _git_text(repo, ["config", "--get", "remote.origin.url"])
    except SourceUnavailableError:
        return None, ()
    if not raw:
        return None, ()
    warnings: list[str] = []
    # Detect credential-bearing URLs before sanitizing.
    if "@" in raw.split("://")[-1] if "://" in raw else "@" in raw:
        warnings.append("remote_credentials_removed")
    if "?" in raw or "#" in raw:
        warnings.append("remote_query_fragment_removed")
    # Apply structural sanitization.
    from expertforge.provenance.software import sanitize_repository_url

    sanitized = sanitize_repository_url(raw)
    if sanitized is None and raw:
        warnings.append("remote_local_or_unsupported_removed")
    return sanitized, tuple(warnings)


# ---------------------------------------------------------------------------
# Dirty detection + evidence
# ---------------------------------------------------------------------------


def _is_dirty(repo: Path) -> bool:
    """True if tracked changes or untracked files exist."""
    out = _run_git(repo, ["status", "--porcelain", "-z", "--untracked-files=all"])
    return any(part.strip() for part in out.split(b"\x00"))


def _staged_diff_digest(repo: Path) -> str:
    """Deterministic digest of staged changes (binary, no textconv, no ext-diff)."""
    out = _run_git(repo, ["diff", "--cached", "--no-textconv", "--no-ext-diff", "--binary"])
    return hashlib.sha256(out).hexdigest()


def _unstaged_diff_digest(repo: Path) -> str:
    """Deterministic digest of unstaged tracked changes (binary, no textconv, no ext-diff)."""
    out = _run_git(repo, ["diff", "--no-textconv", "--no-ext-diff", "--binary"])
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


def _file_kind_and_digest(repo: Path, rel_path: str) -> UntrackedEntry:
    """Record kind/mode/digest for a working-tree path WITHOUT following symlinks.

    Returns a typed ``UntrackedEntry``. ``digest`` is ``None`` when the file is
    unreadable — the caller uses that to derive honest ``partial`` completeness.
    """
    abs_path = repo / rel_path
    try:
        st = os.lstat(abs_path)
    except OSError:
        return UntrackedEntry(path=rel_path, kind="unknown", mode=None, digest=None)

    import stat as stat_mod

    mode = stat_mod.S_IMODE(st.st_mode)
    mode_str = oct(mode)
    if stat_mod.S_ISLNK(st.st_mode):
        try:
            target = os.readlink(abs_path)
            digest = hashlib.sha256(target.encode("utf-8")).hexdigest()
        except OSError:
            digest = None
        return UntrackedEntry(path=rel_path, kind="symlink", mode=mode_str, digest=digest)
    if stat_mod.S_ISREG(st.st_mode):
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
            return UntrackedEntry(path=rel_path, kind="file", mode=mode_str, digest=h.hexdigest())
        except OSError:
            return UntrackedEntry(path=rel_path, kind="file", mode=mode_str, digest=None)
    return UntrackedEntry(path=rel_path, kind="other", mode=mode_str, digest=None)


def _build_dirty_evidence(repo: Path) -> SourceEvidence:
    """Build the structured dirty evidence record with honest completeness.

    Unreadable files, failed submodule inspection, and external/ignored files
    produce typed limitations and ``partial`` completeness — never silently
    ``complete``.
    """
    status_raw = _run_git(repo, ["status", "--porcelain", "-z", "--untracked-files=all"])
    entries = _parse_porcelain_z(status_raw)

    staged_digest = _staged_diff_digest(repo)
    unstaged_digest = _unstaged_diff_digest(repo)

    untracked: list[UntrackedEntry] = []
    staged_count = 0
    unstaged_count = 0
    renamed_count = 0
    deleted_count = 0
    unreadable_count = 0
    for xy, path, _orig_path in entries:
        x, y = xy[0], xy[1]
        if x == "?" and y == "?":
            entry = _file_kind_and_digest(repo, path)
            untracked.append(entry)
            if entry.digest is None:
                unreadable_count += 1
        if x not in ("?", " ", "!", "_"):
            staged_count += 1
        if y not in ("?", " ", "!", "_"):
            unstaged_count += 1
        if "R" in xy or "C" in xy:
            renamed_count += 1
        if "D" in xy:
            deleted_count += 1

    untracked.sort(key=lambda e: e.path)

    # Submodule status — preserve full SHA + state prefix, record failures.
    submodule_entries: list[SubmoduleEntry] = []
    submodule_failed = False
    try:
        sub_raw = _run_git(repo, ["submodule", "status"])
        for line in sub_raw.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            prefix = line[0] if line[0] in "-+~ " else " "
            sha_full = line[1:41].strip() if len(line) > 40 else ""
            rest = line[42:].strip() if len(line) > 42 else ""
            name = rest.split()[0] if rest else "unknown"
            state_map = {
                " ": "initialized",
                "-": "uninitialized",
                "+": "changed",
                "~": "conflicted",
            }
            submodule_entries.append(
                SubmoduleEntry(
                    name=name,
                    commit=sha_full if sha_full else "unknown",
                    state=state_map.get(prefix, "initialized"),
                )
            )
    except SourceUnavailableError:
        submodule_failed = True

    # Derive honest completeness + limitations.
    limitations: list[str] = []
    warnings: list[str] = []
    if unreadable_count > 0:
        limitations.append(f"unreadable_untracked_files:{unreadable_count}")
        warnings.append("unreadable_untracked_content")
    if submodule_failed:
        limitations.append("submodule_inspection_failed")
        warnings.append("submodule_inspection_failed")
    # Dirty/changed submodules are not content-snapshotted by `git submodule
    # status` — mark evidence partial.
    dirty_submodules = [s for s in submodule_entries if s.state in ("changed", "conflicted")]
    if dirty_submodules:
        limitations.append(f"dirty_submodules_not_snapshotted:{len(dirty_submodules)}")
        warnings.append("dirty_submodule_content_not_captured")

    # Always note that ignored files and external symlink targets are not included.
    limitations.append("ignored_files_not_included")
    limitations.append("external_symlink_targets_not_followed")

    completeness = (
        "complete"
        if not unreadable_count and not submodule_failed and not dirty_submodules
        else "partial"
    )

    return SourceEvidence(
        staged_digest=staged_digest,
        unstaged_digest=unstaged_digest,
        untracked=tuple(untracked),
        submodule_status=tuple(submodule_entries),
        counts=SourceCounts(
            staged=staged_count,
            unstaged=unstaged_count,
            untracked=len(untracked),
            renamed_or_copied=renamed_count,
            deleted=deleted_count,
            submodules=len(submodule_entries),
        ),
        completeness=completeness,
        limitations=tuple(limitations),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class UntrackedEntry(BaseModel):
    """One untracked working-tree file with kind/mode/digest (no symlink following)."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    path: str = Field(..., min_length=1)
    kind: str  # file | symlink | unknown | other
    mode: str | None = Field(default=None)
    digest: str | None = Field(default=None)  # None when unreadable


class SubmoduleEntry(BaseModel):
    """One submodule's status with full commit SHA and state prefix preserved."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    name: str = Field(..., min_length=1)
    commit: str = Field(..., min_length=1)  # full SHA, not truncated
    state: str  # initialized | uninitialized | changed | conflicted (from prefix)


class SourceCounts(BaseModel):
    """Typed dirty-evidence counts."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    staged: int = Field(default=0, ge=0)
    unstaged: int = Field(default=0, ge=0)
    untracked: int = Field(default=0, ge=0)
    renamed_or_copied: int = Field(default=0, ge=0)
    deleted: int = Field(default=0, ge=0)
    submodules: int = Field(default=0, ge=0)


class SourceEvidence(BaseModel):
    """Structured dirty-source evidence (staged/unstaged/untracked/submodule).

    All containers are tuples of frozen models — deeply immutable.
    Completeness is honestly derived: ``partial`` when any file was unreadable
    or submodule inspection failed, with typed limitations/warnings.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    staged_digest: str = Field(..., min_length=1)
    unstaged_digest: str = Field(..., min_length=1)
    untracked: tuple[UntrackedEntry, ...] = Field(default_factory=tuple)
    submodule_status: tuple[SubmoduleEntry, ...] = Field(default_factory=tuple)
    counts: SourceCounts = Field(default_factory=SourceCounts)
    completeness: str = Field(default="complete")  # complete | partial | error
    limitations: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class RemoteWarning(BaseModel):
    """Typed warning about remote URL sanitization."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    code: str  # e.g. remote_credentials_removed, remote_local_or_unsupported_removed
    message: str = Field(default="")


class SourceSnapshot(BaseModel):
    """The captured source-code content snapshot.

    The ``input_digest`` is the SHA-256 of the versioned canonical envelope —
    this is the value that enters the #6 ``ImmutableInput("source.snapshot")``.
    The ``remote_url`` is sanitized at capture time; the raw URL is never exposed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    snapshot_version: int = Field(default=SOURCE_SNAPSHOT_VERSION)
    commit_sha: str = Field(..., min_length=1)
    branch: str = Field(default="HEAD")
    remote_url: str | None = Field(default=None)  # already sanitized at capture time
    remote_warnings: tuple[RemoteWarning, ...] = Field(default_factory=tuple)
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

    @model_validator(mode="after")
    def _verify_input_digest(self) -> SourceSnapshot:
        """Recompute the envelope digest from the stored fields and verify it
        matches ``input_digest``. Detects tampering of any envelope component."""
        recomputed = _envelope_digest(self.tree_digest, self.evidence)
        if recomputed != self.input_digest:
            raise ValueError(
                f"input_digest {self.input_digest!r} does not match recomputed "
                f"envelope digest {recomputed!r}."
            )
        return self


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
    sanitized_url, remote_warns = _remote_url(repo_path)
    evidence = _build_dirty_evidence(repo_path) if dirty else None
    input_digest = _envelope_digest(tree_digest, evidence)
    return SourceSnapshot(
        commit_sha=commit_sha,
        branch=branch,
        remote_url=sanitized_url,  # already sanitized at capture time
        remote_warnings=tuple(RemoteWarning(code=w) for w in remote_warns),
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
