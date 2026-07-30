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
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

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
# 64 lowercase hexadecimal characters (SHA-256). Used to validate all digest
# fields on the source-snapshot models so a tampered sidecar cannot smuggle in
# a malformed digest.
_DIGEST_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
# 40 lowercase hexadecimal characters (SHA-1). Git submodule commits are full
# SHA-1 object names; a tampered sidecar must not smuggle in a truncated or
# malformed commit.
_SHA1_HEX_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _normalize_canonical_rel_path(value: str, label: str) -> str:
    """Enforce a canonical repository-relative POSIX path.

    Backslashes are normalized to forward slashes (Windows porcelain output may
    use ``\\``). The normalized path must NOT be absolute (POSIX leading ``/`` or
    Windows drive-letter form), must NOT contain a ``..`` OR ``.`` component,
    and must NOT contain empty components (``a//b`` or a trailing slash). Shared
    by :class:`UntrackedEntry.path` and :class:`SubmoduleEntry.name`.
    """
    normalized = value.replace("\\", "/")
    if not normalized:
        raise ValueError(f"{label} must be non-empty.")
    # Reject POSIX absolute form.
    if normalized.startswith("/"):
        raise ValueError(f"{label} {value!r} must be repo-relative, not absolute.")
    # Reject Windows drive-letter absolute form (C:/ — backslashes already
    # normalized to forward slashes above).
    if re.match(r"^[A-Za-z]:/", normalized):
        raise ValueError(f"{label} {value!r} must be repo-relative, not absolute.")
    parts = normalized.split("/")
    # Reject any traversal or self-referential component anywhere in the path.
    if any(part in ("..", ".") for part in parts):
        raise ValueError(f"{label} {value!r} contains a '.' or '..' component.")
    # Reject empty components (``a//b``, leading or trailing slash).
    if any(part == "" for part in parts):
        raise ValueError(f"{label} {value!r} contains an empty path component.")
    return normalized


# Deterministic locale environment for git subprocess calls. Strip GIT_*
# environment variables that could alter diff behavior (external diff drivers,
# aliases, config overrides) — only carry safe locale settings.
_GIT_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
_GIT_ENV["LC_ALL"] = "C"
_GIT_ENV["LANG"] = "C"


def _validate_optional_digest(v: str | None) -> str | None:
    """Allow None (nullable) or an exact 64-lowercase-hex SHA-256 digest."""
    if v is None:
        return None
    if not _DIGEST_HEX_PATTERN.fullmatch(v):
        raise ValueError(f"digest must be 64 lowercase hex chars (sha256); got {v!r}.")
    return v


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


_RemoteWarningCode = Literal[
    "remote_credentials_removed",
    "remote_query_fragment_removed",
    "remote_local_or_unsupported_removed",
]


def _raw_remote_url(repo: Path) -> str:
    """Read the raw ``origin`` remote URL (NOT sanitized) from git config.

    Returns the empty string if no origin is configured. The caller is
    responsible for sanitizing — the raw value is intentionally exposed to the
    SourceSnapshot model_validator which can correlate sanitization with
    ``remote_warnings``.
    """
    try:
        return _git_text(repo, ["config", "--get", "remote.origin.url"])
    except SourceUnavailableError:
        return ""


def _classify_remote_url_sanitization(
    raw: str,
) -> tuple[str | None, tuple[_RemoteWarningCode, ...]]:
    """Classify the sanitization of a raw remote URL into (sanitized, warnings).

    A URL with raw credentials, query/fragment, or a local/unsupported shape is
    sanitized and the appropriate stable warnings are emitted. SCP-style URLs
    like ``git@github.com:org/repo.git`` are NOT credential-bearing — the
    ``git`` part is a protocol user, not a credential. Only ``@`` inside a URL
    that also has a ``://`` scheme (e.g. ``https://user:token@host``) is treated
    as credentials.
    """
    if not raw:
        return None, ()
    warnings: list[_RemoteWarningCode] = []
    # Detect credential-bearing URLs BEFORE sanitizing. Only an ``@`` that
    # appears inside a ``://``-scheme URL counts as credentials. The SCP-style
    # ``git@host:path`` form has no scheme; its ``git`` is a protocol user, not
    # a credential, and is preserved verbatim by sanitize_repository_url().
    if "://" in raw:
        after_scheme = raw.split("://", 1)[1]
        if "@" in after_scheme and not after_scheme.startswith("git@"):
            warnings.append("remote_credentials_removed")
    if "?" in raw or "#" in raw:
        warnings.append("remote_query_fragment_removed")
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


def _tracked_gitlink_paths(repo: Path) -> set[str]:
    """Return the set of tracked submodule gitlink paths from ``git ls-files --stage``.

    A submodule is recorded in the index as a "gitlink" tree entry with mode
    ``160000``. Git porcelain status for a modified submodule does NOT add a
    trailing slash — the porcelain path is exactly the gitlink path. Intersecting
    the tracked gitlink set with the non-clean porcelain entries is therefore
    the correct way to detect "dirty submodule content that this snapshot does
    not capture" (the porcelain entry may show a clean XY like `` M`` for a
    worktree-only change, or ``??`` if the submodule is brand new but its
    gitlink is already tracked under a different state).

    The ``git ls-files --stage`` output format is::

        <mode> <sha1> <stage>\t<path>

    Each line is NUL-delimited when ``-z`` is passed.

    Raises :class:`SourceUnavailableError` if the ``git ls-files --stage`` call
    fails — discovery must FAIL CLOSED. The caller (``_build_dirty_evidence``)
    marks the resulting evidence ``partial`` with a ``gitlink_discovery_failed``
    limitation rather than silently continuing without submodule detection.
    """
    raw = _run_git(repo, ["ls-files", "--stage", "-z"])
    gitlinks: set[str] = set()
    for part in raw.split(b"\x00"):
        if not part.strip():
            continue
        # Format: "<mode> <sha> <stage>\t<path>"
        # The mode is the first whitespace-delimited token.
        try:
            decoded = part.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            continue
        # Split off the path (after the tab).
        if "\t" not in decoded:
            continue
        meta, path = decoded.split("\t", 1)
        meta_parts = meta.split()
        if not meta_parts:
            continue
        mode = meta_parts[0]
        if mode == "160000":
            gitlinks.add(path.strip())
    return gitlinks


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

    # Tracked gitlink (mode 160000) paths from ``git ls-files --stage``. Git
    # porcelain does NOT add trailing slashes to submodule entries, so trailing-
    # slash heuristics miss real dirty submodules. The correct detection is the
    # intersection of non-clean porcelain paths with tracked gitlink paths.
    # Discovery FAILS CLOSED: if ``git ls-files --stage`` fails, we cannot tell
    # which paths are submodules, so any dirty content might hide an uninspected
    # submodule. Surface that as a typed limitation + ``partial`` completeness
    # rather than silently continuing with an empty gitlink set.
    gitlink_discovery_failed = False
    try:
        gitlink_paths = _tracked_gitlink_paths(repo)
    except SourceUnavailableError:
        gitlink_paths = set()
        gitlink_discovery_failed = True

    staged_digest = _staged_diff_digest(repo)
    unstaged_digest = _unstaged_diff_digest(repo)

    untracked: list[UntrackedEntry] = []
    staged_count = 0
    unstaged_count = 0
    renamed_count = 0
    deleted_count = 0
    unreadable_count = 0
    # A tracked gitlink with a non-clean porcelain entry indicates that the
    # submodule's working tree content differs from what this snapshot
    # captures — its dirty content is NOT content-snapshotted here.
    dirty_gitlink_paths: set[str] = set()
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
        # Gitlink-intersection detection: any non-clean entry whose path is a
        # tracked submodule gitlink → its dirty content is not snapshotted.
        # Renames/copies carry the new path in `path` and the original path in
        # `_orig_path`; check both against the gitlink set.
        candidate_paths = {path} | ({_orig_path} if _orig_path else set())
        is_gitlink_change = any(p in gitlink_paths for p in candidate_paths if p)
        if is_gitlink_change and (x not in (" ", "?") or y not in (" ", "?")):
            # Count each dirty gitlink path once (use the gitlink path that
            # matched — for a rename both old/new are tracked gitlinks).
            for p in candidate_paths:
                if p and p in gitlink_paths:
                    dirty_gitlink_paths.add(p)

    untracked.sort(key=lambda e: e.path)

    # Submodule status — preserve full SHA + state prefix, record failures.
    # Do NOT strip the raw line before reading the prefix: the prefix is the
    # FIRST character of the raw line (space, -, +, ~) and stripping removes
    # leading spaces, misreading an initialized (space-prefixed) submodule.
    submodule_entries: list[SubmoduleEntry] = []
    submodule_failed = False
    try:
        sub_raw = _run_git(repo, ["submodule", "status"])
        for line in sub_raw.decode("utf-8", errors="replace").splitlines():
            # Skip only truly empty raw lines. Read the prefix from the raw
            # (unstripped) line so leading spaces are preserved.
            if not line:
                continue
            try:
                prefix = line[0]
                # Valid prefixes are space, -, +, ~. Anything else is a
                # malformed line — skip it rather than producing a bogus entry.
                if prefix not in " -+~":
                    continue
                # The 40-char SHA follows the single prefix char.
                sha_full = line[1:41].strip()
                # The submodule name/path follows after column 42.
                rest = line[42:].strip()
                name = rest.split()[0] if rest else "unknown"
                state_map: dict[
                    str, Literal["initialized", "uninitialized", "changed", "conflicted"]
                ] = {
                    " ": "initialized",
                    "-": "uninitialized",
                    "+": "changed",
                    "~": "conflicted",
                }
                submodule_entries.append(
                    SubmoduleEntry(
                        name=name,
                        commit=sha_full if sha_full else "unknown",
                        state=state_map[prefix],
                    )
                )
            except (IndexError, ValueError):
                # Malformed single line — skip it, do not abort the whole parse.
                continue
    except SourceUnavailableError:
        submodule_failed = True

    # Sort submodule entries by name so the SourceEvidence model_validator
    # (which requires sorted + unique names) accepts capture output.
    submodule_entries.sort(key=lambda s: s.name)

    # Derive honest completeness + limitations. ``warnings`` is typed as the
    # closed Literal vocabulary so the SourceEvidence field accepts it directly.
    limitations: list[str] = []
    warnings: list[_EvidenceWarningCode] = []
    if unreadable_count > 0:
        limitations.append(f"unreadable_untracked_files:{unreadable_count}")
        warnings.append("unreadable_untracked_content")
    if submodule_failed:
        limitations.append("submodule_inspection_failed")
        warnings.append("submodule_inspection_failed")
    # Gitlink discovery failure is a hard partial-completeness signal: without
    # the tracked gitlink set we cannot tell whether any dirty working-tree
    # content belongs to a submodule (and therefore is not captured here).
    if gitlink_discovery_failed:
        limitations.append("gitlink_discovery_failed")
        warnings.append("gitlink_discovery_failed")
    # Dirty/changed submodules are not content-snapshotted by `git submodule
    # status` — mark evidence partial. Count both the commit-state prefixes
    # (`git submodule status` reports changed/conflicted) AND the tracked-
    # gitlink porcelain entries (which surface dirty submodule content that the
    # commit-state prefix misses when the submodule is uninitialized or porcelain
    # flags it differently). Each distinct dirty submodule is counted once.
    dirty_submodule_names = {
        s.name for s in submodule_entries if s.state in ("changed", "conflicted")
    }
    # Union: dirty gitlink paths + dirty submodule names (some overlap is
    # possible; we want a single count per distinct dirty submodule). Use the
    # larger of the two sets so a gitlink-detected change is never masked.
    dirty_count = max(len(dirty_submodule_names), len(dirty_gitlink_paths))
    if dirty_count:
        limitations.append(f"dirty_submodules_not_snapshotted:{dirty_count}")
        warnings.append("dirty_submodule_content_not_captured")

    # Always note that ignored files and external symlink targets are not included.
    limitations.append("ignored_files_not_included")
    limitations.append("external_symlink_targets_not_followed")

    completeness: Literal["complete", "partial"] = (
        "complete"
        if not unreadable_count
        and not submodule_failed
        and not gitlink_discovery_failed
        and not dirty_count
        else "partial"
    )

    # The SourceEvidence model_validator requires sorted + unique limitations
    # and warnings. Sort here so capture output is canonical.
    limitations = sorted(set(limitations))
    warnings = sorted(set(warnings))

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
    kind: Literal["file", "symlink", "unknown", "other"]
    mode: str | None = Field(default=None)
    digest: str | None = Field(default=None)  # None when unreadable

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        """Enforce a canonical repository-relative POSIX path (shared validator).

        See :func:`_normalize_canonical_rel_path`: backslashes are normalized to
        forward slashes, absolute forms (POSIX ``/`` or Windows drive-letter) are
        rejected, ``..`` AND ``.`` components are rejected, and empty components
        (``a//b`` or a trailing slash) are rejected.
        """
        return _normalize_canonical_rel_path(v, "untracked path")

    @field_validator("digest")
    @classmethod
    def _validate_digest(cls, v: str | None) -> str | None:
        return _validate_optional_digest(v)

    @model_validator(mode="after")
    def _check_kind_consistency(self) -> UntrackedEntry:
        """Cross-field consistency between ``kind`` and ``mode``/``digest``.

        - ``kind="file"`` must carry a ``mode`` (a regular file always has a
          mode; a missing mode signals an incomplete capture).
        - ``kind="symlink"`` should carry a ``digest`` (the symlink-target path
          digest). ``None`` here is allowed (unreadable target) but only when
          ``mode`` is present.
        - ``kind="unknown"`` must have BOTH ``mode`` and ``digest`` set to None
          — an "unknown" kind with concrete mode/digest is contradictory.
        """
        if self.kind == "file" and self.mode is None:
            raise ValueError(f"UntrackedEntry kind='file' requires a mode (path={self.path!r}).")
        if self.kind == "symlink" and self.mode is None:
            raise ValueError(f"UntrackedEntry kind='symlink' requires a mode (path={self.path!r}).")
        if self.kind == "unknown" and (self.mode is not None or self.digest is not None):
            raise ValueError(
                f"UntrackedEntry kind='unknown' requires mode=None and digest=None "
                f"(path={self.path!r})."
            )
        return self


class SubmoduleEntry(BaseModel):
    """One submodule's status with full commit SHA and state prefix preserved."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    name: str = Field(..., min_length=1)
    commit: str = Field(..., min_length=1)  # full SHA-1, or "unknown"
    state: Literal["initialized", "uninitialized", "changed", "conflicted"]

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        """Enforce a canonical repository-relative POSIX path (same rules as
        :class:`UntrackedEntry.path`). The submodule name is its tracked
        gitlink path."""
        return _normalize_canonical_rel_path(v, "submodule name")

    @field_validator("commit")
    @classmethod
    def _validate_commit(cls, v: str) -> str:
        """Exactly 40 lowercase hex (SHA-1) or the literal ``"unknown"``.

        Git submodule commits are full SHA-1 object names. An uninitialized
        submodule may report no commit; capture records that as ``"unknown"``.
        Any other value (truncated, uppercase, non-hex) is rejected so a
        tampered sidecar cannot smuggle in a malformed commit.
        """
        if v == "unknown":
            return v
        if not _SHA1_HEX_PATTERN.fullmatch(v):
            raise ValueError(
                f"submodule commit must be 40 lowercase hex chars (sha-1) or 'unknown'; got {v!r}."
            )
        return v


class SourceCounts(BaseModel):
    """Typed dirty-evidence counts."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    staged: int = Field(default=0, ge=0)
    unstaged: int = Field(default=0, ge=0)
    untracked: int = Field(default=0, ge=0)
    renamed_or_copied: int = Field(default=0, ge=0)
    deleted: int = Field(default=0, ge=0)
    submodules: int = Field(default=0, ge=0)


# Stable, closed warning-code vocabulary for :class:`SourceEvidence`. A warning
# MUST be one of these codes — free text is rejected so callers can switch on a
# stable contract and a tampered sidecar cannot smuggle in fabricated warnings.
_EvidenceWarningCode = Literal[
    "unreadable_untracked_content",
    "submodule_inspection_failed",
    "dirty_submodule_content_not_captured",
    "gitlink_discovery_failed",
]
_EVIDENCE_WARNING_CODES: frozenset[str] = frozenset(
    {
        "unreadable_untracked_content",
        "submodule_inspection_failed",
        "dirty_submodule_content_not_captured",
        "gitlink_discovery_failed",
    }
)


def _validate_evidence_warning(v: str) -> str:
    """Reject any warning code outside the closed vocabulary."""
    if v not in _EVIDENCE_WARNING_CODES:
        raise ValueError(
            f"evidence warning {v!r} is not a recognized code; "
            f"allowed: {sorted(_EVIDENCE_WARNING_CODES)}."
        )
    return v


# Stable, closed limitation-code vocabulary for :class:`SourceEvidence`. Two
# codes carry a dynamic count suffix (``unreadable_untracked_files:N`` and
# ``dirty_submodules_not_snapshotted:N``); the rest are exact literals. A
# limitation MUST match an allowed code or a recognized ``<prefix>:<count>``
# pattern — free text is rejected so the limitation contract is stable.
_EVIDENCE_LIMITATION_EXACT_CODES: frozenset[str] = frozenset(
    {
        "submodule_inspection_failed",
        "ignored_files_not_included",
        "external_symlink_targets_not_followed",
        "gitlink_discovery_failed",
    }
)
_EVIDENCE_LIMITATION_PREFIXES: frozenset[str] = frozenset(
    {
        "unreadable_untracked_files:",
        "dirty_submodules_not_snapshotted:",
    }
)


def _validate_evidence_limitation(v: str) -> str:
    """Reject any limitation code outside the closed vocabulary or its patterns.

    Exact codes must match a known literal. Count-bearing codes must look like
    ``<known_prefix>:<non-negative-integer>``.
    """
    if v in _EVIDENCE_LIMITATION_EXACT_CODES:
        return v
    for prefix in _EVIDENCE_LIMITATION_PREFIXES:
        if v.startswith(prefix):
            count_str = v[len(prefix) :]
            if count_str.isdigit() and int(count_str) >= 0:
                return v
            raise ValueError(
                f"evidence limitation {v!r} has a malformed count suffix "
                f"(expected non-negative integer after {prefix!r})."
            )
    raise ValueError(
        f"evidence limitation {v!r} is not a recognized code or pattern; "
        f"allowed exact codes: {sorted(_EVIDENCE_LIMITATION_EXACT_CODES)}; "
        f"allowed prefixes: {sorted(_EVIDENCE_LIMITATION_PREFIXES)}."
    )


class SourceEvidence(BaseModel):
    """Structured dirty-source evidence (staged/unstaged/untracked/submodule).

    All containers are tuples of frozen models — deeply immutable.
    Completeness is honestly derived: ``partial`` when any file was unreadable
    or submodule inspection failed, with typed limitations/warnings. The
    ``warnings`` and ``limitations`` fields use a closed code vocabulary — free
    text is rejected so the contract is stable and tamper-evident.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    staged_digest: str = Field(..., min_length=1)
    unstaged_digest: str = Field(..., min_length=1)
    untracked: tuple[UntrackedEntry, ...] = Field(default_factory=tuple)
    submodule_status: tuple[SubmoduleEntry, ...] = Field(default_factory=tuple)
    counts: SourceCounts = Field(default_factory=SourceCounts)
    completeness: Literal["complete", "partial", "error"] = Field(default="complete")
    limitations: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[_EvidenceWarningCode, ...] = Field(default_factory=tuple)

    @field_validator("staged_digest", "unstaged_digest")
    @classmethod
    def _validate_digest(cls, v: str) -> str:
        if not _DIGEST_HEX_PATTERN.fullmatch(v):
            raise ValueError(f"digest must be 64 lowercase hex chars (sha256); got {v!r}.")
        return v

    @field_validator("warnings")
    @classmethod
    def _validate_warnings_codes(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_validate_evidence_warning(w) for w in v)

    @field_validator("limitations")
    @classmethod
    def _validate_limitations_codes(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_validate_evidence_limitation(lim) for lim in v)

    # The standard limitations that are always present (ignored files + external
    # symlink targets) and do NOT affect completeness even when
    # ``completeness == "complete"``.
    _STANDARD_LIMITATIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "ignored_files_not_included",
            "external_symlink_targets_not_followed",
        }
    )

    @model_validator(mode="after")
    def _verify_evidence_invariants(self) -> SourceEvidence:
        """Enforce canonical model-level invariants on the dirty-evidence record.

        - ``untracked`` is sorted by ``path`` and has unique paths.
        - ``submodule_status`` is sorted by ``name`` and has unique names.
        - ``counts.untracked == len(untracked)`` and
          ``counts.submodules == len(submodule_status)``.
        - when ``completeness == "complete"``: no entry has ``digest is None``,
          no warnings, and only the standard limitations may be present.
        - ``warnings`` are sorted, unique, and use stable codes only.
        - ``limitations`` are sorted and use stable codes/patterns only.
        """
        # untracked: sorted by path + unique paths.
        untracked_paths = [e.path for e in self.untracked]
        if untracked_paths != sorted(untracked_paths):
            raise ValueError(f"untracked entries must be sorted by path; got {untracked_paths!r}.")
        if len(set(untracked_paths)) != len(untracked_paths):
            dupes = sorted({p for p in untracked_paths if untracked_paths.count(p) > 1})
            raise ValueError(f"untracked entries have duplicate paths: {dupes!r}.")
        # submodule_status: sorted by name + unique names.
        sub_names = [s.name for s in self.submodule_status]
        if sub_names != sorted(sub_names):
            raise ValueError(f"submodule_status entries must be sorted by name; got {sub_names!r}.")
        if len(set(sub_names)) != len(sub_names):
            dupes = sorted({n for n in sub_names if sub_names.count(n) > 1})
            raise ValueError(f"submodule_status entries have duplicate names: {dupes!r}.")
        # Count consistency.
        if self.counts.untracked != len(self.untracked):
            raise ValueError(
                f"counts.untracked ({self.counts.untracked}) must equal "
                f"len(untracked) ({len(self.untracked)})."
            )
        if self.counts.submodules != len(self.submodule_status):
            raise ValueError(
                f"counts.submodules ({self.counts.submodules}) must equal "
                f"len(submodule_status) ({len(self.submodule_status)})."
            )
        # Warnings: sorted + stable codes.
        if list(self.warnings) != sorted(self.warnings):
            raise ValueError(f"warnings must be sorted; got {list(self.warnings)!r}.")
        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError(f"warnings must be unique; got {list(self.warnings)!r}.")
        # Limitations: sorted.
        if list(self.limitations) != sorted(self.limitations):
            raise ValueError(f"limitations must be sorted; got {list(self.limitations)!r}.")
        # Completeness="complete" must mean no unreadable content and no
        # non-standard limitations and no warnings.
        if self.completeness == "complete":
            for entry in self.untracked:
                if entry.digest is None:
                    raise ValueError(
                        f"completeness='complete' forbids an untracked entry with "
                        f"digest=None (path={entry.path!r})."
                    )
            if self.warnings:
                raise ValueError(
                    f"completeness='complete' forbids warnings; got {self.warnings!r}."
                )
            # Only the standard always-present limitations are allowed when
            # complete. Anything else signals hidden incompleteness.
            non_standard = [
                lim for lim in self.limitations if lim not in self._STANDARD_LIMITATIONS
            ]
            if non_standard:
                raise ValueError(
                    f"completeness='complete' forbids non-standard limitations "
                    f"{sorted(set(non_standard))!r}."
                )
        return self


class RemoteWarning(BaseModel):
    """Typed warning about remote URL sanitization (code-only; no free text)."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    code: Literal[
        "remote_credentials_removed",
        "remote_query_fragment_removed",
        "remote_local_or_unsupported_removed",
    ]


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

    @field_validator("tree_digest", "input_digest")
    @classmethod
    def _validate_digest(cls, v: str) -> str:
        if not _DIGEST_HEX_PATTERN.fullmatch(v):
            raise ValueError(f"digest must be 64 lowercase hex chars (sha256); got {v!r}.")
        return v

    @model_validator(mode="after")
    def _verify_remote_url_warnings(self) -> SourceSnapshot:
        """Correlate ``remote_url`` with ``remote_warnings`` at the model level.

        The correlation is TWO-WAY. Computes the canonical sanitization of the
        stored ``remote_url`` using the same classifier used at capture time:

        - Required warnings must be present: whenever the sanitized value
          differs from what was supplied, the appropriate warning(s) MUST be in
          ``remote_warnings`` — otherwise the lossy sanitization would be silent.
        - Impossible warnings must be ABSENT: when ``remote_url`` is a clean
          HTTPS URL with no credentials/query/fragment (or any URL the classifier
          leaves untouched), the supplied warnings MUST exactly equal the
          expected set. A fabricated warning (e.g. ``remote_credentials_removed``
          on a URL that never carried credentials) is rejected as contradictory.

        A ``None`` ``remote_url`` (no origin) MUST carry no warnings: warnings
        without a URL are contradictory and a likely sign of a tampered sidecar.

        The stored URL is replaced with the sanitized form (bypassing the frozen
        check) so callers always see the canonical value. SCP-style
        ``git@host:path`` URLs have no ``://`` scheme; their ``git`` is a
        protocol user, not a credential — they are NOT flagged.
        """
        supplied_codes = {w.code for w in self.remote_warnings}
        if self.remote_url is None:
            # No origin → no sanitization ever happened → no warnings are valid.
            if supplied_codes:
                raise ValueError(
                    f"remote_url is None but remote_warnings are present "
                    f"{sorted(supplied_codes)!r}; a missing remote URL cannot "
                    "carry sanitization warnings."
                )
            return self
        expected_sanitized, expected_codes = _classify_remote_url_sanitization(self.remote_url)
        expected_set = set(expected_codes)
        # Each expected warning must be present (a missing warning means the
        # sanitization lost information silently).
        missing = [c for c in expected_codes if c not in supplied_codes]
        if missing:
            raise ValueError(
                f"remote_url sanitization required warning codes {sorted(missing)!r} "
                "but they are absent from remote_warnings; remote URL sanitization "
                "must not lose information silently."
            )
        # No extra/impossible warnings: the supplied set must equal the expected
        # set. A warning that the classifier did not require is fabricated
        # (e.g. ``remote_credentials_removed`` on a clean URL with no userinfo).
        extra = sorted(supplied_codes - expected_set)
        if extra:
            raise ValueError(
                f"remote_url {self.remote_url!r} does not justify remote_warnings "
                f"{extra!r}; the classifier expected only {sorted(expected_set)!r}. "
                "Fabricated or impossible remote warnings are not permitted."
            )
        # Replace the stored value with its canonical sanitized form. The model
        # is frozen; use object.__setattr__ to bypass the frozen check (this is
        # canonicalization, not mutation of caller-supplied semantics).
        if expected_sanitized != self.remote_url:
            object.__setattr__(self, "remote_url", expected_sanitized)
        return self

    @model_validator(mode="after")
    def _verify_source_invariants(self) -> SourceSnapshot:
        """Enforce clean/dirty evidence invariants.

        - ``is_clean == (evidence is None)``: a clean tree has no evidence and
          vice-versa. A tampered sidecar cannot claim ``is_clean`` with
          evidence present, nor ``not is_clean`` with no evidence.
        - ``is_clean`` implies ``is_canonical``: a clean tree is canonical.
        - ``not is_clean`` implies ``not is_canonical``: a dirty tree is
          non-canonical.
        - when ``evidence`` is present its ``completeness`` must be ``complete``
          or ``partial`` (never ``error``); error-status evidence would have
          aborted capture rather than recording a snapshot.
        """
        if self.is_clean != (self.evidence is None):
            raise ValueError(
                "is_clean must equal (evidence is None): a clean tree carries no "
                "evidence and a dirty tree always carries evidence."
            )
        if self.is_clean and not self.is_canonical:
            raise ValueError("is_clean implies is_canonical (a clean tree is canonical).")
        if not self.is_clean and self.is_canonical:
            raise ValueError(
                "not is_clean implies not is_canonical (a dirty tree is non-canonical)."
            )
        if self.evidence is not None and self.evidence.completeness == "error":
            raise ValueError(
                "source evidence completeness must be 'complete' or 'partial', "
                "never 'error' (an error-state capture aborts instead of recording)."
            )
        return self

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
    # Capture the RAW remote URL and pre-compute the expected sanitization
    # warnings. The SourceSnapshot model_validator recomputes the sanitization
    # from the (still raw) stored URL and verifies the warnings match — so the
    # raw URL must be supplied to the constructor, not the sanitized form.
    raw_url = _raw_remote_url(repo_path)
    # Pre-compute the expected sanitization warning codes. The model_validator
    # recomputes the sanitization from the raw stored URL and verifies these
    # warnings are present, so capture must supply them.
    _, remote_warn_codes = _classify_remote_url_sanitization(raw_url)
    evidence = _build_dirty_evidence(repo_path) if dirty else None
    input_digest = _envelope_digest(tree_digest, evidence)
    return SourceSnapshot(
        commit_sha=commit_sha,
        branch=branch,
        remote_url=raw_url or None,  # raw; model_validator sanitizes + verifies warnings
        remote_warnings=tuple(RemoteWarning(code=w) for w in remote_warn_codes),
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
