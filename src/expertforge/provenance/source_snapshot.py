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
# Canonical octal mode string for an untracked working-tree entry, e.g.
# ``0o644`` (3 octal digits) or ``0o100644`` (4 octal digits covering the
# setuid/setgid/sticky bits + rwxrwxrwx). The leading ``0o`` prefix matches
# Python's ``oct()`` output, which is what the capture path emits. The digit
# count is 1-4: ``oct()`` of a low mode value can produce as few as one octal
# digit (e.g. ``0o0`` for mode 0, ``0o44`` for mode 36) — these are valid
# canonical ``oct()`` outputs and must not be rejected.
_UNTRACKED_MODE_PATTERN = re.compile(r"^0o[0-7]{1,4}$")


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

    Returns the empty string if no origin is configured. The caller
    (:func:`capture_source_snapshot`) is responsible for sanitizing via
    :func:`_classify_remote_url_sanitization` BEFORE constructing the
    :class:`SourceSnapshot`. The raw value is intentionally exposed only to
    the capture path; the model never sees the raw URL.
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
    that also has a ``://`` scheme (e.g. ``https://user:token@host`` or
    ``https://git@host/repo.git``) is treated as credentials.
    """
    if not raw:
        return None, ()
    warnings: list[_RemoteWarningCode] = []
    # Detect credential-bearing URLs BEFORE sanitizing. Only an ``@`` that
    # appears inside a ``://``-scheme URL counts as credentials. The SCP-style
    # ``git@host:path`` form has no scheme; its ``git`` is a protocol user, not
    # a credential, and is preserved verbatim by sanitize_repository_url().
    # The SCP-style exemption applies ONLY when there is NO ``://`` scheme:
    # ``https://git@host/repo.git`` has a scheme + userinfo, so its ``@`` IS
    # credential-bearing.
    if "://" in raw:
        after_scheme = raw.split("://", 1)[1]
        if "@" in after_scheme:
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
    return set(_tracked_gitlink_records(repo).keys())


def _tracked_gitlink_records(repo: Path) -> dict[str, str]:
    """Return ``{gitlink_path: sha1}`` for tracked submodule gitlinks (mode 160000).

    Used by :func:`_build_dirty_evidence` to synthesize a :class:`SubmoduleEntry`
    for a dirty gitlink that ``git submodule status`` did not report as
    changed/conflicted (e.g. an initialized submodule whose working tree has
    untracked content but whose checked-out commit matches the recorded one).
    Recording such entries makes the ``dirty_submodules_not_snapshotted:N`` count
    fully verifiable from the model's ``submodule_status`` (N equals the number
    of entries with ``state`` in ``changed``/``conflicted``).

    Raises :class:`SourceUnavailableError` if the ``git ls-files --stage`` call
    fails — discovery must FAIL CLOSED.
    """
    raw = _run_git(repo, ["ls-files", "--stage", "-z"])
    gitlinks: dict[str, str] = {}
    for part in raw.split(b"\x00"):
        if not part.strip():
            continue
        # Format: "<mode> <sha> <stage>\t<path>"
        try:
            decoded = part.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            continue
        if "\t" not in decoded:
            continue
        meta, path = decoded.split("\t", 1)
        meta_parts = meta.split()
        if len(meta_parts) < 2:
            continue
        mode, sha = meta_parts[0], meta_parts[1]
        if mode == "160000":
            gitlinks[path.strip()] = sha
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
        gitlink_records = _tracked_gitlink_records(repo)
    except SourceUnavailableError:
        gitlink_records = {}
        gitlink_discovery_failed = True
    gitlink_paths = set(gitlink_records.keys())

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

    # A dirty gitlink (tracked submodule path with a non-clean porcelain entry)
    # indicates the submodule's working-tree content differs from what this
    # snapshot captures — its content is NOT snapshotted here. ``git submodule
    # status`` reports commit-state prefixes that MISS worktree-only submodule
    # changes (a submodule whose checked-out commit matches the recorded one but
    # whose working tree has new/modified content shows `` M`` in porcelain but
    # `` `` (initialized) in ``git submodule status``). So that the
    # ``dirty_submodules_not_snapshotted:N`` count is fully verifiable from the
    # model (N == number of entries with state changed/conflicted), represent
    # every dirty gitlink as a changed-state entry: back-fill a missing entry,
    # OR upgrade an existing non-dirty (initialized/uninitialized) entry to
    # ``changed``. ``git submodule status`` is still the primary source for the
    # commit SHA; the gitlink's recorded SHA back-fills entries it omits.
    entries_by_name: dict[str, SubmoduleEntry] = {s.name: s for s in submodule_entries}
    for dirty_path in dirty_gitlink_paths:
        existing = entries_by_name.get(dirty_path)
        if existing is None:
            recorded_sha = gitlink_records.get(dirty_path)
            if recorded_sha and _SHA1_HEX_PATTERN.fullmatch(recorded_sha):
                entries_by_name[dirty_path] = SubmoduleEntry(
                    name=dirty_path, commit=recorded_sha, state="changed"
                )
        elif existing.state not in ("changed", "conflicted"):
            # Upgrade a non-dirty entry (initialized/uninitialized) to changed:
            # porcelain confirmed its working-tree content differs.
            entries_by_name[dirty_path] = existing.model_copy(update={"state": "changed"})
    submodule_entries = list(entries_by_name.values())

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
    # Dirty/changed submodules are not content-snapshotted by this snapshot —
    # mark evidence partial. The count is the number of submodule_status entries
    # with state changed/conflicted (dirty gitlinks missing a commit-state
    # prefix are synthesized above so the count is honest AND verifiable from
    # the model). Each distinct dirty submodule is counted once.
    dirty_submodule_names = {
        s.name for s in submodule_entries if s.state in ("changed", "conflicted")
    }
    dirty_count = len(dirty_submodule_names)
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

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str | None) -> str | None:
        """Validate ``mode`` as a canonical octal mode string when present.

        Accepts ``None`` (the only valid value for ``kind="unknown"`` /
        ``kind="other"``). A present value MUST match ``^0o[0-7]{1,4}$`` so a
        tampered sidecar cannot smuggle in a free-text or non-octal mode. The
        capture path emits ``oct(stat.S_IMODE(...))`` which already produces
        this canonical form (1-4 octal digits, e.g. ``0o644`` or ``0o0`` for a
        zero mode).
        """
        if v is None:
            return None
        if not _UNTRACKED_MODE_PATTERN.fullmatch(v):
            raise ValueError(
                f"UntrackedEntry mode must be a canonical octal mode string "
                f"matching {{:0o[0-7]{{1,4}}}} (e.g. '0o644'); got {v!r}."
            )
        return v

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
        - ``kind="other"`` (a special/non-regular/non-symlink entry, e.g. a
          FIFO/socket/block/char device) must have ``digest=None`` (no content
          digest is meaningful for a non-file/non-symlink kind).
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
        if self.kind == "other" and self.digest is not None:
            raise ValueError(
                f"UntrackedEntry kind='other' requires digest=None "
                f"(no digest for non-file/non-symlink kinds; path={self.path!r})."
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

    @model_validator(mode="after")
    def _check_commit_state_consistency(self) -> SubmoduleEntry:
        """``commit="unknown"`` is only valid with ``state="uninitialized"``.

        For any other state (``initialized`` / ``changed`` / ``conflicted``),
        ``commit`` must be a valid 40-hex SHA-1 (a real git object name), not
        ``"unknown"``. ``git submodule status`` always emits a full SHA for an
        initialized/changed/conflicted submodule; ``"unknown"`` is only ever
        the capture-time encoding for an uninitialized submodule (no checkout).
        """
        if self.commit == "unknown" and self.state != "uninitialized":
            raise ValueError(
                f"SubmoduleEntry commit='unknown' is only valid with "
                f"state='uninitialized'; got state={self.state!r} "
                f"(name={self.name!r})."
            )
        return self


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
    ``<known_prefix>:<positive-integer>`` — a zero count (e.g.
    ``unreadable_untracked_files:0``) is nonsensical (it would describe zero
    unreadable files, which is not a limitation at all) and is rejected.
    """
    if v in _EVIDENCE_LIMITATION_EXACT_CODES:
        return v
    for prefix in _EVIDENCE_LIMITATION_PREFIXES:
        if v.startswith(prefix):
            count_str = v[len(prefix) :]
            if count_str.isdigit() and int(count_str) >= 1:
                return v
            raise ValueError(
                f"evidence limitation {v!r} has a malformed count suffix "
                f"(expected a positive integer >= 1 after {prefix!r})."
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
        - count-bearing limitations reject a zero count suffix (a zero count is
          not a real limitation — e.g. ``unreadable_untracked_files:0`` is
          nonsensical).
        - ``completeness="partial"`` requires at least one warning OR one
          non-standard limitation (beyond the two always-present standard
          limitations). A partial record with neither signals an explanation
          is missing.
        - the ``dirty_submodule_content_not_captured`` warning and the
          ``dirty_submodules_not_snapshotted:N`` limitation are correlated: one
          cannot appear without the other (and vice-versa).
        - count-bearing limitations are validated against the verifiable facts:
          ``unreadable_untracked_files:N`` must equal the count of untracked
          entries with ``digest is None``; ``dirty_submodules_not_snapshotted:N``
          must equal the count of dirty (changed/conflicted) submodule entries.
        - bidirectional warning↔limitation pairing: the
          ``unreadable_untracked_content`` /
          ``submodule_inspection_failed`` / ``gitlink_discovery_failed`` warnings
          each require their matching limitation (and vice-versa).
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
        # Limitations: sorted + unique. Duplicate limitations would make the
        # behavioral envelope non-canonical even if their set of meanings were
        # unchanged.
        if list(self.limitations) != sorted(self.limitations):
            raise ValueError(f"limitations must be sorted; got {list(self.limitations)!r}.")
        if len(set(self.limitations)) != len(self.limitations):
            raise ValueError(f"limitations must be unique; got {list(self.limitations)!r}.")
        # Count-bearing limitations reject a zero count (a zero count is not a
        # real limitation). The field_validator already enforced the prefix + a
        # positive integer; this is a defensive re-check that also produces a
        # clear cross-field error.
        for lim in self.limitations:
            for prefix in _EVIDENCE_LIMITATION_PREFIXES:
                if lim.startswith(prefix):
                    count_str = lim[len(prefix) :]
                    if count_str.isdigit() and int(count_str) < 1:
                        raise ValueError(
                            f"evidence limitation {lim!r} carries a zero count; a "
                            "count-bearing limitation must describe at least one item."
                        )
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
        # Completeness="partial" must be explained by at least one warning OR
        # one non-standard limitation. The two always-present standard
        # limitations (ignored files + external symlink targets) do NOT explain
        # partial-ness by themselves — they are present on every record.
        if self.completeness == "partial":
            has_explanation = bool(self.warnings) or any(
                lim not in self._STANDARD_LIMITATIONS for lim in self.limitations
            )
            if not has_explanation:
                raise ValueError(
                    "completeness='partial' requires at least one warning OR one "
                    "non-standard limitation (beyond the standard "
                    "ignored_files_not_included / "
                    "external_symlink_targets_not_followed) to explain the "
                    "partial-ness."
                )
        # Dirty-submodule warning ↔ limitation correlation. The warning
        # ``dirty_submodule_content_not_captured`` and the limitation
        # ``dirty_submodules_not_snapshotted:N`` (N >= 1) describe the SAME
        # underlying condition from two angles; one cannot appear without the
        # other.
        has_dirty_submodule_warning = "dirty_submodule_content_not_captured" in self.warnings
        has_dirty_submodule_limitation = any(
            lim.startswith("dirty_submodules_not_snapshotted:") for lim in self.limitations
        )
        if has_dirty_submodule_warning != has_dirty_submodule_limitation:
            raise ValueError(
                "dirty_submodule_content_not_captured warning and the "
                "dirty_submodules_not_snapshotted:N limitation must appear "
                "together (one cannot be present without the other); got "
                f"warnings={list(self.warnings)!r}, limitations={list(self.limitations)!r}."
            )
        # Count-bearing limitations must agree with the verifiable facts.
        # ``unreadable_untracked_files:N`` — N must equal the number of untracked
        # entries whose ``digest is None``. A tampered sidecar that inflates or
        # deflates the count (relative to the actual unreadable entries) is
        # rejected.
        unreadable_actual = sum(1 for e in self.untracked if e.digest is None)
        for lim in self.limitations:
            if lim.startswith("unreadable_untracked_files:"):
                stated = int(lim[len("unreadable_untracked_files:") :])
                if stated != unreadable_actual:
                    raise ValueError(
                        f"evidence limitation {lim!r} count does not match the number of "
                        f"untracked entries with digest=None ({unreadable_actual}); the count "
                        "must equal the actual unreadable-entry count."
                    )
        # ``dirty_submodules_not_snapshotted:N`` — N must equal the number of
        # ``submodule_status`` entries with ``state`` in changed/conflicted (the
        # dirty submodule entries the model can see). The capture path
        # synthesizes a changed-state entry for any dirty gitlink that
        # ``git submodule status`` did not report, so this count is honest and
        # verifiable.
        dirty_actual = sum(1 for s in self.submodule_status if s.state in ("changed", "conflicted"))
        for lim in self.limitations:
            if lim.startswith("dirty_submodules_not_snapshotted:"):
                stated = int(lim[len("dirty_submodules_not_snapshotted:") :])
                if stated != dirty_actual:
                    raise ValueError(
                        f"evidence limitation {lim!r} count does not match the number of "
                        f"dirty submodule entries (state changed/conflicted) in "
                        f"submodule_status ({dirty_actual}); the count must equal the "
                        "actual dirty-submodule-entry count."
                    )
        # Bidirectional warning ↔ limitation pairing. Each of these warnings
        # describes a condition whose durable count/limitation MUST also be
        # present (and vice-versa) so a tampered sidecar cannot fabricate a
        # warning without the matching limitation, or drop the limitation while
        # keeping the warning.
        warning_limitation_pairs: tuple[tuple[_EvidenceWarningCode, str | None, str], ...] = (
            # warning, limitation-prefix-or-None, human label
            (
                "unreadable_untracked_content",
                "unreadable_untracked_files:",
                "unreadable_untracked_files:N",
            ),
            ("submodule_inspection_failed", None, "submodule_inspection_failed"),
            ("gitlink_discovery_failed", None, "gitlink_discovery_failed"),
        )
        for warn_code, lim_prefix, lim_label in warning_limitation_pairs:
            has_warn = warn_code in self.warnings
            if lim_prefix is not None:
                has_lim = any(lim.startswith(lim_prefix) for lim in self.limitations)
            else:
                has_lim = lim_label in self.limitations
            if has_warn != has_lim:
                raise ValueError(
                    f"evidence warning {warn_code!r} and its limitation {lim_label!r} "
                    "must appear together (one cannot be present without the other); "
                    f"got warnings={list(self.warnings)!r}, "
                    f"limitations={list(self.limitations)!r}."
                )
        # Represented incomplete facts MUST carry their exact durable
        # explanation. Existing warning↔limitation pairing and count checks are
        # insufficient when an unrelated warning is used to mask a missing pair.
        unreadable_limits = [
            lim for lim in self.limitations if lim.startswith("unreadable_untracked_files:")
        ]
        has_unreadable_warning = "unreadable_untracked_content" in self.warnings
        if unreadable_actual > 0:
            expected = f"unreadable_untracked_files:{unreadable_actual}"
            if unreadable_limits != [expected] or not has_unreadable_warning:
                raise ValueError(
                    "unreadable untracked entries require both "
                    "unreadable_untracked_content and exactly "
                    f"{expected!r}; got warnings={list(self.warnings)!r}, "
                    f"limitations={list(self.limitations)!r}."
                )
        elif unreadable_limits or has_unreadable_warning:
            raise ValueError(
                "unreadable_untracked_content and unreadable_untracked_files:N "
                "are forbidden when no untracked entry has digest=None."
            )

        dirty_limits = [
            lim for lim in self.limitations if lim.startswith("dirty_submodules_not_snapshotted:")
        ]
        has_dirty_warning = "dirty_submodule_content_not_captured" in self.warnings
        if dirty_actual > 0:
            expected = f"dirty_submodules_not_snapshotted:{dirty_actual}"
            if dirty_limits != [expected] or not has_dirty_warning:
                raise ValueError(
                    "changed/conflicted submodule entries require both "
                    "dirty_submodule_content_not_captured and exactly "
                    f"{expected!r}; got warnings={list(self.warnings)!r}, "
                    f"limitations={list(self.limitations)!r}."
                )
        elif dirty_limits or has_dirty_warning:
            raise ValueError(
                "dirty_submodule_content_not_captured and "
                "dirty_submodules_not_snapshotted:N are forbidden when no "
                "submodule entry is changed/conflicted."
            )
        return self


class RemoteWarning(BaseModel):
    """Typed warning about remote URL sanitization (code-only; no free text).

    The codes are durable redaction OBSERVATIONS produced by the capture path
    (:func:`_classify_remote_url_sanitization`) at the moment the raw URL is
    sanitized. They are NOT re-derived from the (already-sanitized) stored URL
    on load — the codes are Literal-validated, which is sufficient to make a
    tampered sidecar unable to fabricate or smuggle in unknown warning text.
    The capture path is the sole place where URL↔warning correlation is
    performed; the :class:`SourceSnapshot` model accepts whatever codes are
    supplied because that correlation cannot survive serialization (the stored
    URL is already sanitized, so re-deriving required warnings on load would
    incorrectly reject the retained warnings as fabricated).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    code: Literal[
        "remote_credentials_removed",
        "remote_query_fragment_removed",
        "remote_local_or_unsupported_removed",
    ]


# Canonical accepted forms for the stored (already-sanitized) ``remote_url``:
#   - ``https://host/path...`` or ``http://host/path...`` — scheme URL with NO
#     userinfo (``@``), query (``?``), or fragment (``#``); the host/path run
#     contains only non-credential, non-delimiter characters.
#   - ``git@host:path...`` — SCP-style with NO scheme; the ``git`` part is a
#     protocol user (not a credential), and no query/fragment is allowed.
# A tampered sidecar that smuggles in credentials, a query/fragment, or a local
# path (``file://``, drive letter ``C:``, leading ``/`` or ``\``) is rejected
# so model parsing cannot be bypassed with a raw or local locator.
_REMOTE_URL_HTTP_PATTERN = re.compile(r"^https?://[^\s@?#]+$")
_REMOTE_URL_SCP_PATTERN = re.compile(r"^git@[^\s@?#]*:[^\s@?#]*$")


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
    # The sanitized remote URL (or ``None`` when no origin was configured). The
    # raw URL is sanitized in the capture path (:func:`_classify_remote_url_sanitization`)
    # BEFORE construction, so the model only ever stores the already-sanitized
    # form. ``remote_warnings`` carry the Literal redaction-observation codes
    # (e.g. ``remote_credentials_removed``); they are self-validating after
    # serialization because the codes are Literal-validated, and the capture
    # path produces them in lock-step with sanitization. The model therefore
    # does NOT correlate ``remote_url`` with ``remote_warnings``: that
    # correlation cannot survive a sidecar round-trip (the stored URL is
    # already sanitized, so the classifier would derive no required warnings
    # and reject the retained warnings as fabricated).
    remote_url: str | None = Field(default=None)
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

    @field_validator("remote_url")
    @classmethod
    def _validate_remote_url(cls, v: str | None) -> str | None:
        """Validate the stored ``remote_url`` is already canonical/sanitized.

        ``None`` is allowed (a local/unsupported remote that was redacted to
        ``None`` at capture time). A present value MUST be one of the accepted
        canonical forms so a tampered sidecar cannot smuggle in a raw or local
        locator:

        - ``https://host/path...`` / ``http://host/path...`` — scheme URL with
          NO userinfo (``@``), query (``?``), or fragment (``#``).
        - ``git@host:path...`` — SCP-style with NO scheme (the ``git`` part is
          a protocol user, not a credential).

        Rejected shapes include credential-bearing scheme URLs
        (``https://user:token@host``), any query/fragment, ``file://``, Windows
        drive letters (``C:``), and POSIX/local absolute paths (leading ``/``
        or ``\\``).
        """
        if v is None:
            return None
        # Reject the local/unsupported locator shapes outright (these sanitize
        # to None at capture time, so a present value of this shape is tampering).
        if v.startswith("file://"):
            raise ValueError(f"remote_url {v!r} must not be a file:// locator.")
        if re.match(r"^[A-Za-z]:[\\/]", v):
            raise ValueError(f"remote_url {v!r} must not be a Windows drive path.")
        if v.startswith("/") or v.startswith("\\"):
            raise ValueError(f"remote_url {v!r} must not be a local absolute path.")
        # Reject any credential/query/fragment leakage that sanitization should
        # have stripped. ``@`` is only credential-bearing inside a ``://``-scheme
        # URL; the SCP-style ``git@host:path`` form has no scheme and its ``@``
        # is a protocol user, so it is matched by the SCP pattern below instead.
        if "://" in v:
            if not _REMOTE_URL_HTTP_PATTERN.fullmatch(v):
                raise ValueError(
                    f"remote_url {v!r} is not a canonical sanitized HTTP(S) URL; "
                    "it must not contain '@' (credentials), '?' (query), or '#' (fragment)."
                )
            return v
        # No scheme: only the SCP-style ``git@host:path`` form is accepted.
        if not _REMOTE_URL_SCP_PATTERN.fullmatch(v):
            raise ValueError(
                f"remote_url {v!r} is not a canonical sanitized locator; accepted "
                "forms are 'https://host/path', 'http://host/path', 'git@host:path', or None."
            )
        return v

    @field_validator("remote_warnings")
    @classmethod
    def _validate_remote_warnings_canonical(
        cls, v: tuple[RemoteWarning, ...]
    ) -> tuple[RemoteWarning, ...]:
        """``remote_warnings`` must be sorted by code and unique.

        The codes are Literal-validated by :class:`RemoteWarning`; this validator
        enforces the canonical ordering so a tampered sidecar cannot smuggle in
        a duplicate or out-of-order warning.
        """
        codes = [w.code for w in v]
        if codes != sorted(codes):
            raise ValueError(f"remote_warnings must be sorted by code; got {codes!r}.")
        if len(set(codes)) != len(codes):
            raise ValueError(f"remote_warnings must be unique; got {codes!r}.")
        return v

    @field_validator("tree_digest", "input_digest")
    @classmethod
    def _validate_digest(cls, v: str) -> str:
        if not _DIGEST_HEX_PATTERN.fullmatch(v):
            raise ValueError(f"digest must be 64 lowercase hex chars (sha256); got {v!r}.")
        return v

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
        # Enforce the remote-warning relationships that remain observable after
        # sanitization. Historical credential/query redactions may accompany a
        # non-null sanitized locator. Local/unsupported redaction necessarily
        # omits the locator, and an omitted locator carrying other redaction
        # observations must also explain that omission.
        remote_codes = {warning.code for warning in self.remote_warnings}
        has_local_removed = "remote_local_or_unsupported_removed" in remote_codes
        has_supported_redaction = bool(
            remote_codes & {"remote_credentials_removed", "remote_query_fragment_removed"}
        )
        if self.remote_url is not None and has_local_removed:
            raise ValueError(
                "remote_local_or_unsupported_removed requires remote_url=None; "
                "a non-null sanitized locator cannot carry that observation."
            )
        if self.remote_url is None and has_supported_redaction and not has_local_removed:
            raise ValueError(
                "credential/query warnings with remote_url=None require "
                "remote_local_or_unsupported_removed to explain the omitted locator."
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
    # Capture the RAW remote URL and compute the sanitization (sanitized value
    # + warning codes) in the CAPTURE PATH. The capture path is the sole place
    # where URL↔warning correlation is performed: it sanitizes the raw URL,
    # derives the matching warning codes, and supplies BOTH the sanitized URL
    # and the codes to the constructor. The :class:`SourceSnapshot` model
    # accepts whatever it is given — the codes are Literal-validated and the
    # URL is already sanitized — so a sidecar round-trip (which sees the
    # already-sanitized URL and the retained codes) re-validates without
    # re-deriving redactions from the sanitized URL.
    raw_url = _raw_remote_url(repo_path)
    sanitized_url, remote_warn_codes = _classify_remote_url_sanitization(raw_url)
    evidence = _build_dirty_evidence(repo_path) if dirty else None
    input_digest = _envelope_digest(tree_digest, evidence)
    # Canonicalize the warning codes (sorted + unique) so the
    # ``remote_warnings`` field_validator accepts capture output on the first
    # pass regardless of the order the classifier appended the codes.
    canonical_warn_codes = sorted(set(remote_warn_codes))
    return SourceSnapshot(
        commit_sha=commit_sha,
        branch=branch,
        remote_url=sanitized_url,
        remote_warnings=tuple(RemoteWarning(code=w) for w in canonical_warn_codes),
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
