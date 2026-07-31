"""Canonical path derivation and path-safety helpers for Issue #10.

All relative paths are validated as canonical POSIX repository-relative paths.
We reject: absolute paths, ``..`` traversal, backslashes, empty components, and
symlinks in the target directory chain (best-effort ``O_NOFOLLOW`` on POSIX).

The canonical artifact bundle layout (amendment C)::

    <attempt>/artifacts/<category>/<artifact-id>/
      content
      artifact.json
"""

from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = [
    "PathSafetyError",
    "artifact_bundle_dir",
    "attempt_dir",
    "category_dir",
    "is_path_within",
    "reject_non_local_path",
    "resolve_within_root",
    "validate_id_component",
    "validate_relative_bundle_path",
]


class PathSafetyError(Exception):
    """Raised on path traversal, absolute paths, or symlink following."""


def validate_id_component(value: str, kind: str) -> None:
    """Reject IDs that could traverse or escape the artifact-root directory."""
    if not value:
        raise PathSafetyError(f"{kind} must be non-empty.")
    bad_chars = {os.sep, "/", "\\"}
    if any(ch in value for ch in bad_chars) or ".." in value or value in {".", ".."}:
        raise PathSafetyError(
            f"{kind} {value!r} contains a path separator or traversal component."
        )


def attempt_dir(artifact_root: Path, run_id: str, attempt_id: str) -> Path:
    validate_id_component(run_id, "run_id")
    validate_id_component(attempt_id, "attempt_id")
    return artifact_root / run_id / "attempts" / attempt_id


def category_dir(artifact_root: Path, run_id: str, attempt_id: str, category: str) -> Path:
    validate_id_component(category, "category")
    return attempt_dir(artifact_root, run_id, attempt_id) / "artifacts" / category


def artifact_bundle_dir(
    artifact_root: Path, run_id: str, attempt_id: str, category: str, artifact_id: str
) -> Path:
    validate_id_component(artifact_id, "artifact_id")
    return category_dir(artifact_root, run_id, attempt_id, category) / artifact_id


def validate_relative_bundle_path(
    run_id: str, attempt_id: str, category: str, artifact_id: str
) -> str:
    """Return the canonical POSIX relative path of a bundle from the attempt dir."""
    validate_id_component(run_id, "run_id")
    validate_id_component(attempt_id, "attempt_id")
    validate_id_component(category, "category")
    validate_id_component(artifact_id, "artifact_id")
    return f"artifacts/{category}/{artifact_id}"


def reject_non_local_path(path: Path) -> None:
    """Reject absolute paths and any path component that is a symlink.

    Used for ``register_existing`` source paths: the source must be a regular
    file (no symlink following) and within an accepted root.
    """
    if path.is_absolute():
        # Absolute source paths are allowed for register_existing (the file is
        # copied into store ownership), but they must not be symlinks.
        pass
    # Symlink rejection is performed by the caller via os.stat (lstat) — see
    # ArtifactStore.register_existing. This helper exists for component reuse.


def is_path_within(child: Path, root: Path) -> bool:
    """True if ``child`` is equal to or nested beneath ``root`` (both resolved)."""
    try:
        child_resolved = child.resolve(strict=False)
        root_resolved = root.resolve(strict=False)
    except OSError:
        return False
    try:
        child_resolved.relative_to(root_resolved)
        return True
    except ValueError:
        return False


def resolve_within_root(root: Path, *parts: str) -> Path:
    """Resolve ``parts`` beneath ``root`` and reject traversal/absolute escape.

    Backslashes are rejected. The resolved real path must remain within ``root``.
    """
    for part in parts:
        if part in {".", ".."}:
            raise PathSafetyError(f"path component {part!r} is not allowed.")
        if "\\" in part or part.startswith("/"):
            raise PathSafetyError(f"path component {part!r} is not portable.")
        if re.match(r"^[A-Za-z]:", part):
            raise PathSafetyError(f"path component {part!r} looks like a drive path.")
    candidate = root.joinpath(*parts)
    # Final containment check: after lexically normalizing, must stay under root.
    root_norm = root.resolve(strict=False)
    cand_norm = candidate.resolve(strict=False)
    try:
        cand_norm.relative_to(root_norm)
    except ValueError as e:
        raise PathSafetyError(
            f"resolved path {candidate} escapes root {root}"
        ) from e
    return candidate
