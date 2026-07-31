"""Platform file locking for registry mutation serialization (amendment E).

Do not rely on ``O_APPEND`` plus one ``os.write`` as a cross-process
record-atomicity guarantee for regular files. Every registry mutation acquires
a per-attempt advisory lock for sequence allocation, transition validation,
append, and fsync.

Standard-library platform adapters are used:

- ``fcntl`` (``flock``) on POSIX;
- ``msvcrt`` (``locking``) on Windows.

Version 1 supports local filesystems only; network/distributed filesystems are
outside the contract.
"""

from __future__ import annotations

import os
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

__all__ = ["AttemptLock", "LockUnavailableError"]

_IS_WINDOWS = sys.platform == "win32" or os.name == "nt"

# Platform-conditional standard-library lock adapters. Imported at module load
# behind the platform guard so the binding exists on the running platform; the
# unused binding on the other platform is typed as Any and never accessed at
# runtime (the acquire/release methods branch on _IS_WINDOWS). mypy analyzes
# both branches, so attribute accesses are guarded by ``# type: ignore`` where
# the typeshed stub for the absent platform is not loaded.
_fcntl: Any = None
_msvcrt: Any = None
if not _IS_WINDOWS:
    try:
        import fcntl as _fcntl
    except ImportError:  # pragma: no cover - POSIX always has fcntl
        _fcntl = None
else:
    try:
        import msvcrt as _msvcrt
    except ImportError:  # pragma: no cover - Windows always has msvcrt
        _msvcrt = None


class LockUnavailableError(Exception):
    """Raised when the platform/filesystem cannot provide a registry lock."""


def _verify_no_symlinks_in_chain(anchor: Path, target: Path) -> None:
    """Reject any symlink component between ``anchor`` and ``target`` (item #1).

    Walks the lexical components of ``target`` starting at ``anchor`` and
    ``lstat``s each intermediate component. A symlink component (including a
    symlinked final element) raises :class:`LockUnavailableError`. This mirrors
    ``expertforge.artifacts.store._verify_no_symlinks_in_chain`` without
    importing the store module (which would create a circular import).
    """
    import stat as _stat

    try:
        rel = target.relative_to(anchor)
    except ValueError as e:
        raise LockUnavailableError(f"lock parent {target} is not within anchor {anchor}") from e
    current = anchor
    for part in rel.parts:
        current = current / part
        try:
            st = os.lstat(current)
        except FileNotFoundError:
            # A missing component is expected during makedirs; it is not a
            # symlink, so it is allowed.
            continue
        except OSError as e:
            raise LockUnavailableError(f"cannot lstat lock-path component {current}: {e}") from e
        if _stat.S_ISLNK(st.st_mode):
            raise LockUnavailableError(
                f"lock-path component {current} is a symlink; refused (item #1)."
            )


def _symlink_safe_makedirs(target: Path) -> None:
    """Create ``target`` (and parents) only after verifying the existing path
    chain is symlink-free, then re-verify after creation (item #1).

    Checks both intermediate components AND the target itself for symlinks.
    The deepest existing ancestor (anchor) is also checked — if it is itself
    a symlink, ``mkdir(parents=True)`` would follow it and create directories
    outside the artifact tree.
    """
    import stat as _stat

    # Check if target itself is a symlink (lstat doesn't follow).
    try:
        target_st = os.lstat(target)
        if _stat.S_ISLNK(target_st.st_mode):
            raise LockUnavailableError(f"target {target} is a symlink; refused (item #1).")
    except FileNotFoundError:
        pass  # target doesn't exist yet — fine

    # Find the deepest existing ancestor to anchor the chain walk on.
    anchor = target.parent
    while anchor != anchor.parent:
        try:
            os.lstat(anchor)
            break
        except FileNotFoundError:
            anchor = anchor.parent

    # Explicitly check the anchor itself for symlink — _verify_no_symlinks_in_chain
    # only checks components BELOW the anchor, so a symlinked ancestor would be missed.
    try:
        anchor_st = os.lstat(anchor)
        if _stat.S_ISLNK(anchor_st.st_mode):
            raise LockUnavailableError(
                f"deepest existing ancestor {anchor} is a symlink; "
                "mkdir would follow it outside the artifact tree (item #1)."
            )
    except FileNotFoundError:
        pass  # anchor doesn't exist — nothing to check

    _verify_no_symlinks_in_chain(anchor, target)
    target.mkdir(parents=True, exist_ok=True)
    _verify_no_symlinks_in_chain(anchor, target)


class AttemptLock(AbstractContextManager["AttemptLock"]):
    """Per-attempt advisory lock context manager.

    On POSIX this is a ``fcntl.flock`` exclusive lock on a sibling lock file
    beneath the attempt directory. On Windows this is an ``msvcrt.locking``
    exclusive lock on byte 0 of the same lock file. On any unsupported platform
    :class:`LockUnavailableError` is raised.

    The lock file is created with ``O_CREAT | O_EXCL``-free best-effort: a plain
    open with read+write+create is used because the lock itself is the
    coordination primitive, not the file's existence.
    """

    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._fd: int | None = None
        self._owns_dir = False

    def acquire(self) -> None:
        if _IS_WINDOWS:
            self._acquire_windows()
        else:
            self._acquire_posix()

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        if _IS_WINDOWS:
            self._release_windows(fd)
        else:
            self._release_posix(fd)

    # -- POSIX -------------------------------------------------------------

    def _acquire_posix(self) -> None:
        if _fcntl is None:
            raise LockUnavailableError(
                "POSIX advisory locks (fcntl) are unavailable on this platform."
            )
        self._ensure_parent()
        # O_CLOEXEC keeps the lock fd out of forked children.
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(self._lock_path, flags, 0o600)
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX)
        except OSError as e:
            os.close(fd)
            raise LockUnavailableError(
                f"Could not acquire POSIX advisory lock {self._lock_path}: {e}"
            ) from e
        self._fd = fd

    def _release_posix(self, fd: int) -> None:
        if _fcntl is not None:
            try:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)

    # -- Windows -----------------------------------------------------------

    def _acquire_windows(self) -> None:
        if _msvcrt is None:
            raise LockUnavailableError(
                "Windows advisory locks (msvcrt) are unavailable on this platform."
            )
        self._ensure_parent()
        # ``os.O_BINARY`` only exists on Windows; guard with ``getattr`` so mypy
        # is clean on both Linux and Windows typesheds (the Linux stub does not
        # declare O_BINARY even though this branch only runs on Windows).
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        fd = os.open(self._lock_path, flags, 0o600)
        # Lock byte 0 exclusively. Retry briefly to avoid spurious failures
        # under contention (msvcrt.locking raises on contention rather than
        # blocking). A bounded spin keeps this deterministic.
        import time

        deadline_spins = 1000
        for _ in range(deadline_spins):
            try:
                _msvcrt.locking(fd, _msvcrt.LK_LOCK, 1)
                self._fd = fd
                return
            except OSError:
                time.sleep(0.001)
        os.close(fd)
        raise LockUnavailableError(
            f"Could not acquire Windows advisory lock {self._lock_path} "
            f"after {deadline_spins} spins."
        )

    def _release_windows(self, fd: int) -> None:
        if _msvcrt is not None:
            try:
                # Seek to byte 0 and unlock the held byte.
                os.lseek(fd, 0, os.SEEK_SET)
                _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        os.close(fd)

    # -- shared ------------------------------------------------------------

    def _ensure_parent(self) -> None:
        parent = self._lock_path.parent
        # Always verify the path chain for symlinks, even when the parent
        # already exists (it may exist AS a symlink). Then create if needed.
        _symlink_safe_makedirs(parent)
        if not parent.exists():
            self._owns_dir = True

    def __enter__(self) -> AttemptLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
