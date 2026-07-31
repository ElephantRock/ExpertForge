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
        flags = os.O_RDWR | os.O_CREAT | os.O_BINARY
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
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
            self._owns_dir = True

    def __enter__(self) -> AttemptLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
