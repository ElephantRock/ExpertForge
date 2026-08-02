"""Descriptor-bound sealing-intent reads.

This module exports the public :class:`ArtifactStore` implementation while
keeping the large Issue #10 store implementation in ``store.py``.  The only
behavioural override is the sealing-intent reader: authorization is derived from
a verified open file descriptor, never from a path inspection that can go stale.
"""

from __future__ import annotations

import os
import stat
from os import stat_result

from expertforge.artifacts.models import ARTIFACT_ID_PATTERN
from expertforge.artifacts.store import ArtifactStore as _ArtifactStore
from expertforge.artifacts.store import ArtifactStoreError

__all__ = ["ArtifactStore"]

_MAX_SEALING_INTENT_BYTES = 256


def _file_identity(st: stat_result, *, source: str) -> tuple[int, int, int, int, int]:
    """Return a fail-closed identity tuple for a regular file."""
    dev = getattr(st, "st_dev", None)
    ino = getattr(st, "st_ino", None)
    ctime_ns = getattr(st, "st_ctime_ns", None)
    if dev in (None, 0) or ino in (None, 0) or ctime_ns in (None, 0):
        raise ArtifactStoreError(
            f"sealing intent {source} identity unavailable; cannot verify path/fd binding"
        )
    return int(dev), int(ino), int(ctime_ns), int(st.st_size), stat.S_IFMT(st.st_mode)


def _validate_intent_bytes(raw: bytes) -> str:
    if len(raw) > _MAX_SEALING_INTENT_BYTES:
        raise ArtifactStoreError("sealing intent marker exceeds bounded read")
    if not raw:
        raise ArtifactStoreError("sealing intent marker is empty (corrupt)")
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ArtifactStoreError(f"sealing intent marker is not valid UTF-8: {exc}") from exc
    if not text:
        raise ArtifactStoreError("sealing intent marker is blank (corrupt)")
    if not ARTIFACT_ID_PATTERN.fullmatch(text):
        raise ArtifactStoreError(
            f"sealing intent marker contains an invalid artifact_id {text!r}"
        )
    return text


class ArtifactStore(_ArtifactStore):
    """Artifact store with descriptor-bound sealing-intent reads."""

    def _read_sealing_intent_artifact_id(self) -> str | None:
        """Read and validate the intent through a verified open descriptor.

        On POSIX, the descriptor is opened first with ``O_NOFOLLOW`` and kept
        open while the pathname is re-inspected.  Because the original inode is
        pinned by the descriptor, unlink/recreate cannot hide behind immediate
        inode reuse.  On Windows, where deleting an open file is not generally
        available with ``os.open``, a pre/open/post identity tuple is compared;
        the tuple includes kernel-managed change time in addition to device and
        inode.  Any unavailable identity fails closed.
        """
        marker = self.sealing_marker_path
        if os.name == "nt":
            return self._read_sealing_intent_windows(marker)
        return self._read_sealing_intent_posix(marker)

    def _read_sealing_intent_posix(self, marker: os.PathLike[str]) -> str | None:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(marker, flags)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ArtifactStoreError(f"cannot open sealing intent marker: {exc}") from exc

        try:
            try:
                fd_st = os.fstat(fd)
            except OSError as exc:
                raise ArtifactStoreError(f"cannot fstat sealing intent fd: {exc}") from exc
            if not stat.S_ISREG(fd_st.st_mode):
                raise ArtifactStoreError("sealing intent fd is not a regular file")

            inject = self.sealing_read_inject
            if inject is not None:
                inject(self.sealing_marker_path)

            try:
                path_st = self.sealing_marker_path.lstat()
            except FileNotFoundError as exc:
                raise ArtifactStoreError(
                    "sealing intent marker was removed between open and path verification"
                ) from exc
            except OSError as exc:
                raise ArtifactStoreError(f"sealing intent marker unreadable: {exc}") from exc
            if not stat.S_ISREG(path_st.st_mode):
                raise ArtifactStoreError("sealing intent marker is not a regular file")

            if _file_identity(path_st, source="path") != _file_identity(fd_st, source="fd"):
                raise ArtifactStoreError(
                    "sealing intent marker was replaced between open and path verification"
                )
            try:
                raw = os.read(fd, _MAX_SEALING_INTENT_BYTES + 1)
            except OSError as exc:
                raise ArtifactStoreError(f"cannot read sealing intent marker: {exc}") from exc
        finally:
            os.close(fd)
        return _validate_intent_bytes(raw)

    def _read_sealing_intent_windows(self, marker: os.PathLike[str]) -> str | None:
        try:
            before_st = self.sealing_marker_path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ArtifactStoreError(f"sealing intent marker unreadable: {exc}") from exc
        if not stat.S_ISREG(before_st.st_mode):
            raise ArtifactStoreError("sealing intent marker is not a regular file")

        inject = self.sealing_read_inject
        if inject is not None:
            inject(self.sealing_marker_path)

        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
        try:
            fd = os.open(marker, flags)
        except FileNotFoundError as exc:
            raise ArtifactStoreError(
                "sealing intent marker was removed between lstat and open"
            ) from exc
        except OSError as exc:
            raise ArtifactStoreError(f"cannot open sealing intent marker: {exc}") from exc

        try:
            try:
                fd_st = os.fstat(fd)
                after_st = self.sealing_marker_path.lstat()
            except OSError as exc:
                raise ArtifactStoreError(f"cannot verify sealing intent marker: {exc}") from exc
            if not stat.S_ISREG(fd_st.st_mode) or not stat.S_ISREG(after_st.st_mode):
                raise ArtifactStoreError("sealing intent marker is not a regular file")
            fd_identity = _file_identity(fd_st, source="fd")
            if _file_identity(before_st, source="pre-open path") != fd_identity:
                raise ArtifactStoreError(
                    "sealing intent marker was replaced between lstat and open"
                )
            if _file_identity(after_st, source="post-open path") != fd_identity:
                raise ArtifactStoreError(
                    "sealing intent marker changed after open"
                )
            try:
                raw = os.read(fd, _MAX_SEALING_INTENT_BYTES + 1)
            except OSError as exc:
                raise ArtifactStoreError(f"cannot read sealing intent marker: {exc}") from exc
        finally:
            os.close(fd)
        return _validate_intent_bytes(raw)
