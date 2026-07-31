"""Deterministic POSIX ustar tar writer (Issue #11, amendment F).

Strict POSIX ustar bytes only — no PAX extensions, no GNU extensions. Fixed for
byte-for-byte determinism: member order, modes, uid/gid, uname/gname, mtime,
padding, and terminator. Header numeric fields use one canonical octal
representation; base-256/GNU numeric encoding is never emitted.

The writer validates path/prefix field limits before writing and emits exactly
two terminal zero blocks.
"""

from __future__ import annotations

from typing import Sequence

__all__ = ["USTAR_BLOCK_SIZE", "TarMember", "build_ustar_archive", "TarWriterError"]

USTAR_BLOCK_SIZE: int = 512

# Field width limits per POSIX ustar.
_NAME_LIMIT = 100
_PREFIX_LIMIT = 155


class TarWriterError(Exception):
    """Raised on tar writer contract violations (bad paths, oversize fields)."""


class TarMember:
    """One ordered archive member (name + raw bytes)."""

    __slots__ = ("name", "data")

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self.data = data

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"TarMember(name={self.name!r}, size={len(self.data)})"


def _split_name_prefix(name: str) -> tuple[str, str]:
    """Split ``name`` into (name, prefix) fitting ustar 100/155-byte limits.

    POSIX ustar: a name longer than 100 bytes may be split by placing a prefix
    (≤155 bytes, no trailing slash) in the prefix field and the remainder (≤100
    bytes) in the name field, with the split occurring at a '/' boundary. If no
    valid split exists, raise.
    """
    encoded = name.encode("utf-8")
    if len(encoded) <= _NAME_LIMIT:
        return name, ""
    # Find the last '/' such that the suffix (after it) is ≤100 bytes and the
    # prefix (up to and including nothing) is ≤155 bytes.
    # Walk from the longest valid prefix.
    best: tuple[str, str] | None = None
    # Consider split points at '/' boundaries where suffix <= 100 bytes.
    parts = name.split("/")
    for i in range(1, len(parts)):
        prefix = "/".join(parts[:i])
        suffix = "/".join(parts[i:])
        if len(suffix.encode("utf-8")) <= _NAME_LIMIT and len(prefix.encode("utf-8")) <= _PREFIX_LIMIT:
            best = (suffix, prefix)
            break
    if best is None:
        raise TarWriterError(
            f"member name {name!r} cannot be split to fit ustar 100/155-byte limits"
        )
    return best


def _validate_member_name(name: str) -> None:
    """Reject absolute, traversal, backslash, and empty-member names."""
    if not name:
        raise TarWriterError("member name must be non-empty")
    if name.startswith("/"):
        raise TarWriterError(f"member name {name!r} must be relative, not absolute")
    if "\\" in name:
        raise TarWriterError(f"member name {name!r} must not contain backslashes")
    parts = name.split("/")
    if any(p in ("", "..", ".") for p in parts):
        raise TarWriterError(
            f"member name {name!r} must not contain '.', '..', or empty components"
        )


def _octal_field(value: int, width: int, *, trailing_null: bool = True) -> bytes:
    """Encode ``value`` as a canonical octal numeric field of ``width`` bytes.

    ustar numeric fields are width bytes: a NUL- or space-terminated octal
    representation. Size is 12 bytes (11 octal digits + NUL); mode is 8 bytes
    (7 octal digits + space or NUL); uid/gid are 8 bytes; mtime is 12 bytes.

    We use the canonical form: zero-padded octal digits followed by a single
    NUL terminator (the canonical POSIX form), except where a space terminator
    is conventional (none here — we use NUL throughout for determinism).
    """
    # The field reserves the final byte for the terminator.
    digits_width = width - 1
    oct_digits = oct(value)[2:]  # no '0o' prefix
    if len(oct_digits) > digits_width:
        raise TarWriterError(
            f"value {value} does not fit in {digits_width} octal digits (width={width})"
        )
    padded = oct_digits.rjust(digits_width, "0").encode("ascii")
    if trailing_null:
        return padded + b"\x00"
    return padded + b"\x00"


def _build_header(
    *,
    name: str,
    size: int,
    mode: int = 0o644,
    mtime: int = 0,
    typeflag: bytes = b"0",
) -> bytes:
    """Build a single 512-byte ustar header for a regular file."""
    name_field, prefix_field = _split_name_prefix(name)
    name_bytes = name_field.encode("utf-8")
    if len(name_bytes) > _NAME_LIMIT:  # pragma: no cover - _split_name_prefix guards
        raise TarWriterError(f"name field overflow for {name!r}")

    prefix_bytes = prefix_field.encode("utf-8")
    if len(prefix_bytes) > _PREFIX_LIMIT:
        raise TarWriterError(f"prefix field overflow for {name!r}")

    header = bytearray(USTAR_BLOCK_SIZE)
    # name: 100 bytes, NUL-padded.
    header[0:100] = name_bytes.ljust(100, b"\x00")
    # mode: 8 bytes octal.
    header[100:108] = _octal_field(mode, 8)
    # uid: 8 bytes octal (0).
    header[108:116] = _octal_field(0, 8)
    # gid: 8 bytes octal (0).
    header[116:124] = _octal_field(0, 8)
    # size: 12 bytes octal.
    header[124:136] = _octal_field(size, 12)
    # mtime: 12 bytes octal (0 for determinism).
    header[136:148] = _octal_field(mtime, 12)
    # chksum: 8 bytes, filled with spaces during checksum computation.
    header[148:156] = b"        "
    # typeflag: 1 byte regular file '0'.
    header[156:157] = typeflag
    # linkname: 100 bytes NUL.
    # (158:258 left as zeros)
    # magic: 6 bytes "ustar\0".
    header[257:263] = b"ustar\x00"
    # version: 2 bytes "00".
    header[263:265] = b"00"
    # uname: 32 bytes empty (NUL-filled).
    # gname: 32 bytes empty (NUL-filled).
    # (265:329 left as zeros)
    # devmajor: 8 bytes octal 0.
    header[329:337] = _octal_field(0, 8)
    # devminor: 8 bytes octal 0.
    header[337:345] = _octal_field(0, 8)
    # prefix: 155 bytes.
    header[345:500] = prefix_bytes.ljust(155, b"\x00")
    # padding to 512 left as zeros.

    # Compute the checksum: sum of all unsigned bytes in the header with the
    # chksum field treated as 8 spaces (already set above).
    chksum = sum(header) & 0o777777
    chk_field = f"{chksum:06o}\x00 ".encode("ascii")
    header[148:156] = chk_field
    return bytes(header)


def _pad_to_block(data: bytes) -> bytes:
    """Pad ``data`` with NULs to a multiple of USTAR_BLOCK_SIZE."""
    remainder = len(data) % USTAR_BLOCK_SIZE
    if remainder == 0:
        return data
    return data + b"\x00" * (USTAR_BLOCK_SIZE - remainder)


def build_ustar_archive(members: Sequence[TarMember]) -> bytes:
    """Build a strict POSIX ustar archive from ordered ``members``.

    Guarantees:
    - members written in the given order;
    - mode 0o644, uid/gid 0, uname/gname empty, mtime 0;
    - magic ``ustar\\0``, version ``00``;
    - canonical octal numeric fields (no base-256);
    - 512-byte block alignment; two trailing zero-block terminator;
    - total size a multiple of 512 bytes.
    """
    seen: set[str] = set()
    out = bytearray()
    for member in members:
        _validate_member_name(member.name)
        if member.name in seen:
            raise TarWriterError(f"duplicate member name {member.name!r}")
        seen.add(member.name)
        header = _build_header(name=member.name, size=len(member.data))
        out.extend(header)
        out.extend(_pad_to_block(member.data))
    # Two zero-block terminator.
    out.extend(b"\x00" * (USTAR_BLOCK_SIZE * 2))
    return bytes(out)
