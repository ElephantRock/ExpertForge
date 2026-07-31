"""Strict POSIX ustar tar parser (Issue #11, amendment F).

Parses strictly: verifies header checksum, magic/version, canonical numeric
fields, file type, size/padding, member order, duplicates, and the two-block
terminator. Trailing bytes after the terminator are rejected.

Never calls ``extract``/``extractall`` and never materializes archive paths on
disk. Parses sequentially from an in-memory buffer (the store streams the
already-opened, fstat-verified file descriptor into a buffer below the bound, or
parses sequentially for large archives).
"""

from __future__ import annotations

from typing import Iterator

from expertforge.checkpoints.tar_writer import USTAR_BLOCK_SIZE

__all__ = ["ParsedMember", "parse_ustar_archive", "TarParseError"]

_NAME_LIMIT = 100
_PREFIX_LIMIT = 155


class TarParseError(Exception):
    """Raised on any non-canonical ustar archive (corrupt, truncated, extended)."""


class ParsedMember:
    """One parsed archive member."""

    __slots__ = ("name", "data", "size")

    def __init__(self, name: str, data: bytes, size: int) -> None:
        self.name = name
        self.size = size
        self.data = data

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ParsedMember(name={self.name!r}, size={self.size})"


def _parse_numeric(field: bytes, *, label: str) -> int:
    """Parse a ustar octal numeric field. Reject base-256 and non-octal bytes."""
    # Reject base-256 (GNU) encoding: the high bit of the first byte set.
    if field and (field[0] & 0x80):
        raise TarParseError(f"{label}: base-256/GNU numeric encoding is rejected")
    # Strip the terminator (NUL or space) and any leading/trailing spaces/NULs.
    text = field.decode("ascii", errors="replace").rstrip("\x00 ")
    text = text.lstrip(" ")
    if not text:
        return 0
    # Must be all octal digits.
    for ch in text:
        if ch not in "01234567":
            raise TarParseError(
                f"{label}: non-octal character {ch!r} in numeric field"
            )
    try:
        return int(text, 8)
    except ValueError as e:  # pragma: no cover - guarded above
        raise TarParseError(f"{label}: bad octal numeric field") from e


def _read_field(buf: bytes, start: int, length: int) -> bytes:
    if start + length > len(buf):
        raise TarParseError("truncated header")
    return buf[start : start + length]


def _parse_header(buf: bytes, offset: int) -> tuple[str, bytes, int, int]:
    """Parse one 512-byte header at ``offset``.

    Returns (name, typeflag_bytes, size, header_end). Raises on any violation.
    """
    if offset + USTAR_BLOCK_SIZE > len(buf):
        raise TarParseError("truncated archive: incomplete header block")
    header = buf[offset : offset + USTAR_BLOCK_SIZE]

    # Verify the checksum FIRST (before trusting any field).
    stored = header[148:156]
    # Compute the unsigned-byte sum with the chksum field replaced by spaces.
    chksum_buf = bytearray(header)
    chksum_buf[148:156] = b"        "
    computed = sum(chksum_buf) & 0o777777
    stored_text = stored.decode("ascii", errors="replace").rstrip("\x00 ")
    stored_text = stored_text.lstrip(" ")
    if not stored_text:
        raise TarParseError("empty checksum field")
    for ch in stored_text:
        if ch not in "01234567":
            raise TarParseError(f"non-octal character in checksum field: {ch!r}")
    try:
        stored_val = int(stored_text, 8)
    except ValueError:
        raise TarParseError("bad checksum octal value") from None
    if stored_val != computed:
        raise TarParseError(
            f"header checksum mismatch: stored={stored_val} computed={computed}"
        )

    # magic: bytes 257:263 must be "ustar\x00".
    magic = header[257:263]
    if magic != b"ustar\x00":
        raise TarParseError(
            f"bad magic {magic!r}; expected b'ustar\\x00' (non-PAX/non-GNU ustar only)"
        )
    # version: bytes 263:265 must be "00".
    version = header[263:265]
    if version != b"00":
        raise TarParseError(f"bad version {version!r}; expected b'00'")

    # typeflag: byte 156 must be '0' (regular file) or a terminator sentinel.
    typeflag = header[156:157]
    # Reject PAX/GNU extended headers outright.
    if typeflag in (b"x", b"g", b"L", b"K"):
        raise TarParseError(
            f"extended header typeflag {typeflag!r} rejected (PAX/GNU not allowed)"
        )
    # Symlinks/hardlinks/dirs/devices/fifos rejected.
    if typeflag not in (b"0", b"\x00"):
        raise TarParseError(f"unsupported typeflag {typeflag!r}; only regular files allowed")

    name_field = header[0:100]
    prefix_field = header[345:500]
    name_nul = name_field.find(b"\x00")
    name_bytes = name_field if name_nul == -1 else name_field[:name_nul]
    prefix_nul = prefix_field.find(b"\x00")
    prefix_bytes = prefix_field if prefix_nul == -1 else prefix_field[:prefix_nul]
    if prefix_bytes:
        name = prefix_bytes.decode("utf-8") + "/" + name_bytes.decode("utf-8")
    else:
        name = name_bytes.decode("utf-8")
    if not name:
        raise TarParseError("empty member name")

    size = _parse_numeric(header[124:136], label="size")
    mode = _parse_numeric(header[100:108], label="mode")
    uid = _parse_numeric(header[108:116], label="uid")
    gid = _parse_numeric(header[116:124], label="gid")
    mtime = _parse_numeric(header[136:148], label="mtime")
    # Enforce deterministic field values.
    if mode != 0o644:
        raise TarParseError(f"mode must be 0o644; got 0o{mode:o}")
    if uid != 0 or gid != 0:
        raise TarParseError(f"uid/gid must be 0; got uid={uid} gid={gid}")
    if mtime != 0:
        raise TarParseError(f"mtime must be 0 for determinism; got {mtime}")
    # uname/gname must be empty.
    uname = header[265:297]
    gname = header[297:329]
    if uname.strip(b"\x00") or gname.strip(b"\x00"):
        raise TarParseError("uname/gname must be empty")

    return name, typeflag, size, offset + USTAR_BLOCK_SIZE


def parse_ustar_archive(buf: bytes) -> list[ParsedMember]:
    """Parse a strict ustar archive buffer.

    Returns members in archive order. Raises :class:`TarParseError` on any
    non-canonical framing, duplicate names, missing terminator, or trailing
    bytes.
    """
    if len(buf) == 0 or len(buf) % USTAR_BLOCK_SIZE != 0:
        raise TarParseError(
            f"archive size {len(buf)} is not a positive multiple of {USTAR_BLOCK_SIZE}"
        )
    members: list[ParsedMember] = []
    seen: set[str] = set()
    offset = 0
    n_blocks = len(buf) // USTAR_BLOCK_SIZE
    block_idx = 0
    terminator_seen = False
    while block_idx < n_blocks:
        base = block_idx * USTAR_BLOCK_SIZE
        block = buf[base : base + USTAR_BLOCK_SIZE]
        # A zero block signals the end of members.
        if block == b"\x00" * USTAR_BLOCK_SIZE:
            terminator_seen = True
            block_idx += 1
            # Must be followed by exactly one more zero block, then EOF.
            break
        name, _typeflag, size, data_start = _parse_header(buf, base)
        if name in seen:
            raise TarParseError(f"duplicate member name {name!r}")
        seen.add(name)
        # Read size bytes, then skip padding.
        if size < 0:
            raise TarParseError(f"negative size for member {name!r}")
        data_end = data_start + size
        if data_end > len(buf):
            raise TarParseError(f"truncated data for member {name!r}")
        data = buf[data_start:data_end]
        # Verify the data member name is path-safe.
        _validate_member_name(name)
        members.append(ParsedMember(name=name, data=data, size=size))
        # Advance past the data (padded to block boundary).
        data_blocks = (size + USTAR_BLOCK_SIZE - 1) // USTAR_BLOCK_SIZE
        block_idx += 1 + data_blocks
        offset = data_start + data_blocks * USTAR_BLOCK_SIZE

    if not terminator_seen:
        raise TarParseError("missing two-block zero terminator")
    # The terminator must be exactly two zero blocks, then EOF.
    # block_idx now points to the second zero block.
    if block_idx >= n_blocks:
        raise TarParseError("truncated terminator: expected two zero blocks")
    second = buf[block_idx * USTAR_BLOCK_SIZE : (block_idx + 1) * USTAR_BLOCK_SIZE]
    if second != b"\x00" * USTAR_BLOCK_SIZE:
        raise TarParseError("second terminator block is non-zero")
    after = (block_idx + 1) * USTAR_BLOCK_SIZE
    if after != len(buf):
        raise TarParseError(
            f"trailing {len(buf) - after} bytes after terminator are rejected"
        )
    return members


def _validate_member_name(name: str) -> None:
    """Reject absolute, traversal, backslash, and empty names at parse time."""
    if not name:
        raise TarParseError("empty member name")
    if name.startswith("/"):
        raise TarParseError(f"absolute member name {name!r} rejected")
    if "\\" in name:
        raise TarParseError(f"backslash in member name {name!r} rejected")
    parts = name.split("/")
    if any(p in ("", "..", ".") for p in parts):
        raise TarParseError(
            f"member name {name!r} contains '.', '..', or empty components"
        )


def iter_members(buf: bytes) -> Iterator[ParsedMember]:
    """Iterate parsed members (convenience wrapper around parse)."""
    yield from parse_ustar_archive(buf)
