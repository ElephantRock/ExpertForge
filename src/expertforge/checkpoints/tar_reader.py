"""Strict POSIX ustar tar parser (Issue #11, amendment F).

Parses strictly: verifies header checksum, magic/version, canonical numeric
fields, file type, size/padding, member order, duplicates, and the two-block
terminator. Trailing bytes after the terminator are rejected. Enforces the
archive/member/tensor count and byte limits during parsing (item 11).

Never calls ``extract``/``extractall`` and never materializes archive paths on
disk. Parses sequentially from an in-memory buffer (the store streams the
already-opened, fstat-verified file descriptor into a buffer below the bound, or
parses sequentially for large archives).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

from expertforge.checkpoints.tar_writer import USTAR_BLOCK_SIZE

__all__ = [
    "BLOCK_READ_CHUNK",
    "ParsedMember",
    "parse_ustar_archive",
    "parse_ustar_archive_streaming",
    "TarParseError",
]

_NAME_LIMIT = 100
_PREFIX_LIMIT = 155

# Resource limits enforced DURING parsing (item 11). Imported lazily to avoid a
# circular import (models imports nothing from tar_reader, but keep it lazy).
_MAX_ARCHIVE_BYTES_DEFAULT = 64 * 1024 * 1024 * 1024
_MAX_MEMBER_COUNT_DEFAULT = 10_000
_MAX_TENSOR_COUNT_DEFAULT = 10_000
_MAX_MEMBER_BYTES_DEFAULT = 256 * 1024 * 1024  # per-component-member fallback
_MAX_TENSOR_BYTES_DEFAULT = 32 * 1024 * 1024 * 1024  # total-tensor fallback

# Canonical member-name patterns for the fixed component/tensor order (item 11).
_STATE_MEMBER_RE = re.compile(r"^state/[A-Za-z0-9_]+\.json$")
_TENSOR_MEMBER_RE = re.compile(r"^tensors/[0-9]+\.bin$")


class TarParseError(Exception):
    """Raised on any non-canonical ustar archive (corrupt, truncated, extended)."""


class ParsedMember:
    """One parsed archive member.

    For large tensor members, ``data`` may be ``None`` and ``temp_path`` holds
    a Path to a temp file containing the member bytes. This keeps peak heap
    bounded to one member's data during streaming parse.
    """

    __slots__ = ("name", "data", "size", "temp_path")

    def __init__(
        self,
        name: str,
        data: bytes | None,
        size: int,
        temp_path: Path | None = None,
    ) -> None:
        self.name = name
        self.size = size
        self.data = data
        self.temp_path = temp_path

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ParsedMember(name={self.name!r}, size={self.size})"


def _parse_numeric(field: bytes, *, label: str) -> int:
    """Parse a ustar octal numeric field, requiring the EXACT writer form (item 8).

    The writer emits exactly: zero-padded ASCII octal digits filling all but the
    final byte, followed by a single NUL terminator. This is byte-canonical:
    every numeric field (mode, uid, gid, size, mtime, devmajor, devminor) carries
    the identical encoding the writer produces.

    Rejected (all classified as non-canonical):
    - base-256 / GNU numeric encoding (high bit of the first byte set);
    - old-style all-NUL zero (the writer never emits this for a numeric field —
      it emits ``0000000\\0`` for zero);
    - space-terminated fields, leading/trailing spaces, embedded NULs, and any
      byte that is not an ASCII octal digit before the final NUL terminator.

    The uname/gname fields are NOT numeric (they are validated as empty NUL
    buffers elsewhere), so no all-NUL acceptance is needed here.
    """
    # Reject base-256 (GNU) encoding: the high bit of the first byte set.
    if field and (field[0] & 0x80):
        raise TarParseError(f"{label}: base-256/GNU numeric encoding is rejected")
    # Reject old-style all-NUL zero: the writer emits the canonical
    # ``<zero-padded octal digits> + NUL`` form for every numeric field, including
    # zero (e.g. ``0000000\\0`` for an 8-byte field). An all-NUL field is a
    # different byte encoding and is therefore non-canonical (item 8).
    if field == b"\x00" * len(field):
        raise TarParseError(
            f"{label}: old-style all-NUL numeric field rejected "
            "(writer emits canonical zero-padded octal + NUL)"
        )
    # Canonical form: <zero-padded octal digits> + <single NUL terminator>. The
    # final byte MUST be NUL; everything before it MUST be ASCII octal digits
    # (no leading spaces, no embedded spaces/NULs, no old-style terminator).
    if field[-1:] != b"\x00":
        raise TarParseError(
            f"{label}: non-canonical octal field (missing NUL terminator): {field!r}"
        )
    digits = field[:-1]
    if not digits:
        raise TarParseError(f"{label}: empty numeric field")
    for ch in digits:
        if ch not in b"01234567":
            raise TarParseError(
                f"{label}: non-canonical octal field "
                f"(non-octal/space byte {bytes([ch])!r}): {field!r}"
            )
    text = digits.decode("ascii")
    # Leading zeros ARE canonical (the writer zero-pads), so accept them.
    return int(text, 8)


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

    # Verify the checksum FIRST (before trusting any field). The writer emits
    # the EXACT canonical form ``<6 zero-padded octal digits> + NUL + space``
    # (item 8): require byte-for-byte agreement rather than the lenient
    # NUL/space-stripped parse that accepts alternate spacings.
    stored = header[148:156]
    # Compute the unsigned-byte sum with the chksum field replaced by spaces.
    chksum_buf = bytearray(header)
    chksum_buf[148:156] = b"        "
    computed = sum(chksum_buf) & 0o777777
    canonical_chk = f"{computed:06o}\x00 ".encode("ascii")
    if stored != canonical_chk:
        raise TarParseError(
            f"header checksum field {stored!r} is not the canonical writer form "
            f"{canonical_chk!r} (computed sum={computed})"
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

    # typeflag: byte 156 must be EXACTLY b"0" (regular file). The writer always
    # emits b"0"; a NUL byte (the alternate pre-POSIX regular-file marker) is a
    # different byte encoding and is rejected (item 8).
    typeflag = header[156:157]
    # Reject PAX/GNU extended headers outright.
    if typeflag in (b"x", b"g", b"L", b"K"):
        raise TarParseError(f"extended header typeflag {typeflag!r} rejected (PAX/GNU not allowed)")
    if typeflag != b"0":
        raise TarParseError(
            f"unsupported typeflag {typeflag!r}; the writer emits exactly b'0' "
            "(NUL/alternate regular-file markers are non-canonical)"
        )

    name_field = header[0:100]
    prefix_field = header[345:500]
    name_nul = name_field.find(b"\x00")
    name_bytes = name_field if name_nul == -1 else name_field[:name_nul]
    prefix_nul = prefix_field.find(b"\x00")
    prefix_bytes = prefix_field if prefix_nul == -1 else prefix_field[:prefix_nul]
    # Item 8: reject nonzero bytes after the first NUL in the name/prefix fields.
    # The writer NUL-pads these fixed-width fields; any nonzero byte after the
    # terminating NUL is a different (non-canonical) encoding.
    if name_nul != -1:
        tail = name_field[name_nul + 1 :]
        if any(b != 0 for b in tail):
            raise TarParseError("nonzero bytes after NUL in the name field")
    if prefix_nul != -1:
        ptail = prefix_field[prefix_nul + 1 :]
        if any(b != 0 for b in ptail):
            raise TarParseError("nonzero bytes after NUL in the prefix field")
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
    # Item 7: validate ALL remaining fixed header fields for byte-canonical
    # v1 archives. linkname (157:257, 100 bytes) must be all-NUL because every
    # checkpoint member is a regular file with no link target.
    linkname = header[157:257]
    if linkname.strip(b"\x00"):
        raise TarParseError("linkname must be empty for regular-file members")
    # devmajor (329:337) and devminor (337:345) must decode to canonical zero.
    # The writer emits the octal form ``0000000\\0``; both that and the all-NUL
    # form parse to zero. Any nonzero value or alternative encoding is rejected
    # by _parse_numeric (it rejects base-256 and non-canonical octal).
    devmajor = _parse_numeric(header[329:337], label="devmajor")
    devminor = _parse_numeric(header[337:345], label="devminor")
    if devmajor != 0 or devminor != 0:
        raise TarParseError(
            f"devmajor/devminor must be zero; got devmajor={devmajor} devminor={devminor}"
        )
    # The trailing reserved/padding area (500:512) must be all-NUL.
    reserved = header[500:512]
    if reserved.strip(b"\x00"):
        raise TarParseError("trailing header reserved bytes must be zero")

    return name, typeflag, size, offset + USTAR_BLOCK_SIZE


def parse_ustar_archive(
    buf: bytes,
    *,
    max_archive_bytes: int | None = None,
    max_member_count: int | None = None,
    max_tensor_count: int | None = None,
    validate_member_order: bool = True,
) -> list[ParsedMember]:
    """Parse a strict ustar archive buffer (item 11).

    Returns members in archive order. Raises :class:`TarParseError` on any
    non-canonical framing, non-canonical octal fields, nonzero padding, duplicate
    names, missing terminator, trailing bytes, resource-limit violations, or (by
    default) a violation of the fixed checkpoint member order
    (manifest.json → state/*.json → tensors/*.bin).
    """
    # Item 11: enforce archive byte limit BEFORE any allocation/parsing.
    archive_limit = max_archive_bytes or _get_limit("MAX_ARCHIVE_BYTES", _MAX_ARCHIVE_BYTES_DEFAULT)
    if len(buf) > archive_limit:
        raise TarParseError(f"archive size {len(buf)} exceeds max_archive_bytes ({archive_limit})")
    if len(buf) == 0 or len(buf) % USTAR_BLOCK_SIZE != 0:
        raise TarParseError(
            f"archive size {len(buf)} is not a positive multiple of {USTAR_BLOCK_SIZE}"
        )
    member_limit = max_member_count or _get_limit("MAX_MEMBER_COUNT", _MAX_MEMBER_COUNT_DEFAULT)
    tensor_limit = max_tensor_count or _get_limit("MAX_TENSOR_COUNT", _MAX_TENSOR_COUNT_DEFAULT)
    component_member_limit = _get_limit("MAX_COMPONENT_MEMBER_BYTES", _MAX_MEMBER_BYTES_DEFAULT)
    tensor_member_limit = _get_limit("MAX_TENSOR_MEMBER_BYTES", _MAX_MEMBER_BYTES_DEFAULT)
    total_tensor_limit = _get_limit("MAX_TENSOR_BYTES", _MAX_TENSOR_BYTES_DEFAULT)
    members: list[ParsedMember] = []
    seen: set[str] = set()
    n_blocks = len(buf) // USTAR_BLOCK_SIZE
    block_idx = 0
    terminator_seen = False
    total_tensor_bytes = 0
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
        # Item 7: enforce per-member byte limits DURING parsing. manifest.json
        # and state/<role>.json components share the component-member bound;
        # tensors/<n>.bin members use the (larger) tensor-member bound, and a
        # running total tensor bound catches an aggregate overflow.
        is_state = bool(_STATE_MEMBER_RE.fullmatch(name))
        is_tensor = bool(_TENSOR_MEMBER_RE.fullmatch(name))
        if name == "manifest.json" or is_state:
            if size > component_member_limit:
                raise TarParseError(
                    f"member {name!r} size {size} exceeds MAX_COMPONENT_MEMBER_BYTES "
                    f"({component_member_limit})"
                )
        elif is_tensor:
            if size > tensor_member_limit:
                raise TarParseError(
                    f"member {name!r} size {size} exceeds MAX_TENSOR_MEMBER_BYTES "
                    f"({tensor_member_limit})"
                )
            total_tensor_bytes += size
            if total_tensor_bytes > total_tensor_limit:
                raise TarParseError(
                    f"total tensor bytes {total_tensor_bytes} exceed MAX_TENSOR_BYTES "
                    f"({total_tensor_limit})"
                )
        data_end = data_start + size
        if data_end > len(buf):
            raise TarParseError(f"truncated data for member {name!r}")
        data = buf[data_start:data_end]
        # Item 11: validate that data padding bytes are zero. The padding fills
        # the data out to the next 512-byte block boundary.
        data_blocks = (size + USTAR_BLOCK_SIZE - 1) // USTAR_BLOCK_SIZE
        padded_end = data_start + data_blocks * USTAR_BLOCK_SIZE
        padding = buf[data_end:padded_end]
        if any(b != 0 for b in padding):
            raise TarParseError(f"member {name!r} has non-zero padding bytes")
        # Verify the data member name is path-safe.
        _validate_member_name(name)
        members.append(ParsedMember(name=name, data=data, size=size))
        # Item 11: enforce member/tensor count limits during parsing.
        if len(members) > member_limit:
            raise TarParseError(f"member count exceeds max_member_count ({member_limit})")
        # Advance past the data (padded to block boundary).
        block_idx += 1 + data_blocks

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
        raise TarParseError(f"trailing {len(buf) - after} bytes after terminator are rejected")
    # Item 11: enforce the fixed checkpoint member order and tensor count.
    if validate_member_order:
        _validate_checkpoint_member_order(members, tensor_limit)
    return members


def _get_limit(name: str, default: int) -> int:
    """Fetch a resource-limit constant from the models module (lazy import)."""
    try:
        from expertforge.checkpoints import models as _m

        return int(getattr(_m, name))
    except Exception:  # pragma: no cover - defensive
        return default


def parse_ustar_archive_streaming(
    fd: int,
    expected_size: int,
    *,
    max_archive_bytes: int | None = None,
    max_member_count: int | None = None,
    max_tensor_count: int | None = None,
    validate_member_order: bool = True,
) -> list[ParsedMember]:
    """Stream-parse a strict ustar archive from an open descriptor (item 8).

    This is the large-archive load path (Issue #11 item 8): the archive is
    parsed SEQUENTIALLY from ``fd`` so the complete tar byte stream is NEVER
    materialized in Python heap. Each member's header is read, validated, then
    that member's data bytes are read individually (bounded by the per-member
    limits); only the parsed member bytes the caller requests are retained. Peak
    heap during the parse is one member's data plus one header block, not the
    whole archive.

    The framing/numeric/header validation is byte-for-byte identical to
    :func:`parse_ustar_archive`: every check applied to the in-memory form is
    applied here too (checksum, magic/version, typeflag, canonical octal
    fields, member order, byte limits, two-block terminator, no trailing bytes).

    ``expected_size`` is the record-declared archive size; the descriptor is
    fstat-checked to be a regular file of exactly that size before any read, and
    the descriptor MUST be positioned at the start of the archive (the caller
    opens it fresh). Raises :class:`TarParseError` on any non-canonical framing.
    """
    import stat as _stat

    archive_limit = max_archive_bytes or _get_limit("MAX_ARCHIVE_BYTES", _MAX_ARCHIVE_BYTES_DEFAULT)
    if expected_size > archive_limit:
        raise TarParseError(
            f"archive size {expected_size} exceeds max_archive_bytes ({archive_limit})"
        )
    if expected_size <= 0 or expected_size % USTAR_BLOCK_SIZE != 0:
        raise TarParseError(
            f"archive size {expected_size} is not a positive multiple of {USTAR_BLOCK_SIZE}"
        )
    # The caller already fstat-verified regularity + exact size; re-assert here
    # so this function is safe to call directly.
    st = os.fstat(fd)
    if not _stat.S_ISREG(st.st_mode):
        raise TarParseError("streaming parse requires a regular-file descriptor")
    if st.st_size != expected_size:
        raise TarParseError(f"on-disk size {st.st_size} != expected_size {expected_size}")

    member_limit = max_member_count or _get_limit("MAX_MEMBER_COUNT", _MAX_MEMBER_COUNT_DEFAULT)
    tensor_limit = max_tensor_count or _get_limit("MAX_TENSOR_COUNT", _MAX_TENSOR_COUNT_DEFAULT)
    component_member_limit = _get_limit("MAX_COMPONENT_MEMBER_BYTES", _MAX_MEMBER_BYTES_DEFAULT)
    tensor_member_limit = _get_limit("MAX_TENSOR_MEMBER_BYTES", _MAX_TENSOR_BYTES_DEFAULT)
    total_tensor_limit = _get_limit("MAX_TENSOR_BYTES", _MAX_TENSOR_BYTES_DEFAULT)

    total_read = 0

    def _read_exact(n: int, what: str) -> bytes:
        nonlocal total_read
        buf = bytearray()
        remaining = n
        while remaining > 0:
            chunk = os.read(fd, min(BLOCK_READ_CHUNK, remaining))
            if not chunk:
                raise TarParseError(f"unexpected EOF reading {what}: got {len(buf)} of {n} bytes")
            buf.extend(chunk)
            remaining -= len(chunk)
        total_read += n
        return bytes(buf)

    members: list[ParsedMember] = []
    seen: set[str] = set()
    terminator_seen = False
    total_tensor_bytes = 0
    n_blocks = expected_size // USTAR_BLOCK_SIZE

    # Parse member-by-member. We cannot seek on a pipe, but the descriptor is a
    # regular file; we still read strictly sequentially (no seeks) so the whole
    # archive is never buffered.
    blocks_consumed = 0
    # We need header bytes for _parse_header, which expects the WHOLE buffer to
    # bounds-check member data offsets. We give it a 1-block slice so it can only
    # see the header (data is read separately below). _parse_header computes the
    # data_start as offset + USTAR_BLOCK_SIZE but we ignore that and read data
    # ourselves.
    while blocks_consumed < n_blocks:
        header_block = _read_exact(USTAR_BLOCK_SIZE, "header block")
        blocks_consumed += 1
        # A zero block signals the end of members.
        if header_block == b"\x00" * USTAR_BLOCK_SIZE:
            terminator_seen = True
            # Must be followed by exactly one more zero block, then EOF.
            break
        # _parse_header validates the header (checksum, magic, typeflag, numeric
        # fields, name/prefix, devmajor/devminor, linkname, reserved). We pass a
        # 1-block buffer; the data-start it returns is irrelevant here.
        name, _typeflag, size, _data_start = _parse_header(header_block, 0)
        if name in seen:
            raise TarParseError(f"duplicate member name {name!r}")
        seen.add(name)
        if size < 0:
            raise TarParseError(f"negative size for member {name!r}")
        # Enforce per-member byte limits DURING parsing (same as the in-memory
        # parser).
        is_state = bool(_STATE_MEMBER_RE.fullmatch(name))
        is_tensor = bool(_TENSOR_MEMBER_RE.fullmatch(name))
        if name == "manifest.json" or is_state:
            if size > component_member_limit:
                raise TarParseError(
                    f"member {name!r} size {size} exceeds MAX_COMPONENT_MEMBER_BYTES "
                    f"({component_member_limit})"
                )
        elif is_tensor:
            if size > tensor_member_limit:
                raise TarParseError(
                    f"member {name!r} size {size} exceeds MAX_TENSOR_MEMBER_BYTES "
                    f"({tensor_member_limit})"
                )
            total_tensor_bytes += size
            if total_tensor_bytes > total_tensor_limit:
                raise TarParseError(
                    f"total tensor bytes {total_tensor_bytes} exceed MAX_TENSOR_BYTES "
                    f"({total_tensor_limit})"
                )
        # Read the data. For large tensor members, spill to a temp file to
        # keep peak heap bounded (review 4833143258 item 2).
        _SPILL_THRESHOLD = 4 * 1024 * 1024  # 4 MiB
        temp_path: Path | None = None
        if is_tensor and size > _SPILL_THRESHOLD:
            import tempfile

            fd_tmp, temp_name = tempfile.mkstemp(suffix=".tensor")
            try:
                remaining = size
                while remaining > 0:
                    chunk_sz = min(remaining, 65536)
                    chunk = _read_exact(chunk_sz, f"data for {name!r}")
                    os.write(fd_tmp, chunk)
                    remaining -= chunk_sz
                os.fsync(fd_tmp)
            finally:
                os.close(fd_tmp)
            data = None
            temp_path = Path(temp_name)
        else:
            data = _read_exact(size, f"data for {name!r}") if size else b""
        data_blocks = (size + USTAR_BLOCK_SIZE - 1) // USTAR_BLOCK_SIZE
        padding_len = data_blocks * USTAR_BLOCK_SIZE - size
        if padding_len:
            padding = _read_exact(padding_len, f"padding for {name!r}")
            if any(b != 0 for b in padding):
                raise TarParseError(f"member {name!r} has non-zero padding bytes")
        blocks_consumed += data_blocks
        _validate_member_name(name)
        members.append(ParsedMember(name=name, data=data, size=size, temp_path=temp_path))
        if len(members) > member_limit:
            raise TarParseError(f"member count exceeds max_member_count ({member_limit})")

    if not terminator_seen:
        raise TarParseError("missing two-block zero terminator")
    # The terminator must be exactly two zero blocks, then EOF.
    if blocks_consumed >= n_blocks:
        raise TarParseError("truncated terminator: expected two zero blocks")
    second = _read_exact(USTAR_BLOCK_SIZE, "second terminator block")
    blocks_consumed += 1
    if second != b"\x00" * USTAR_BLOCK_SIZE:
        raise TarParseError("second terminator block is non-zero")
    if total_read != expected_size:
        raise TarParseError(f"streamed {total_read} bytes != expected_size {expected_size}")
    # Any trailing bytes beyond the two terminator blocks are corruption.
    extra = os.read(fd, 1)
    if extra:
        raise TarParseError("trailing bytes after terminator are rejected")
    if validate_member_order:
        _validate_checkpoint_member_order(members, tensor_limit)
    return members


# Streaming reads use this chunk size (bounds peak heap to one chunk + header).
BLOCK_READ_CHUNK: int = 64 * 1024


def _validate_checkpoint_member_order(members: list[ParsedMember], tensor_limit: int) -> None:
    """Enforce the EXACT fixed checkpoint member order (item 7/11).

    The ratified canonical order is byte-canonical: ``manifest.json`` first,
    then the ``state/<role>.json`` components in the EXACT fixed role order
    (identity, configuration, provenance, rng, data_cursor, counters, model,
    optimizer, scheduler, scaler — scaler optional, all others required), then
    all ``tensors/<n>.bin`` members in contiguous ascending index order. No other
    member names are permitted and roles may not be reordered. Tensor indices
    must be contiguous from 0 and the count must not exceed ``tensor_limit``.
    """
    if not members:
        return
    # The exact fixed role order (amendment B / item 7). scaler is optional and
    # last; every other role is required and must appear in this exact sequence.
    _REQUIRED_ROLE_ORDER: tuple[str, ...] = (
        "identity",
        "configuration",
        "provenance",
        "rng",
        "data_cursor",
        "counters",
        "model",
        "optimizer",
        "scheduler",
    )
    _OPTIONAL_TRAILING_ROLE = "scaler"
    phases: list[list[str]] = [[], [], []]
    for m in members:
        if m.name == "manifest.json":
            phases[0].append(m.name)
        elif _STATE_MEMBER_RE.fullmatch(m.name):
            phases[1].append(m.name)
        elif _TENSOR_MEMBER_RE.fullmatch(m.name):
            phases[2].append(m.name)
        else:
            raise TarParseError(f"member {m.name!r} is not a recognized checkpoint component")
    # manifest.json exactly once, first.
    if phases[0] != ["manifest.json"]:
        raise TarParseError("manifest.json must be present exactly once as the first member")
    # Item 7: the state components must appear in the EXACT fixed role order.
    state_names = phases[1]
    expected_state: list[str] = [f"state/{role}.json" for role in _REQUIRED_ROLE_ORDER]
    if state_names[: len(expected_state)] != expected_state:
        raise TarParseError(
            "state components must appear in the exact fixed order "
            "(identity, configuration, provenance, rng, data_cursor, counters, "
            "model, optimizer, scheduler); got "
            f"{state_names!r}"
        )
    # scaler, if present, must come immediately after scheduler and at most once.
    remainder = state_names[len(expected_state) :]
    if remainder:
        if remainder != [f"state/{_OPTIONAL_TRAILING_ROLE}.json"]:
            raise TarParseError(
                f"unexpected state components after the fixed order: {remainder!r} "
                "(only state/scaler.json may trail, exactly once)"
            )
    # No state component may appear after the first tensor (fixed phase order).
    seen_tensor = False
    for m in members[1:]:
        is_state = bool(_STATE_MEMBER_RE.fullmatch(m.name))
        is_tensor = bool(_TENSOR_MEMBER_RE.fullmatch(m.name))
        if is_tensor:
            seen_tensor = True
        if is_state and seen_tensor:
            raise TarParseError(
                f"state component {m.name!r} appears after a tensor member "
                "(fixed order: manifest → state/* → tensors/*)"
            )
    # Tensor indices contiguous from 0, ascending.
    tensor_names = phases[2]
    if len(tensor_names) > tensor_limit:
        raise TarParseError(f"tensor member count exceeds max_tensor_count ({tensor_limit})")
    for i, n in enumerate(tensor_names):
        expected = f"tensors/{i}.bin"
        if n != expected:
            raise TarParseError(
                f"tensor members must be contiguous from 0; expected {expected!r} "
                f"at position {i}, got {n!r}"
            )


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
        raise TarParseError(f"member name {name!r} contains '.', '..', or empty components")


def iter_members(buf: bytes) -> Iterator[ParsedMember]:
    """Iterate parsed members (convenience wrapper around parse)."""
    yield from parse_ustar_archive(buf)
