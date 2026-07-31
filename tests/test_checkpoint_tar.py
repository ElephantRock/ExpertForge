"""Regression tests for the strict ustar tar writer and reader (Issue #11, F)."""

from __future__ import annotations

import pytest

from expertforge.checkpoints.tar_reader import (
    ParsedMember,
    TarParseError,
    parse_ustar_archive,
)
from expertforge.checkpoints.tar_writer import (
    USTAR_BLOCK_SIZE,
    TarMember,
    TarWriterError,
    build_ustar_archive,
)


def _build(members: list[tuple[str, bytes]]) -> bytes:
    return build_ustar_archive([TarMember(n, d) for n, d in members])


def _recompute_checksum(buf: bytearray, header_idx: int = 0) -> None:
    """Recompute and write the header checksum of the block at ``header_idx*512``.

    Used by corruption tests that mutate a header field then need a valid
    checksum so the parser reaches the targeted field-specific check.
    """
    base = header_idx * USTAR_BLOCK_SIZE
    header = bytearray(buf[base : base + USTAR_BLOCK_SIZE])
    header[148:156] = b"        "
    chk = sum(header) & 0o777777
    buf[base + 148 : base + 156] = f"{chk:06o}\x00 ".encode("ascii")


class TestTarWriter:
    def test_empty_archive_has_two_zero_blocks(self) -> None:
        out = _build([])
        assert len(out) == USTAR_BLOCK_SIZE * 2
        assert out == b"\x00" * (USTAR_BLOCK_SIZE * 2)

    def test_total_size_multiple_of_512(self) -> None:
        out = _build([("manifest.json", b'{"a":1}')])
        assert len(out) % USTAR_BLOCK_SIZE == 0

    def test_member_order_preserved(self) -> None:
        out = _build([("a.bin", b"1"), ("b.bin", b"22"), ("c.bin", b"333")])
        # Generic framing test: disable the checkpoint-specific member-order gate.
        members = parse_ustar_archive(out, validate_member_order=False)
        assert [m.name for m in members] == ["a.bin", "b.bin", "c.bin"]
        assert [m.data for m in members] == [b"1", b"22", b"333"]

    def test_deterministic_same_input_same_bytes(self) -> None:
        members = [("manifest.json", b'{"x":1}'), ("tensors/0.bin", b"\x00" * 10)]
        a = _build(members)
        b = _build(members)
        assert a == b

    def test_header_fields_deterministic(self) -> None:
        out = _build([("manifest.json", b"hi")])
        header = out[:USTAR_BLOCK_SIZE]
        # magic ustar\0 at 257
        assert header[257:263] == b"ustar\x00"
        # version 00 at 263
        assert header[263:265] == b"00"
        # mode 0o644 at 100: 0000644\0
        assert header[100:108] == b"0000644\x00"
        # uid/gid 0
        assert header[108:116] == b"0000000\x00"
        assert header[116:124] == b"0000000\x00"
        # size 2: 00000000002\0
        assert header[124:136] == b"00000000002\x00"
        # mtime 0
        assert header[136:148] == b"00000000000\x00"
        # typeflag '0'
        assert header[156:157] == b"0"
        # uname/gname empty
        assert header[265:297] == b"\x00" * 32
        assert header[297:329] == b"\x00" * 32

    def test_checksum_is_correct(self) -> None:
        out = _build([("manifest.json", b"hi")])
        header = bytearray(out[:USTAR_BLOCK_SIZE])
        stored = header[148:156]
        header[148:156] = b"        "
        expected = sum(header) & 0o777777
        stored_text = stored.rstrip(b"\x00 ").lstrip(b" ").decode("ascii")
        assert int(stored_text, 8) == expected

    def test_rejects_absolute_name(self) -> None:
        with pytest.raises(TarWriterError):
            _build([("/etc/passwd", b"x")])

    def test_rejects_traversal_name(self) -> None:
        with pytest.raises(TarWriterError):
            _build([("../escape", b"x")])

    def test_rejects_backslash_name(self) -> None:
        with pytest.raises(TarWriterError):
            _build([("dir\\file", b"x")])

    def test_rejects_duplicate_members(self) -> None:
        with pytest.raises(TarWriterError):
            _build([("a", b"1"), ("a", b"2")])

    def test_padding_alignment(self) -> None:
        # 1 byte of data -> padded to 512 in the data section.
        out = _build([("a.bin", b"x")])
        # header (512) + data block (512) + terminator (1024) = 2048
        assert len(out) == 2048


class TestTarReader:
    def test_round_trip(self) -> None:
        out = _build([("manifest.json", b'{"k":1}'), ("state/x.json", b'{"y":2}')])
        # Generic framing round-trip: the checkpoint member-order gate is strict
        # (it requires the full fixed role order), so disable it here.
        members = parse_ustar_archive(out, validate_member_order=False)
        assert [m.name for m in members] == ["manifest.json", "state/x.json"]
        assert members[0].data == b'{"k":1}'
        assert members[1].data == b'{"y":2}'

    def test_empty_archive_parses(self) -> None:
        members = parse_ustar_archive(_build([]))
        assert members == []

    def test_rejects_truncated_archive(self) -> None:
        out = _build([("a.bin", b"x" * 100)])
        truncated = out[:-1]
        with pytest.raises(TarParseError):
            parse_ustar_archive(truncated)

    def test_rejects_bad_checksum(self) -> None:
        out = bytearray(_build([("a.bin", b"x")]))
        # Corrupt one byte in the header (not the checksum field).
        out[0] = ord("Z") if out[0] != ord("Z") else ord("Y")
        with pytest.raises(TarParseError, match="checksum"):
            parse_ustar_archive(bytes(out))

    def test_rejects_bad_magic(self) -> None:
        out = bytearray(_build([("a.bin", b"x")]))
        out[257:263] = b"ustar "  # wrong magic
        _recompute_checksum(out)
        with pytest.raises(TarParseError, match="magic"):
            parse_ustar_archive(bytes(out))

    def test_rejects_base256_size(self) -> None:
        out = bytearray(_build([("a.bin", b"x")]))
        # Set high bit on size field's first byte (base-256 marker).
        out[124] = out[124] | 0x80
        _recompute_checksum(out)
        with pytest.raises(TarParseError, match="base-256"):
            parse_ustar_archive(bytes(out))

    def test_rejects_non_octal_in_numeric(self) -> None:
        out = bytearray(_build([("a.bin", b"x")]))
        # Put an '8' in the size field (invalid octal).
        out[124] = ord("8")
        _recompute_checksum(out)
        with pytest.raises(TarParseError, match="non-octal"):
            parse_ustar_archive(bytes(out))

    def test_rejects_pax_extended_header(self) -> None:
        out = bytearray(_build([("a.bin", b"x")]))
        # Set typeflag to 'x' (PAX local extended header).
        out[156] = ord("x")
        # Recompute the checksum so we reach the typeflag check.
        out[148:156] = b"        "
        chk = sum(out[:USTAR_BLOCK_SIZE]) & 0o777777
        out[148:156] = f"{chk:06o}\x00 ".encode("ascii")
        with pytest.raises(TarParseError, match="extended header"):
            parse_ustar_archive(bytes(out))

    def test_rejects_symlink_typeflag(self) -> None:
        out = bytearray(_build([("a.bin", b"x")]))
        out[156] = ord("2")  # symlink
        out[148:156] = b"        "
        chk = sum(out[:USTAR_BLOCK_SIZE]) & 0o777777
        out[148:156] = f"{chk:06o}\x00 ".encode("ascii")
        with pytest.raises(TarParseError, match="unsupported typeflag"):
            parse_ustar_archive(bytes(out))

    def test_rejects_nonzero_mtime(self) -> None:
        out = bytearray(_build([("a.bin", b"x")]))
        out[136:148] = b"00000000001\x00"
        # Recompute checksum.
        out[148:156] = b"        "
        chk = sum(out[:USTAR_BLOCK_SIZE]) & 0o777777
        out[148:156] = f"{chk:06o}\x00 ".encode("ascii")
        with pytest.raises(TarParseError, match="mtime"):
            parse_ustar_archive(bytes(out))

    def test_rejects_duplicate_members(self) -> None:
        archive = _build([("a.bin", b"x"), ("b.bin", b"y")])
        # Overwrite b's header name with a's name to create a duplicate.
        # Locate the second header (first header 512 + data padding 512 = 1024).
        out = bytearray(archive)
        # Replace 'b.bin' (5 chars) with 'a.bin' at offset 1024.
        name_field = bytearray(b"a.bin") + b"\x00" * (100 - 5)
        out[USTAR_BLOCK_SIZE * 2 : USTAR_BLOCK_SIZE * 2 + 100] = name_field
        # Recompute checksum for the second header.
        second = bytearray(out[USTAR_BLOCK_SIZE * 2 : USTAR_BLOCK_SIZE * 3])
        second[148:156] = b"        "
        chk = sum(second) & 0o777777
        out[USTAR_BLOCK_SIZE * 2 + 148 : USTAR_BLOCK_SIZE * 2 + 156] = f"{chk:06o}\x00 ".encode(
            "ascii"
        )
        with pytest.raises(TarParseError, match="duplicate"):
            parse_ustar_archive(bytes(out))

    def test_rejects_trailing_bytes_after_terminator(self) -> None:
        out = bytearray(_build([("a.bin", b"x")]))
        out.extend(b"\x00" * 512)  # extra block
        with pytest.raises(TarParseError, match="trailing"):
            parse_ustar_archive(bytes(out))

    def test_rejects_missing_terminator(self) -> None:
        # Build then strip the terminator entirely.
        out = bytearray(_build([("a.bin", b"x")]))
        truncated = bytes(out[: -USTAR_BLOCK_SIZE * 2])
        with pytest.raises(TarParseError):
            parse_ustar_archive(truncated)

    def test_rejects_non_block_aligned_size(self) -> None:
        # An archive whose length isn't a multiple of 512.
        out = _build([("a.bin", b"x")])
        with pytest.raises(TarParseError, match="multiple"):
            parse_ustar_archive(out + b"\x01")

    def test_large_member_uses_multiple_data_blocks(self) -> None:
        data = b"Z" * (USTAR_BLOCK_SIZE * 3 + 7)
        out = _build([("big.bin", data)])
        # Generic framing test: disable the checkpoint-specific member-order gate.
        members = parse_ustar_archive(out, validate_member_order=False)
        assert members[0].data == data
        assert members[0].size == len(data)


def test_parsed_member_repr() -> None:
    m = ParsedMember(name="a", data=b"x", size=1)
    assert "a" in repr(m)


def test_size_field_octal_width() -> None:
    # The 12-byte size field encodes up to 11 octal digits + a NUL terminator.
    # A size near the field max must encode without overflow; a size beyond the
    # 11-octal-digit max must raise rather than truncate.
    from expertforge.checkpoints.tar_writer import _octal_field

    # 11 octal digits of all-7s is the max representable value.
    max_size = 0o77_777_777_777
    field = _octal_field(max_size, 12)
    assert field == b"77777777777\x00"
    assert field.endswith(b"\x00")
    with pytest.raises(TarWriterError):
        _octal_field(max_size + 1, 12)
