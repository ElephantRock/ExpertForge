"""Exact D0 document and prompt normalization."""

from __future__ import annotations

import hashlib

from expertforge.d0.errors import NormalizationError


def normalize_text(text: str) -> str:
    """Apply the frozen D0 newline mapping and preserve every other code point.

    Python strings can contain lone surrogate code points even though they are
    not valid UTF-8. A strict encode therefore remains necessary even when the
    caller already has a ``str`` rather than raw bytes.
    """

    if "\x00" in text:
        raise NormalizationError("text contains a forbidden NUL code point")
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise NormalizationError("text is not representable as strict UTF-8") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_utf8_bytes(payload: bytes) -> str:
    """Decode strict UTF-8 bytes and apply :func:`normalize_text`."""

    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise NormalizationError("document bytes are not valid UTF-8") from exc
    return normalize_text(text)


def normalized_utf8_bytes(text: str) -> bytes:
    """Return exact UTF-8 bytes after D0 normalization."""

    return normalize_text(text).encode("utf-8")


def normalized_text_sha256(text: str) -> str:
    """Return ``sha256(normalized_utf8_text)`` for duplicate identity."""

    return hashlib.sha256(normalized_utf8_bytes(text)).hexdigest()
