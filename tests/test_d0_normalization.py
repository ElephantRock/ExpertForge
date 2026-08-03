from __future__ import annotations

import pytest

from expertforge.d0.errors import NormalizationError
from expertforge.d0.normalization import (
    normalize_text,
    normalize_utf8_bytes,
    normalized_text_sha256,
)


def test_normalization_maps_only_newline_forms() -> None:
    text = " alpha\r\nbeta\rgamma\n delta\t "
    assert normalize_text(text) == " alpha\nbeta\ngamma\n delta\t "


def test_normalization_rejects_nul() -> None:
    with pytest.raises(NormalizationError, match="NUL"):
        normalize_text("alpha\x00beta")


def test_normalization_rejects_invalid_utf8_bytes() -> None:
    with pytest.raises(NormalizationError, match="valid UTF-8"):
        normalize_utf8_bytes(b"alpha\xffbeta")


def test_normalization_rejects_lone_surrogate() -> None:
    with pytest.raises(NormalizationError, match="strict UTF-8"):
        normalize_text("alpha\ud800beta")


def test_duplicate_digest_is_over_normalized_utf8() -> None:
    assert normalized_text_sha256("alpha\r\nbeta") == normalized_text_sha256("alpha\nbeta")
