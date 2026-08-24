"""Golden-vector tests for the corrected D0 tokenizer (integration tier).

These tests bind the corrected tokenizer (Decision 0012 Resolution 1) against a
local verified cache and assert literal encode/decode vectors. They require the
``d0-data`` extra (the ``tokenizers`` library) and the derived
``tokenizer.json`` to be present in ``.cache/d0-tokenizer-corrected/``. They
skip with an explicit reason otherwise.

Provenance of the literal vectors below:
- manifest: tokenizers/manifests/d0-gpt-neox-corrected-v1.json
  manifest_sha256 = 1142baf61c330a318be6e1b153f6769892a7f5af3b7241363a1d0a1f2ca453e1
- tokenizer.json sha256 = c1e2f38b28b816e3af5a7047982b694c657222e4eb4bed647540801385770362
- tokenizers == 0.22.2
- add_special_tokens=False; document form appends boundary id 0 manually
- derived via scripts/derive_d0_tokenizer.py from the rejected upstream
  GPT-NeoX tokenizer.json (sha256 c24618a1...)

The vectors are NOT equivalent to the upstream tokenizer: the dropped high-id
whitespace-run tokens (50257..50276) now fall back to byte-level BPE, e.g.
``"  hello"`` encodes to ``[209, 23120]`` instead of the rejected
``[50276, 25521]``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from expertforge.d0.data.tokenizer import D0Tokenizer, bind_tokenizer

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tokenizers" / "manifests" / "d0-gpt-neox-corrected-v1.json"
CACHE_ROOT = ROOT / ".cache" / "d0-tokenizer-corrected"


def _corrected_cache_present() -> bool:
    return (CACHE_ROOT / "tokenizer.json").is_file()


def _tokenizers_available() -> bool:
    try:
        import tokenizers  # noqa: F401
    except ModuleNotFoundError:
        return False
    import importlib

    module = importlib.import_module("tokenizers")
    return getattr(module, "__file__", None) is not None and hasattr(module, "Tokenizer")


pytestmark = pytest.mark.integration
_skip_reason = (
    "corrected D0 tokenizer cache (.cache/d0-tokenizer-corrected/tokenizer.json) "
    "or the d0-data extra (tokenizers) is not available; regenerate via "
    "scripts/derive_d0_tokenizer.py and `uv sync --extra d0-data`"
)


@pytest.fixture(scope="module")
def bound_tokenizer() -> D0Tokenizer:
    if not (_corrected_cache_present() and _tokenizers_available()):
        pytest.skip(_skip_reason)
    return bind_tokenizer(manifest_path=MANIFEST_PATH, cache_root=CACHE_ROOT)


# ---------------------------------------------------------------------------
# Frozen artifact identity (run only when the cache is present)
# ---------------------------------------------------------------------------


class TestBoundArtifactIdentity:
    def test_vocab_size_is_50257(self, bound_tokenizer: D0Tokenizer) -> None:
        assert bound_tokenizer.vocab_size == 50257

    def test_boundary_token_is_zero(self, bound_tokenizer: D0Tokenizer) -> None:
        assert bound_tokenizer.document_boundary_token_id == 0

    def test_padding_token_is_none(self, bound_tokenizer: D0Tokenizer) -> None:
        assert bound_tokenizer.padding_token_id is None

    def test_manifest_sha256_matches_corrected_artifact(self, bound_tokenizer: D0Tokenizer) -> None:
        assert (
            bound_tokenizer.manifest_sha256
            == "1142baf61c330a318be6e1b153f6769892a7f5af3b7241363a1d0a1f2ca453e1"
        )


# ---------------------------------------------------------------------------
# Literal encode vectors: encode_no_special
# ---------------------------------------------------------------------------

# Each tuple is (input_text, expected_ids). Computed once via
# Tokenizer.from_buffer over the verified corrected bytes; see file docstring.
ENCODE_NO_SPECIAL_VECTORS: list[tuple[str, tuple[int, ...]]] = [
    ("", ()),
    ("hello", (25521,)),
    (" hello", (23120,)),
    ("  hello", (209, 23120)),
    ("   hello", (245, 23120)),
    ("hello\nworld", (25521, 187, 10186)),
    ("<|endoftext|>", (0,)),
    ("café", (68, 2320, 860)),
    ("مرحبا", (5843, 6900, 21931, 13621, 3142)),
    ("你好", (24553, 34439)),
    ("🙂", (14931, 18713)),
]


class TestEncodeNoSpecialGoldenVectors:
    @pytest.mark.parametrize("text,expected", ENCODE_NO_SPECIAL_VECTORS)
    def test_encode_no_special(
        self, bound_tokenizer: D0Tokenizer, text: str, expected: tuple[int, ...]
    ) -> None:
        assert bound_tokenizer.encode_no_special(text) == expected

    def test_all_no_special_ids_in_range(self, bound_tokenizer: D0Tokenizer) -> None:
        for text, ids in ENCODE_NO_SPECIAL_VECTORS:
            for token_id in ids:
                assert 0 <= token_id <= 50256, f"{text!r} produced id {token_id}"


# ---------------------------------------------------------------------------
# Literal encode vectors: encode_document (appends boundary id 0)
# ---------------------------------------------------------------------------

ENCODE_DOCUMENT_VECTORS: list[tuple[str, tuple[int, ...]]] = [
    (text, ids + (0,)) for text, ids in ENCODE_NO_SPECIAL_VECTORS
]


class TestEncodeDocumentGoldenVectors:
    @pytest.mark.parametrize("text,expected", ENCODE_DOCUMENT_VECTORS)
    def test_encode_document(
        self, bound_tokenizer: D0Tokenizer, text: str, expected: tuple[int, ...]
    ) -> None:
        encoded = bound_tokenizer.encode_document(text)
        assert encoded.ids == expected
        assert encoded.n_documents == 1

    def test_endoftext_source_appends_second_zero(self, bound_tokenizer: D0Tokenizer) -> None:
        """Source text '<|endoftext|>' encodes to id 0; the document form
        appends a second 0. This pins that the boundary is always appended
        regardless of source content."""
        encoded = bound_tokenizer.encode_document("<|endoftext|>")
        assert encoded.ids == (0, 0)


# ---------------------------------------------------------------------------
# Literal decode vectors
# ---------------------------------------------------------------------------


class TestDecodeGoldenVectors:
    def test_decode_round_trips_no_special(self, bound_tokenizer: D0Tokenizer) -> None:
        # Ordinary text round-trips through encode_no_special -> decode.
        # Excludes "<|endoftext|>" because decode uses skip_special_tokens=True,
        # so special-token text does NOT round-trip (see test below).
        for text, ids in ENCODE_NO_SPECIAL_VECTORS:
            if text == "<|endoftext|>":
                continue
            assert bound_tokenizer.decode(ids) == text, f"round-trip failed for {text!r}"

    def test_decode_document_skips_boundary(self, bound_tokenizer: D0Tokenizer) -> None:
        # The document form appends a trailing boundary id 0, which decode
        # skips, so ordinary text still round-trips through the document form.
        for text, ids in ENCODE_DOCUMENT_VECTORS:
            if text == "<|endoftext|>":
                continue
            assert bound_tokenizer.decode(ids) == text, f"document decode failed for {text!r}"

    def test_special_token_text_does_not_round_trip(self, bound_tokenizer: D0Tokenizer) -> None:
        """'<|endoftext|>' encodes to id 0, but decode(skip_special_tokens=True)
        skips id 0, so the literal text does NOT round-trip — it decodes to ''.
        This is the deliberate D0 decode contract (specials are skipped)."""
        assert bound_tokenizer.decode(bound_tokenizer.encode_no_special("<|endoftext|>")) == ""

    def test_decode_single_boundary_is_empty(self, bound_tokenizer: D0Tokenizer) -> None:
        assert bound_tokenizer.decode((0,)) == ""

    def test_decode_double_boundary_is_empty(self, bound_tokenizer: D0Tokenizer) -> None:
        assert bound_tokenizer.decode((0, 0)) == ""

    def test_decode_endoftext_source_document(self, bound_tokenizer: D0Tokenizer) -> None:
        """'<|endoftext|>' document form is (0, 0); decode skips both specials -> ''."""
        assert bound_tokenizer.decode((0, 0)) == ""


# ---------------------------------------------------------------------------
# Provable upper bound: no reachable id exceeds 50256
# ---------------------------------------------------------------------------


class TestNoReachableIdExceedsMax:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "hello",
            " hello",
            "  hello",
            "   hello",
            "    hello",
            "     hello",
            "hello\nworld",
            "café",
            "مرحبا",
            "你好",
            "🙂",
            "a" + " " * 30 + "b",
            "          ",
            "\t\t",
            " " * 22,  # longest kept space-run token (id 50256)
            " " * 23,  # just past the longest kept token -> byte fallback
            " " * 24,
            " " * 25,
            " " * 30,
        ],
    )
    def test_id_in_range(self, bound_tokenizer: D0Tokenizer, text: str) -> None:
        ids = bound_tokenizer.encode_no_special(text)
        for token_id in ids:
            assert 0 <= token_id <= 50256, f"{text!r} produced id {token_id}"

    def test_max_reachable_id_is_50256(self, bound_tokenizer: D0Tokenizer) -> None:
        """The 22-space run (the longest kept space-run token) is id 50256,
        which is the maximum reachable id under the corrected artifact."""
        assert bound_tokenizer.encode_no_special(" " * 22) == (50256,)
