"""Contract tests for the D0 tokenizer binding (fast tier, no optional dep).

These run on the fast tier without the ``tokenizers`` library installed because
they test contract parsing, frozen constants, normalization enforcement, and
the namespace-shadow defense — not the loaded tokenizer. Encode/decode golden
vectors live in ``tests/test_d0_tokenizer_golden.py`` (integration tier).
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from expertforge.d0.data.tokenizer import (
    D0_DOCUMENT_BOUNDARY_TOKEN_ID,
    D0_MAX_REACHABLE_TOKEN_ID,
    D0_PADDING_TOKEN_ID,
    D0_TOKENIZER_MANIFEST_SHA256,
    D0_TOKENIZER_VOCAB_SIZE,
    D0TokenizerContract,
    PreTokenizerSpec,
    TokenizerBindingError,
    TokenizerSpecialToken,
    parse_tokenizer_contract,
)
from expertforge.d0.errors import MissingOptionalDependencyError

ROOT = Path(__file__).resolve().parents[1]
CORRECTED_MANIFEST_PATH = ROOT / "tokenizers" / "manifests" / "d0-gpt-neox-corrected-v1.json"


def _corrected_tokenizer_block() -> Mapping[str, Any]:
    """The literal ``tokenizer`` block from the committed corrected manifest."""

    manifest = json.loads(CORRECTED_MANIFEST_PATH.read_text(encoding="utf-8"))
    return cast(Mapping[str, Any], manifest["tokenizer"])


# ---------------------------------------------------------------------------
# Frozen identity constants
# ---------------------------------------------------------------------------


class TestFrozenConstants:
    def test_corrected_manifest_digest_matches_committed_file(self) -> None:
        """The binding's frozen manifest digest must track the committed manifest."""
        manifest = json.loads(CORRECTED_MANIFEST_PATH.read_text(encoding="utf-8"))
        assert manifest["manifest_sha256"] == D0_TOKENIZER_MANIFEST_SHA256

    def test_vocab_size(self) -> None:
        assert D0_TOKENIZER_VOCAB_SIZE == 50257

    def test_max_reachable_id(self) -> None:
        assert D0_MAX_REACHABLE_TOKEN_ID == 50256

    def test_document_boundary_id(self) -> None:
        assert D0_DOCUMENT_BOUNDARY_TOKEN_ID == 0

    def test_padding_token_is_none(self) -> None:
        assert D0_PADDING_TOKEN_ID is None


# ---------------------------------------------------------------------------
# parse_tokenizer_contract — happy path against the committed manifest
# ---------------------------------------------------------------------------


class TestParseContractHappyPath:
    def test_parses_committed_manifest_tokenizer_block(self) -> None:
        contract = parse_tokenizer_contract(_corrected_tokenizer_block())
        assert isinstance(contract, D0TokenizerContract)
        assert contract.family == "byte_level_bpe"
        assert contract.vocabulary_size == 50257
        assert contract.normalizer == "none"
        assert contract.pre_tokenizer == PreTokenizerSpec(type="ByteLevel", add_prefix_space=False)
        assert contract.padding_token_id is None
        assert contract.training_document_boundary_token_id == 0

    def test_special_token_contract(self) -> None:
        contract = parse_tokenizer_contract(_corrected_tokenizer_block())
        assert len(contract.special_tokens) == 1
        only = contract.special_tokens[0]
        assert only == TokenizerSpecialToken(
            token="<|endoftext|>",
            token_id=0,
            roles=("bos", "document_boundary", "eos", "unk"),
        )


# ---------------------------------------------------------------------------
# parse_tokenizer_contract — rejection cases
# ---------------------------------------------------------------------------


def _base_block() -> dict[str, Any]:
    """A minimal valid contract block, mutated per-test."""

    return {
        "family": "byte_level_bpe",
        "vocabulary_size": 50257,
        "normalizer": "none",
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
        "special_tokens": [
            {
                "token": "<|endoftext|>",
                "token_id": 0,
                "roles": ["bos", "document_boundary", "eos", "unk"],
            }
        ],
        "padding_token": None,
        "training_document_boundary_token_id": 0,
    }


class TestParseContractRejections:
    def test_rejects_non_mapping(self) -> None:
        with pytest.raises(TokenizerBindingError, match="must be a JSON object"):
            parse_tokenizer_contract([])  # type: ignore[arg-type]

    def test_rejects_extra_field(self) -> None:
        block = _base_block()
        block["unexpected"] = True
        with pytest.raises(TokenizerBindingError, match="fields changed"):
            parse_tokenizer_contract(block)

    def test_rejects_missing_field(self) -> None:
        block = _base_block()
        del block["normalizer"]
        with pytest.raises(TokenizerBindingError, match="fields changed"):
            parse_tokenizer_contract(block)

    def test_rejects_bool_vocabulary_size(self) -> None:
        block = _base_block()
        block["vocabulary_size"] = True
        with pytest.raises(TokenizerBindingError, match="vocabulary_size must be an integer"):
            parse_tokenizer_contract(block)

    def test_rejects_bool_token_id(self) -> None:
        block = _base_block()
        block["special_tokens"][0]["token_id"] = True
        with pytest.raises(TokenizerBindingError, match="token_id must be an integer"):
            parse_tokenizer_contract(block)

    def test_rejects_more_than_one_special_token(self) -> None:
        block = _base_block()
        block["special_tokens"].append({"token": "<|extra|>", "token_id": 5, "roles": ["bos"]})
        with pytest.raises(TokenizerBindingError, match="exactly one special token"):
            parse_tokenizer_contract(block)

    def test_rejects_zero_special_tokens(self) -> None:
        block = _base_block()
        block["special_tokens"] = []
        with pytest.raises(TokenizerBindingError, match="exactly one special token"):
            parse_tokenizer_contract(block)

    def test_rejects_wrong_special_token_string(self) -> None:
        block = _base_block()
        block["special_tokens"][0]["token"] = "<|wrong|>"
        with pytest.raises(TokenizerBindingError, match="special token string changed"):
            parse_tokenizer_contract(block)

    def test_rejects_wrong_special_token_id(self) -> None:
        block = _base_block()
        block["special_tokens"][0]["token_id"] = 7
        with pytest.raises(TokenizerBindingError, match="special token id changed"):
            parse_tokenizer_contract(block)

    def test_rejects_missing_required_role(self) -> None:
        for missing in ("bos", "document_boundary", "eos", "unk"):
            block = _base_block()
            block["special_tokens"][0]["roles"] = [
                r for r in ("bos", "document_boundary", "eos", "unk") if r != missing
            ]
            with pytest.raises(TokenizerBindingError, match="special token roles changed"):
                parse_tokenizer_contract(block)

    def test_rejects_duplicate_roles(self) -> None:
        block = _base_block()
        block["special_tokens"][0]["roles"] = ["bos", "bos", "eos", "unk"]
        with pytest.raises(TokenizerBindingError, match="must be unique"):
            parse_tokenizer_contract(block)

    def test_rejects_pre_tokenizer_wrong_type(self) -> None:
        block = _base_block()
        block["pre_tokenizer"]["add_prefix_space"] = "false"
        with pytest.raises(TokenizerBindingError, match="add_prefix_space must be a boolean"):
            parse_tokenizer_contract(block)

    def test_rejects_padding_token_int(self) -> None:
        """parse_tokenizer_contract accepts any int-or-None for padding_token
        (structural check); the frozen *value* (None) is enforced by
        bind_tokenizer. Verify the contract carries the value through."""
        block = _base_block()
        block["padding_token"] = 5
        contract = parse_tokenizer_contract(block)
        assert contract.padding_token_id == 5


# ---------------------------------------------------------------------------
# Namespace-shadow defense (the repo's tokenizers/ dir shadows the library)
# ---------------------------------------------------------------------------


class TestNamespaceShadowDefense:
    def test_raises_when_module_has_no_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Simulate the namespace-package shadow: __file__ is None, no Tokenizer."""

        shadow = ModuleType("tokenizers")
        shadow.__file__ = None
        assert not hasattr(shadow, "Tokenizer")

        real_import_module = importlib.import_module

        def fake_import_module(name: str) -> ModuleType:
            if name == "tokenizers":
                return shadow
            return real_import_module(name)

        monkeypatch.setattr(importlib, "import_module", fake_import_module)
        # Ensure no cached real module interferes.
        monkeypatch.delitem(sys.modules, "tokenizers", raising=False)

        from expertforge.d0.data.tokenizer import _load_tokenizers_module

        with pytest.raises(MissingOptionalDependencyError, match="d0-data"):
            _load_tokenizers_module()

    def test_raises_when_module_has_file_but_no_tokenizer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate a module with a real __file__ but missing the Tokenizer attr."""

        shadow = ModuleType("tokenizers")
        shadow.__file__ = "/fake/path/tokenizers/__init__.py"
        assert not hasattr(shadow, "Tokenizer")

        real_import_module = importlib.import_module

        def fake_import_module(name: str) -> ModuleType:
            if name == "tokenizers":
                return shadow
            return real_import_module(name)

        monkeypatch.setattr(importlib, "import_module", fake_import_module)
        monkeypatch.delitem(sys.modules, "tokenizers", raising=False)

        from expertforge.d0.data.tokenizer import _load_tokenizers_module

        with pytest.raises(MissingOptionalDependencyError, match="d0-data"):
            _load_tokenizers_module()

    def test_raises_on_module_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import_module = importlib.import_module

        def fake_import_module(name: str) -> ModuleType:
            if name == "tokenizers":
                raise ModuleNotFoundError("tokenizers")
            return real_import_module(name)

        monkeypatch.setattr(importlib, "import_module", fake_import_module)
        monkeypatch.delitem(sys.modules, "tokenizers", raising=False)

        from expertforge.d0.data.tokenizer import _load_tokenizers_module

        with pytest.raises(MissingOptionalDependencyError, match="d0-data"):
            _load_tokenizers_module()


# ---------------------------------------------------------------------------
# D0Tokenizer method contracts that do not need the real library
# ---------------------------------------------------------------------------


def _make_tokenizer_with_impl(impl: Any) -> Any:
    """Construct a D0Tokenizer with a fake _impl for unit-level method tests."""

    from expertforge.d0.data.tokenizer import D0Tokenizer

    block = _base_block()
    contract = parse_tokenizer_contract(block)
    return D0Tokenizer(
        contract=contract,
        manifest_sha256=D0_TOKENIZER_MANIFEST_SHA256,
        verified_root=Path("/fake"),
        _impl=impl,
    )


class _FakeEncoding:
    def __init__(self, ids: Sequence[int]) -> None:
        self.ids = list(ids)


class _FakeImpl:
    """Records padding/truncation as None and forwards encode/decode."""

    def __init__(self, encode_map: Mapping[str, Sequence[int]] | None = None) -> None:
        self._encode_map = encode_map or {}
        self.padding: Mapping[str, object] | None = None
        self.truncation: Mapping[str, object] | None = None

    def encode(self, sequence: str, *, add_special_tokens: bool) -> _FakeEncoding:
        return _FakeEncoding(self._encode_map.get(sequence, []))

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool) -> str:
        return "decoded"


class TestD0TokenizerMethods:
    def test_encode_no_special_rejects_crlf(self) -> None:
        t = _make_tokenizer_with_impl(_FakeImpl())
        with pytest.raises(TokenizerBindingError, match="must be normalized"):
            t.encode_no_special("a\r\nb")

    def test_encode_no_special_rejects_cr(self) -> None:
        t = _make_tokenizer_with_impl(_FakeImpl())
        with pytest.raises(TokenizerBindingError, match="must be normalized"):
            t.encode_no_special("a\rb")

    def test_encode_no_special_rejects_nul(self) -> None:
        t = _make_tokenizer_with_impl(_FakeImpl())
        with pytest.raises(TokenizerBindingError, match="not canonical D0 text"):
            t.encode_no_special("a\x00b")

    def test_encode_no_special_rejects_lone_surrogate(self) -> None:
        t = _make_tokenizer_with_impl(_FakeImpl())
        with pytest.raises(TokenizerBindingError, match="not canonical D0 text"):
            t.encode_no_special("a\ud800b")

    def test_encode_no_special_rejects_out_of_range_id(self) -> None:
        """Even a fake impl producing id > 50256 must fail closed."""

        t = _make_tokenizer_with_impl(_FakeImpl({"x": [50257]}))
        with pytest.raises(TokenizerBindingError, match="outside \\[0, 50256\\]"):
            t.encode_no_special("x")

    def test_encode_document_appends_boundary(self) -> None:
        t = _make_tokenizer_with_impl(_FakeImpl({"hello": [25521]}))
        encoded = t.encode_document("hello")
        assert encoded.ids == (25521, 0)
        assert encoded.n_documents == 1

    def test_decode_rejects_bool_id(self) -> None:
        t = _make_tokenizer_with_impl(_FakeImpl())
        with pytest.raises(TokenizerBindingError, match="must be an exact integer"):
            t.decode([True])

    def test_decode_rejects_negative_id(self) -> None:
        t = _make_tokenizer_with_impl(_FakeImpl())
        with pytest.raises(TokenizerBindingError, match="outside the frozen vocabulary"):
            t.decode([-1])

    def test_decode_rejects_out_of_range_id(self) -> None:
        t = _make_tokenizer_with_impl(_FakeImpl())
        with pytest.raises(TokenizerBindingError, match="outside the frozen vocabulary"):
            t.decode([50257])

    def test_decode_accepts_max_id(self) -> None:
        t = _make_tokenizer_with_impl(_FakeImpl())
        # id 50256 is the maximum reachable id and must decode without error
        assert t.decode([50256]) == "decoded"


# ---------------------------------------------------------------------------
# bind_tokenizer frozen-value assertions (structural parsing passes; the
# frozen *values* are enforced in bind_tokenizer against the constants)
# ---------------------------------------------------------------------------


class TestBindTokenizerFrozenValueAssertions:
    """parse_tokenizer_contract validates structure; bind_tokenizer enforces the
    frozen values (vocab size, boundary id, padding=None). These tests confirm a
    manifest whose contract parses cleanly but carries a non-frozen value is
    rejected at the binding layer."""

    def _write_manifest_with_contract(
        self, tmp_path: Path, contract_patch: Mapping[str, Any]
    ) -> tuple[Path, str]:
        """Write a patched manifest and return (path, its digest).

        The patched manifest's own digest is returned so tests can pass it as
        ``expected_manifest_sha256`` to ``bind_tokenizer`` — this lets manifest
        verification pass so the frozen-value *contract* assertions (which is
        what these tests target) are actually reached.
        """

        import hashlib

        manifest = json.loads(CORRECTED_MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["tokenizer"] = {**manifest["tokenizer"], **contract_patch}
        body = dict(manifest)
        body.pop("manifest_sha256", None)
        canonical = json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        manifest["manifest_sha256"] = digest
        out = tmp_path / "patched-manifest.json"
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out, digest

    def test_rejects_padding_token_non_none(self, tmp_path: Path) -> None:
        path, digest = self._write_manifest_with_contract(tmp_path, {"padding_token": 5})
        from expertforge.d0.data.tokenizer import bind_tokenizer

        with pytest.raises(TokenizerBindingError, match="padding token changed"):
            bind_tokenizer(
                manifest_path=path,
                cache_root=tmp_path / "no-cache",
                expected_manifest_sha256=digest,
            )

    def test_rejects_wrong_vocab_size(self, tmp_path: Path) -> None:
        path, digest = self._write_manifest_with_contract(tmp_path, {"vocabulary_size": 50277})
        from expertforge.d0.data.tokenizer import bind_tokenizer

        with pytest.raises(TokenizerBindingError, match="vocabulary size changed"):
            bind_tokenizer(
                manifest_path=path,
                cache_root=tmp_path / "no-cache",
                expected_manifest_sha256=digest,
            )

    def test_rejects_wrong_boundary_id(self, tmp_path: Path) -> None:
        path, digest = self._write_manifest_with_contract(
            tmp_path, {"training_document_boundary_token_id": 7}
        )
        from expertforge.d0.data.tokenizer import bind_tokenizer

        with pytest.raises(TokenizerBindingError, match="document boundary token id changed"):
            bind_tokenizer(
                manifest_path=path,
                cache_root=tmp_path / "no-cache",
                expected_manifest_sha256=digest,
            )
