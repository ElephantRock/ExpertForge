"""Frozen D0 tokenizer binding over verified corrected source bytes.

This module binds the corrected D0 tokenizer
(``tokenizers/manifests/d0-gpt-neox-corrected-v1.json``, Decision 0012 Resolution 1)
to a runtime encode/decode surface. It is the first material sub-step of the D0.1b
deterministic training-data pipeline (Issue #53), landing on the D0.0 amendment
branch (#55) because the binding depends on the corrected artifact.

Contract enforced here (mirrors the corrected manifest):

- byte-level BPE, vocabulary size 50,257 (reachable ids exactly ``0..50256``)
- no normalizer; ByteLevel pre-tokenizer with ``add_prefix_space = false``
- a single special token ``<|endoftext|>`` at id 0, carrying the four roles
  ``bos`` / ``document_boundary`` / ``eos`` / ``unk``
- no padding token
- training document boundary token id = 0

Design notes (review corrections on the original binding plan):

1. **Explicit document terminator.** The document form appends the boundary
   token id (0) manually after a no-special encode. We never rely on the
   tokenizer's serialized post-processor.
2. **Single manifest read.** The manifest bytes are read and decoded once, then
   the same mapping is passed to both the source-manifest parser and the
   tokenizer-contract parser.
3. **Descriptor-bound loading.** ``tokenizer.json`` is read through a protected
   descriptor via ``verify_source_file_bytes`` and loaded with
   ``Tokenizer.from_buffer`` over those exact bytes.
4. **Strict vocabulary equality.** ``get_vocab_size(with_added_tokens=True) ==
   50257`` is enforced verbatim. Any tokenizer capable of producing an id above
   ``50256`` fails closed — this is the check that surfaced the original
   divergence and must remain intact.
5. **Typed failure boundaries** are kept distinct: ``SourceManifestError``,
   ``SourceVerificationError``, ``MissingOptionalDependencyError``,
   ``TokenizerBindingError``, ``NormalizationError``.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

from expertforge.d0.errors import (
    MissingOptionalDependencyError,
    NormalizationError,
    TokenizerBindingError,
)
from expertforge.d0.normalization import normalize_text
from expertforge.d0.source_manifest import parse_source_manifest
from expertforge.d0.source_verification import (
    resolve_inventory_path,
    verify_source_file_bytes,
    verify_source_inventory,
)

__all__ = [
    "D0_DOCUMENT_BOUNDARY_TOKEN_ID",
    "D0_MAX_REACHABLE_TOKEN_ID",
    "D0_PADDING_TOKEN_ID",
    "D0_TOKENIZER_MANIFEST_SHA256",
    "D0_TOKENIZER_VOCAB_SIZE",
    "D0Tokenizer",
    "D0TokenizerContract",
    "Encoded",
    "PreTokenizerSpec",
    "TokenizerBindingError",
    "TokenizerSpecialToken",
    "bind_tokenizer",
    "parse_tokenizer_contract",
]

# --- Frozen D0 tokenizer identity (corrected artifact, Decision 0012) ------

# ``manifest_sha256`` of ``tokenizers/manifests/d0-gpt-neox-corrected-v1.json``.
# The binding refuses any manifest whose canonical-JSON digest differs.
D0_TOKENIZER_MANIFEST_SHA256 = "1142baf61c330a318be6e1b153f6769892a7f5af3b7241363a1d0a1f2ca453e1"
D0_TOKENIZER_VOCAB_SIZE = 50257
D0_MAX_REACHABLE_TOKEN_ID = 50256
D0_DOCUMENT_BOUNDARY_TOKEN_ID = 0
D0_PADDING_TOKEN_ID: int | None = None

_REQUIRED_SPECIAL_ROLES: frozenset[str] = frozenset({"bos", "document_boundary", "eos", "unk"})

_TOKENIZER_JSON_PATH = "tokenizer.json"


# --- Contract records ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreTokenizerSpec:
    """Frozen ByteLevel pre-tokenizer configuration."""

    type: str
    add_prefix_space: bool


@dataclass(frozen=True, slots=True)
class TokenizerSpecialToken:
    """One registered special token with its frozen roles."""

    token: str
    token_id: int
    roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class D0TokenizerContract:
    """The frozen D0 tokenizer contract parsed from the manifest ``tokenizer`` block."""

    family: str
    vocabulary_size: int
    normalizer: str
    pre_tokenizer: PreTokenizerSpec
    special_tokens: tuple[TokenizerSpecialToken, ...]
    padding_token_id: int | None
    training_document_boundary_token_id: int


@dataclass(frozen=True, slots=True)
class Encoded:
    """Result of encoding one document.

    ``n_documents`` describes the operation that produced this object (always 1
    for ``encode_document``), never a count of token id 0 occurrences — source
    text may itself contain ``<|endoftext|>`` and legitimately encode to id 0.
    """

    ids: tuple[int, ...]
    n_documents: int


# --- Lazy optional-dependency boundary -------------------------------------


class _EncodingLike(Protocol):
    @property
    def ids(self) -> list[int]: ...


class _TokenizerLike(Protocol):
    @property
    def padding(self) -> Mapping[str, object] | None: ...

    @property
    def truncation(self) -> Mapping[str, object] | None: ...

    def encode(self, sequence: str, *, add_special_tokens: bool) -> _EncodingLike: ...

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool) -> str: ...

    def get_vocab_size(self, *, with_added_tokens: bool = True) -> int: ...

    def token_to_id(self, token: str) -> int | None: ...


class _TokenizerFactoryLike(Protocol):
    @staticmethod
    def from_buffer(buffer: bytes) -> _TokenizerLike: ...


class _TokenizerModuleLike(Protocol):
    Tokenizer: type[_TokenizerFactoryLike]


def _load_tokenizers_module() -> _TokenizerModuleLike:
    """Import the real ``tokenizers`` package, defeating the namespace shadow.

    The repository's top-level ``tokenizers/`` directory (holding manifests) is
    picked up as a namespace package when the PyPI ``tokenizers`` library is
    absent: ``import tokenizers`` then "succeeds" with ``__file__ is None`` and
    no ``Tokenizer`` attribute. A naive ``try/except ModuleNotFoundError``
    loader would pass that shadow and crash later. We therefore require both a
    real ``__file__`` and the ``Tokenizer`` factory.
    """

    try:
        module: ModuleType = importlib.import_module("tokenizers")
    except ModuleNotFoundError as exc:
        raise MissingOptionalDependencyError(
            "D0 tokenizer binding requires the optional 'd0-data' extra; "
            "install ExpertForge with expertforge[d0-data]"
        ) from exc
    module_file = getattr(module, "__file__", None)
    tokenizer_factory = getattr(module, "Tokenizer", None)
    if not isinstance(module_file, str) or not module_file or tokenizer_factory is None:
        raise MissingOptionalDependencyError(
            "D0 tokenizer binding requires the optional 'd0-data' extra; "
            "install ExpertForge with expertforge[d0-data]"
        )
    return cast(_TokenizerModuleLike, module)


# --- Local fail-closed contract-validation helpers -------------------------
# These mirror the style of ``expertforge.d0.source_manifest`` but raise the
# tokenizer-specific ``TokenizerBindingError`` for contract violations. They
# must not import the private source-manifest helpers, because those raise
# ``SourceManifestError`` (which denotes a source-manifest failure, not a
# tokenizer-contract failure).


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TokenizerBindingError(f"{field} must be a JSON object")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TokenizerBindingError(f"{field} must be a non-empty string")
    return value


def _require_exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    # ``type(...) is not int`` rejects bools (True/False are int subclasses) and
    # any non-int, exactly as the source-manifest helpers do.
    if type(value) is not int or value < minimum:
        raise TokenizerBindingError(f"{field} must be an integer >= {minimum}")
    return value


def _require_optional_exact_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _require_exact_int(value, field)


def _parse_special_token(raw: object, index: int) -> TokenizerSpecialToken:
    entry = _require_mapping(raw, f"special_tokens[{index}]")
    if set(entry) != {"token", "token_id", "roles"}:
        raise TokenizerBindingError(f"special_tokens[{index}] fields changed")
    token = _require_string(entry.get("token"), f"special_tokens[{index}].token")
    token_id = _require_exact_int(entry.get("token_id"), f"special_tokens[{index}].token_id")
    raw_roles = entry.get("roles")
    if not isinstance(raw_roles, Sequence) or isinstance(raw_roles, (str, bytes, bytearray)):
        raise TokenizerBindingError(f"special_tokens[{index}].roles must be an array")
    roles_list: list[str] = []
    for role_index, role in enumerate(raw_roles):
        role_str = _require_string(role, f"special_tokens[{index}].roles[{role_index}]")
        roles_list.append(role_str)
    roles = tuple(roles_list)
    if len(set(roles)) != len(roles):
        raise TokenizerBindingError(f"special_tokens[{index}].roles must be unique")
    return TokenizerSpecialToken(token=token, token_id=token_id, roles=roles)


def _parse_pre_tokenizer(raw: object) -> PreTokenizerSpec:
    entry = _require_mapping(raw, "pre_tokenizer")
    if set(entry) != {"type", "add_prefix_space"}:
        raise TokenizerBindingError("pre_tokenizer fields changed")
    type_name = _require_string(entry.get("type"), "pre_tokenizer.type")
    add_prefix_space = entry.get("add_prefix_space")
    if type(add_prefix_space) is not bool:
        raise TokenizerBindingError("pre_tokenizer.add_prefix_space must be a boolean")
    return PreTokenizerSpec(type=type_name, add_prefix_space=add_prefix_space)


def parse_tokenizer_contract(raw_tokenizer: Mapping[str, Any]) -> D0TokenizerContract:
    """Parse and fully validate the frozen D0 tokenizer contract block.

    Raises ``TokenizerBindingError`` on any deviation from the frozen contract.
    """

    block = _require_mapping(raw_tokenizer, "tokenizer")
    expected_fields = {
        "family",
        "vocabulary_size",
        "normalizer",
        "pre_tokenizer",
        "special_tokens",
        "padding_token",
        "training_document_boundary_token_id",
    }
    if set(block) != expected_fields:
        raise TokenizerBindingError("tokenizer contract fields changed")

    family = _require_string(block.get("family"), "tokenizer.family")
    vocabulary_size = _require_exact_int(block.get("vocabulary_size"), "tokenizer.vocabulary_size")
    normalizer = _require_string(block.get("normalizer"), "tokenizer.normalizer")
    pre_tokenizer = _parse_pre_tokenizer(block.get("pre_tokenizer"))

    raw_special_tokens = block.get("special_tokens")
    if not isinstance(raw_special_tokens, Sequence) or isinstance(
        raw_special_tokens, (str, bytes, bytearray)
    ):
        raise TokenizerBindingError("tokenizer.special_tokens must be an array")
    special_tokens = tuple(
        _parse_special_token(raw, index) for index, raw in enumerate(raw_special_tokens)
    )

    padding_token_id = _require_optional_exact_int(
        block.get("padding_token"), "tokenizer.padding_token"
    )
    boundary_id = _require_exact_int(
        block.get("training_document_boundary_token_id"),
        "tokenizer.training_document_boundary_token_id",
    )

    _validate_special_token_contract(special_tokens)

    return D0TokenizerContract(
        family=family,
        vocabulary_size=vocabulary_size,
        normalizer=normalizer,
        pre_tokenizer=pre_tokenizer,
        special_tokens=special_tokens,
        padding_token_id=padding_token_id,
        training_document_boundary_token_id=boundary_id,
    )


def _validate_special_token_contract(special_tokens: tuple[TokenizerSpecialToken, ...]) -> None:
    """Enforce the frozen single-special-token, four-role contract."""

    if len(special_tokens) != 1:
        raise TokenizerBindingError(
            f"tokenizer contract requires exactly one special token; got {len(special_tokens)}"
        )
    only = special_tokens[0]
    if only.token != "<|endoftext|>":
        raise TokenizerBindingError(
            f"special token string changed: expected <|endoftext|>, got {only.token}"
        )
    if only.token_id != D0_DOCUMENT_BOUNDARY_TOKEN_ID:
        raise TokenizerBindingError(
            f"special token id changed: expected {D0_DOCUMENT_BOUNDARY_TOKEN_ID}, "
            f"got {only.token_id}"
        )
    if set(only.roles) != _REQUIRED_SPECIAL_ROLES:
        raise TokenizerBindingError(
            "special token roles changed: expected exactly "
            f"{sorted(_REQUIRED_SPECIAL_ROLES)}, got {sorted(only.roles)}"
        )


# --- Frozen runtime binding ------------------------------------------------


@dataclass(frozen=True, slots=True)
class D0Tokenizer:
    """A bound, verified D0 tokenizer exposing a frozen encode/decode surface."""

    contract: D0TokenizerContract
    manifest_sha256: str
    verified_root: Path
    _impl: _TokenizerLike = field(repr=False, compare=False)

    @property
    def vocab_size(self) -> int:
        return self.contract.vocabulary_size

    @property
    def document_boundary_token_id(self) -> int:
        return self.contract.training_document_boundary_token_id

    @property
    def padding_token_id(self) -> int | None:
        return self.contract.padding_token_id

    def encode_no_special(self, text: str) -> tuple[int, ...]:
        """Encode text without any special tokens.

        The input must already be in canonical D0 form (``normalize_text(text)
        == text``): CRLF/CR must have been mapped to LF, and NUL code points or
        lone surrogates are rejected. This pins the exact byte sequence the BPE
        sees, independent of the caller's platform newlines.
        """

        try:
            canonical = normalize_text(text)
        except NormalizationError as exc:
            raise TokenizerBindingError(f"input text is not canonical D0 text: {exc}") from exc
        if canonical != text:
            raise TokenizerBindingError(
                "input text must be normalized before tokenization (CRLF/CR -> LF)"
            )
        try:
            encoding = self._impl.encode(text, add_special_tokens=False)
        except Exception as exc:
            raise TokenizerBindingError("tokenizer failed to encode text") from exc
        ids = tuple(encoding.ids)
        # Belt-and-braces: even though the corrected artifact cannot produce an
        # out-of-range id, every encoding result is checked here so a future
        # artifact swap cannot silently feed the model an id above 50256.
        for token_id in ids:
            if not 0 <= token_id <= D0_MAX_REACHABLE_TOKEN_ID:
                raise TokenizerBindingError(
                    f"tokenizer produced id {token_id} outside [0, {D0_MAX_REACHABLE_TOKEN_ID}]"
                )
        return ids

    def encode_document(self, text: str) -> Encoded:
        """Encode one document and append the trailing boundary token.

        The D0 contract requires one terminator *after* each document. The
        boundary token (id 0) is appended manually rather than relying on the
        tokenizer's serialized post-processor.

        ``n_documents`` is always 1 here: it describes this operation, never a
        count of token id 0 occurrences.
        """

        payload = self.encode_no_special(text)
        return Encoded(
            ids=payload + (self.document_boundary_token_id,),
            n_documents=1,
        )

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token ids, skipping registered special tokens.

        Each id is validated to be an exact integer within the frozen vocabulary
        before it crosses the native boundary. ``skip_special_tokens=True``
        removes registered special tokens (notably id 0), matching the frozen
        D0 decode contract: ``decode((0,)) == ""``.
        """

        validated: list[int] = []
        for index, token_id in enumerate(ids):
            if type(token_id) is not int:
                raise TokenizerBindingError(f"token id at index {index} must be an exact integer")
            if not 0 <= token_id < self.vocab_size:
                raise TokenizerBindingError(
                    f"token id at index {index} is outside the frozen vocabulary"
                )
            validated.append(token_id)
        try:
            return self._impl.decode(validated, skip_special_tokens=True)
        except Exception as exc:
            raise TokenizerBindingError("tokenizer failed to decode ids") from exc


def bind_tokenizer(
    *,
    manifest_path: Path,
    cache_root: Path,
    expected_manifest_sha256: str = D0_TOKENIZER_MANIFEST_SHA256,
) -> D0Tokenizer:
    """Bind the verified D0 tokenizer from a manifest and a local cache root.

    Steps:
    1. Read and decode the manifest bytes once (single read).
    2. Parse the source manifest against the expected frozen digest.
    3. Enforce ``manifest.kind == "tokenizer"``.
    4. Parse the ``tokenizer`` contract block from the same mapping.
    5. Verify the inventory file(s) against the frozen manifest.
    6. Re-read ``tokenizer.json`` through a protected descriptor and load it via
       ``Tokenizer.from_buffer`` over those exact bytes.
    7. Fail closed on any mismatch in vocab size (strict equality), boundary
       token id, or active runtime padding/truncation.
    """

    raw = manifest_path.read_bytes()
    decoded = raw.decode("utf-8", errors="strict")
    value = _require_mapping(json.loads(decoded), "manifest")
    manifest = parse_source_manifest(
        value,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if manifest.kind != "tokenizer":
        raise TokenizerBindingError(
            f"expected a tokenizer source manifest; got kind={manifest.kind!r}"
        )
    contract = parse_tokenizer_contract(_require_mapping(value.get("tokenizer"), "tokenizer"))

    # Frozen-identity assertions: the parsed contract must match the constants
    # this module is compiled against.
    if contract.vocabulary_size != D0_TOKENIZER_VOCAB_SIZE:
        raise TokenizerBindingError(
            f"vocabulary size changed: expected {D0_TOKENIZER_VOCAB_SIZE}, "
            f"got {contract.vocabulary_size}"
        )
    if contract.training_document_boundary_token_id != D0_DOCUMENT_BOUNDARY_TOKEN_ID:
        raise TokenizerBindingError(
            f"document boundary token id changed: expected "
            f"{D0_DOCUMENT_BOUNDARY_TOKEN_ID}, "
            f"got {contract.training_document_boundary_token_id}"
        )
    if contract.padding_token_id != D0_PADDING_TOKEN_ID:
        raise TokenizerBindingError(
            f"padding token changed: expected {D0_PADDING_TOKEN_ID!r}, "
            f"got {contract.padding_token_id!r}"
        )

    # Verify the full frozen inventory. Reuses the descriptor-bound,
    # symlink-rejecting, size+sha-checking verification path.
    verify_source_inventory(cache_root, manifest)

    # Re-read tokenizer.json through its own protected descriptor and load from
    # the exact verified bytes (no path-based reopen after verification).
    tokenizer_identity = manifest.file(_TOKENIZER_JSON_PATH)
    tokenizer_path = resolve_inventory_path(cache_root, _TOKENIZER_JSON_PATH)
    verified = verify_source_file_bytes(tokenizer_path, tokenizer_identity)

    tokenizer_module = _load_tokenizers_module()
    try:
        impl = tokenizer_module.Tokenizer.from_buffer(verified.payload)
    except Exception as exc:
        raise TokenizerBindingError("verified tokenizer.json could not be loaded") from exc

    # Fail closed if the loaded tokenizer has active padding or truncation.
    if impl.padding is not None:
        raise TokenizerBindingError("loaded tokenizer has active padding")
    if impl.truncation is not None:
        raise TokenizerBindingError("loaded tokenizer has active truncation")

    # Strict vocabulary equality. This is the load-bearing check that surfaced
    # the original divergence: the corrected artifact must report exactly
    # 50,257 reachable ids. Any artifact capable of producing an id above 50256
    # fails closed here.
    try:
        loaded_vocab_size = impl.get_vocab_size(with_added_tokens=True)
    except Exception as exc:
        raise TokenizerBindingError("tokenizer failed to report vocabulary size") from exc
    if loaded_vocab_size != D0_TOKENIZER_VOCAB_SIZE:
        raise TokenizerBindingError(
            f"loaded vocabulary size mismatch: expected {D0_TOKENIZER_VOCAB_SIZE}, "
            f"got {loaded_vocab_size}"
        )
    try:
        loaded_boundary = impl.token_to_id("<|endoftext|>")
    except Exception as exc:
        raise TokenizerBindingError("tokenizer failed to resolve boundary token") from exc
    if loaded_boundary != D0_DOCUMENT_BOUNDARY_TOKEN_ID:
        raise TokenizerBindingError(
            f"loaded boundary token id mismatch: expected "
            f"{D0_DOCUMENT_BOUNDARY_TOKEN_ID}, got {loaded_boundary!r}"
        )

    return D0Tokenizer(
        contract=contract,
        manifest_sha256=manifest.manifest_sha256,
        verified_root=cache_root.resolve(),
        _impl=impl,
    )
