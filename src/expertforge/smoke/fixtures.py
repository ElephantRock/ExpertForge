"""Committed smoke fixtures: loaders and distinct digest helpers (Issue #14).

The corpus and tokenizer definition are immutable specification inputs (amendment
A). Both are committed under ``tests/fixtures/smoke/`` and pinned by the exact
SHA-256 of their bytes. Every U0/R0/R1 attempt passes stable
:class:`ImmutableInput` entries derived from these fixtures to
:func:`expertforge.provenance.orchestrate.prepare_run`.

Three related contracts use DIFFERENT digest representations (review correction
#5) — this module exposes explicit helpers so the representations are never
conflated:

* :class:`ImmutableInput.digest` and checkpoint
  :class:`~expertforge.checkpoints.models.DataIdentity.dataset_digest` /
  ``data_config_digest`` use the **raw 64-character lowercase hex** digest
  (no prefix).
* manifest :class:`~expertforge.experiments.models.DatasetReference` /
  :class:`~expertforge.experiments.models.TokenizerReference` ``content_digest``
  use the **``sha256:<hex>``** prefixed digest.

Use the ``_hex`` helpers for the unprefixed contract and the ``_content_digest``
helpers for the prefixed manifest contract.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from expertforge.experiments.models import DatasetReference, TokenizerReference
from expertforge.identity.fingerprint import ImmutableInput

__all__ = [
    "SMOKE_CORPUS_RESOURCE",
    "SMOKE_TOKENIZER_RESOURCE",
    "corpus_bytes",
    "corpus_content_digest",
    "corpus_digest_hex",
    "corpus_immutable_input",
    "dataset_reference",
    "fixture_immutable_inputs",
    "load_tokenizer_definition",
    "tokenizer_bytes",
    "tokenizer_content_digest",
    "tokenizer_digest_hex",
    "tokenizer_immutable_input",
    "tokenizer_reference",
    "tokenizer_vocab_size",
]

SMOKE_CORPUS_RESOURCE = ("tests.fixtures.smoke", "smoke_corpus.txt")
SMOKE_TOKENIZER_RESOURCE = ("tests.fixtures.smoke", "tokenizer.json")

# Stable immutable-input names. These names enter the specification fingerprint
# and are referenced by the manifest dataset/tokenizer ``immutable_input_name``.
SMOKE_DATASET_INPUT_NAME = "smoke.dataset"
SMOKE_TOKENIZER_INPUT_NAME = "smoke.tokenizer"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _project_root() -> Path:
    """The ExpertForge project root (parent of the ``src`` and ``tests`` dirs)."""
    # This module lives at src/expertforge/smoke/fixtures.py -> up 4 to root.
    return Path(__file__).resolve().parents[3]


def _read_resource(resource: tuple[str, str]) -> bytes:
    """Read a committed fixture resource as ``(package, filename)``.

    Fixtures live under ``tests/fixtures/smoke/`` which is committed to the
    repository and present at runtime (the package is developed/tested in place,
    never installed as a zip). The package is resolved relative to the project
    root (the ``tests`` package's parent); the filename keeps its extension.
    """
    package, filename = resource
    project_root = _project_root()
    fixture_path = project_root.joinpath(*package.split("."), filename)
    if not fixture_path.is_file():
        raise FileNotFoundError(f"smoke fixture not found: {fixture_path}")
    return fixture_path.read_bytes()


@lru_cache(maxsize=1)
def corpus_bytes() -> bytes:
    """The exact committed corpus bytes (raw training data over 16 symbols)."""
    return _read_resource(SMOKE_CORPUS_RESOURCE)


@lru_cache(maxsize=1)
def tokenizer_bytes() -> bytes:
    """The exact committed canonical tokenizer-definition bytes."""
    return _read_resource(SMOKE_TOKENIZER_RESOURCE)


def corpus_digest_hex() -> str:
    """Raw 64-char lowercase hex SHA-256 of the corpus bytes (unprefixed).

    Used for ``ImmutableInput.digest`` and checkpoint
    ``DataIdentity.dataset_digest``.
    """
    return _sha256_hex(corpus_bytes())


def tokenizer_digest_hex() -> str:
    """Raw 64-char lowercase hex SHA-256 of the tokenizer bytes (unprefixed)."""
    return _sha256_hex(tokenizer_bytes())


def corpus_content_digest() -> str:
    """Prefixed ``sha256:<hex>`` digest of the corpus bytes.

    Used for manifest ``DatasetReference.content_digest``.
    """
    return f"sha256:{corpus_digest_hex()}"


def tokenizer_content_digest() -> str:
    """Prefixed ``sha256:<hex>`` digest of the tokenizer bytes.

    Used for manifest ``TokenizerReference.content_digest``.
    """
    return f"sha256:{tokenizer_digest_hex()}"


def corpus_immutable_input() -> ImmutableInput:
    """The corpus immutable input (raw hex digest) for the specification fingerprint."""
    return ImmutableInput(
        name=SMOKE_DATASET_INPUT_NAME,
        algorithm="sha256",
        digest=corpus_digest_hex(),
    )


def tokenizer_immutable_input() -> ImmutableInput:
    """The tokenizer immutable input (raw hex digest) for the specification fingerprint."""
    return ImmutableInput(
        name=SMOKE_TOKENIZER_INPUT_NAME,
        algorithm="sha256",
        digest=tokenizer_digest_hex(),
    )


def fixture_immutable_inputs() -> list[ImmutableInput]:
    """The fixture immutable inputs (corpus + tokenizer), sorted by name.

    Caller prepends the ``source.snapshot`` input produced by ``prepare_run``.
    """
    return [corpus_immutable_input(), tokenizer_immutable_input()]


def load_tokenizer_definition() -> dict[str, Any]:
    """The parsed canonical tokenizer definition."""
    definition: dict[str, Any] = json.loads(tokenizer_bytes())
    return definition


def tokenizer_vocab_size() -> int:
    """The fixture tokenizer vocabulary size (16)."""
    return int(load_tokenizer_definition()["vocab_size"])


def dataset_reference() -> DatasetReference:
    """The manifest dataset reference bound to the corpus immutable input."""
    return DatasetReference(
        dataset_id="smoke-printable16-corpus-v1",
        split="train",
        immutable_input_name=SMOKE_DATASET_INPUT_NAME,
        content_digest=corpus_content_digest(),
        is_fixture=True,
    )


def tokenizer_reference() -> TokenizerReference:
    """The manifest tokenizer reference bound to the tokenizer immutable input."""
    return TokenizerReference(
        tokenizer_id="smoke-printable16-v1",
        vocab_size=tokenizer_vocab_size(),
        immutable_input_name=SMOKE_TOKENIZER_INPUT_NAME,
        content_digest=tokenizer_content_digest(),
        is_fixture=True,
    )
