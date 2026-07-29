"""Specification fingerprint (Issue #6 decision §1).

A versioned deterministic envelope is canonicalized as compact, sorted-key UTF-8
JSON, then SHA-256'd. v1 envelope::

    {
      "schema": "expertforge.specification-fingerprint",
      "version": 1,
      "canonical_config": {"algorithm": "sha256", "digest": "<sha256(canonical_bytes)>"},
      "immutable_inputs": []
    }

``immutable_inputs`` is a frozen, uniquely-named, name-sorted tuple of
``ImmutableInput`` records. Paths, timestamps, secrets, environment ordering,
and display labels never enter the fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "FINGERPRINT_VERSION",
    "FINGERPRINT_VERSION_STR",
    "ImmutableInput",
    "SpecificationFingerprint",
    "specification_fingerprint",
]

# The fingerprint-envelope version. Bumped only when the envelope's byte
# representation changes (Issue #6 §versioning).
FINGERPRINT_VERSION: int = 1
FINGERPRINT_VERSION_STR: str = "v1"


class ImmutableInput(BaseModel):
    """One declared immutable input digest (dataset, tokenizer, source, ...).

    Names are stable identifiers (not paths/labels); algorithms are digest
    algorithm names; digests are lowercase hex.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    name: str = Field(..., min_length=1)
    algorithm: str = Field(..., min_length=1)
    digest: str = Field(..., min_length=1)


@dataclass(frozen=True)
class SpecificationFingerprint:
    """A computed specification fingerprint.

    Attributes:
        envelope: the canonicalizable versioned envelope (a plain dict).
        envelope_canonical_bytes: compact sorted-key UTF-8 JSON of the envelope.
        canonical_config_digest: SHA-256 hex of the input ``canonical_bytes``.
        digest_str: the public ``spec-v1-sha256-<64 hex>`` identifier.
    """

    envelope: dict[str, Any]
    envelope_canonical_bytes: bytes
    canonical_config_digest: str
    digest_str: str


def _canonical_json_bytes(obj: Any) -> bytes:
    """Compact, sorted-key, UTF-8, non-finite-prohibiting JSON encoding."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def specification_fingerprint(
    canonical_config_bytes: bytes,
    immutable_inputs: Sequence[ImmutableInput] | None = None,
) -> SpecificationFingerprint:
    """Compute the specification fingerprint for resolved canonical config bytes.

    ``canonical_config_bytes`` is the output of
    :func:`expertforge.config.resolve.canonical_bytes` — the deterministic
    behavioral-bytes representation of the resolved configuration.
    """
    inputs = list(immutable_inputs or [])
    # Reject duplicate names before sorting/serializing.
    names = [ii.name for ii in inputs]
    if len(set(names)) != len(names):
        dups = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"Duplicate immutable-input names: {dups}")
    # Sort by name for stable envelope serialization.
    inputs_sorted = sorted(inputs, key=lambda ii: ii.name)

    canonical_digest = hashlib.sha256(canonical_config_bytes).hexdigest()
    envelope: dict[str, Any] = {
        "schema": "expertforge.specification-fingerprint",
        "version": FINGERPRINT_VERSION,
        "canonical_config": {"algorithm": "sha256", "digest": canonical_digest},
        "immutable_inputs": [ii.model_dump() for ii in inputs_sorted],
    }
    envelope_bytes = _canonical_json_bytes(envelope)
    envelope_digest = hashlib.sha256(envelope_bytes).hexdigest()
    digest_str = f"spec-{FINGERPRINT_VERSION_STR}-sha256-{envelope_digest}"
    return SpecificationFingerprint(
        envelope=envelope,
        envelope_canonical_bytes=envelope_bytes,
        canonical_config_digest=canonical_digest,
        digest_str=digest_str,
    )
