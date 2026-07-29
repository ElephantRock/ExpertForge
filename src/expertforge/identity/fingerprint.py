"""Specification fingerprint (Issue #6 decision §1, review items 2–4).

A versioned deterministic envelope is canonicalized as compact, sorted-key UTF-8
JSON, then SHA-256'd. v1 envelope::

    {
      "schema": "expertforge.specification-fingerprint",
      "version": 1,
      "canonical_config": {"algorithm": "sha256", "digest": "<sha256(canonical_bytes)>"},
      "immutable_inputs": []
    }

The fingerprint is modeled as **deeply immutable nested Pydantic v2 records**
so it round-trips through the identity sidecar without mutable containers.
Immutable inputs are validated: stable lowercase names, supported algorithms
(``sha256`` for v1), and exactly 64 lowercase-hex digests. Paths, timestamps,
secrets, environment ordering, and display labels never enter the fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "DIGEST_HEX_PATTERN",
    "FINGERPRINT_VERSION",
    "FINGERPRINT_VERSION_STR",
    "FingerprintMismatch",
    "IMMUTABLE_INPUT_NAME_PATTERN",
    "SUPPORTED_ALGORITHMS",
    "CanonicalConfigDigest",
    "ImmutableInput",
    "SpecificationFingerprintRecord",
    "specification_fingerprint",
    "verify_fingerprint",
]


class FingerprintMismatch(Exception):
    """Raised when a recomputed fingerprint does not match a stored record.

    Detects: changed canonical configuration, changed/missing/additional
    immutable input, altered envelope content, altered public digest, and
    unsupported fingerprint-envelope version.
    """


FINGERPRINT_VERSION: int = 1
FINGERPRINT_VERSION_STR: str = "v1"

# Stable, location-independent lowercase name syntax (no path separators,
# no leading dot/dash). Examples: "dataset", "tokenizer.bpe", "source.git".
IMMUTABLE_INPUT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
# 64 lowercase hexadecimal characters (SHA-256).
DIGEST_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
# v1 supports SHA-256 only.
SUPPORTED_ALGORITHMS: frozenset[str] = frozenset({"sha256"})


def _canonical_json_bytes(obj: Any) -> bytes:
    """Compact, sorted-key, UTF-8, non-finite-prohibiting JSON encoding."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class ImmutableInput(BaseModel):
    """One declared immutable input digest (dataset, tokenizer, source, ...).

    Names are stable, location-independent lowercase identifiers (never paths or
    display labels); algorithms are restricted to the supported set; digests are
    exactly 64 lowercase hexadecimal characters.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    name: str = Field(..., min_length=1)
    algorithm: str = Field(..., min_length=1)
    digest: str = Field(..., min_length=1)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not IMMUTABLE_INPUT_NAME_PATTERN.match(v):
            raise ValueError(
                f"immutable-input name {v!r} must match {IMMUTABLE_INPUT_NAME_PATTERN.pattern} "
                "(stable lowercase identifier; no path separators)."
            )
        return v

    @field_validator("algorithm")
    @classmethod
    def _validate_algorithm(cls, v: str) -> str:
        if v not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"immutable-input algorithm {v!r} is not supported; "
                f"v1 supports {sorted(SUPPORTED_ALGORITHMS)}."
            )
        return v

    @field_validator("digest")
    @classmethod
    def _validate_digest(cls, v: str) -> str:
        if not DIGEST_HEX_PATTERN.match(v):
            raise ValueError(f"immutable-input digest must be 64 lowercase hex chars; got {v!r}.")
        return v


class CanonicalConfigDigest(BaseModel):
    """SHA-256 digest of the canonical configuration bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    algorithm: str = Field(default="sha256")
    digest: str = Field(..., min_length=1)

    @field_validator("algorithm")
    @classmethod
    def _validate_algorithm(cls, v: str) -> str:
        if v not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"canonical-config algorithm {v!r} is not supported; "
                f"v1 supports {sorted(SUPPORTED_ALGORITHMS)}."
            )
        return v

    @field_validator("digest")
    @classmethod
    def _validate_digest(cls, v: str) -> str:
        if not DIGEST_HEX_PATTERN.match(v):
            raise ValueError(f"canonical-config digest must be 64 lowercase hex chars; got {v!r}.")
        return v


class SpecificationFingerprintRecord(BaseModel):
    """The reconstructable, deeply immutable specification fingerprint.

    Contains the typed envelope (schema + version + canonical_config + immutable
    inputs) and the public ``spec-v1-sha256-<hex>`` digest. ``digest_str`` is
    derived from the canonical envelope bytes and is verified on construction.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
        populate_by_name=True,
    )

    # Named `schema_name` to avoid shadowing BaseModel.schema; serialized as
    # `schema` to match the canonical envelope contract.
    schema_name: str = Field(default="expertforge.specification-fingerprint", alias="schema")
    version: int = Field(default=FINGERPRINT_VERSION)
    canonical_config: CanonicalConfigDigest
    immutable_inputs: tuple[ImmutableInput, ...] = Field(default_factory=tuple)
    digest_str: str = Field(..., min_length=1)

    @field_validator("version")
    @classmethod
    def _validate_version(cls, v: int) -> int:
        if v != FINGERPRINT_VERSION:
            raise ValueError(
                f"Unsupported fingerprint-envelope version {v}; "
                f"this version supports {FINGERPRINT_VERSION}."
            )
        return v

    @field_validator("digest_str")
    @classmethod
    def _validate_digest_str(cls, v: str) -> str:
        if not v.startswith(f"spec-{FINGERPRINT_VERSION_STR}-sha256-"):
            raise ValueError(
                f"digest_str must be 'spec-{FINGERPRINT_VERSION_STR}-sha256-<64 hex>'; got {v!r}."
            )
        hexpart = v.rsplit("-", 1)[-1]
        if not re.fullmatch(r"[0-9a-f]{64}", hexpart):
            raise ValueError(
                f"digest_str hex suffix must be 64 lowercase hex chars; got {hexpart!r}."
            )
        return v

    def envelope_dict(self) -> dict[str, Any]:
        """The canonicalizable envelope as a plain (sorted on serialization) dict."""
        return {
            "schema": self.schema_name,
            "version": self.version,
            "canonical_config": self.canonical_config.model_dump(),
            "immutable_inputs": [ii.model_dump() for ii in self.immutable_inputs],
        }

    def envelope_canonical_bytes(self) -> bytes:
        """Compact sorted-key UTF-8 JSON of the envelope."""
        return _canonical_json_bytes(self.envelope_dict())

    def verify_digest(self) -> None:
        """Recompute the envelope digest and assert it matches ``digest_str``.

        Raises ValueError on mismatch — used at load/construction to detect
        tampering or drift between the stored envelope and stored digest.
        """
        recomputed = hashlib.sha256(self.envelope_canonical_bytes()).hexdigest()
        expected = self.digest_str.rsplit("-", 1)[-1]
        if recomputed != expected:
            raise ValueError(
                f"digest_str {expected!r} does not match recomputed envelope digest {recomputed!r}."
            )

    @classmethod
    def from_envelope(
        cls,
        canonical_config_digest: str,
        immutable_inputs: Sequence[ImmutableInput],
        *,
        verify: bool = True,
    ) -> SpecificationFingerprintRecord:
        """Build a record from raw canonical-config digest + immutable inputs,
        computing the public digest. Raises ValueError on duplicate names or
        (when ``verify``) on any inconsistency.
        """
        names = [ii.name for ii in immutable_inputs]
        if len(set(names)) != len(names):
            dups = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate immutable-input names: {dups}")
        inputs_sorted = tuple(sorted(immutable_inputs, key=lambda ii: ii.name))
        cc = CanonicalConfigDigest(algorithm="sha256", digest=canonical_config_digest)
        envelope_bytes = _canonical_json_bytes(
            {
                "schema": "expertforge.specification-fingerprint",
                "version": FINGERPRINT_VERSION,
                "canonical_config": cc.model_dump(),
                "immutable_inputs": [ii.model_dump() for ii in inputs_sorted],
            }
        )
        envelope_digest = hashlib.sha256(envelope_bytes).hexdigest()
        digest_str = f"spec-{FINGERPRINT_VERSION_STR}-sha256-{envelope_digest}"
        rec = cls(
            version=FINGERPRINT_VERSION,
            canonical_config=cc,
            immutable_inputs=inputs_sorted,
            digest_str=digest_str,
        )
        if verify:
            rec.verify_digest()
        return rec


def specification_fingerprint(
    canonical_config_bytes: bytes,
    immutable_inputs: Sequence[ImmutableInput] | None = None,
) -> SpecificationFingerprintRecord:
    """Compute the specification fingerprint for resolved canonical config bytes.

    ``canonical_config_bytes`` is the output of
    :func:`expertforge.config.resolve.canonical_bytes` — the deterministic
    behavioral-bytes representation of the resolved configuration.
    """
    inputs = list(immutable_inputs or [])
    canonical_digest = hashlib.sha256(canonical_config_bytes).hexdigest()
    return SpecificationFingerprintRecord.from_envelope(canonical_digest, inputs)


def verify_fingerprint(
    record: SpecificationFingerprintRecord,
    canonical_config_bytes: bytes,
    immutable_inputs: Sequence[ImmutableInput] | None = None,
) -> None:
    """Recompute the specification fingerprint from independent inputs and
    assert it matches ``record``.

    Detects mismatches in: canonical configuration (different
    ``canonical_config_bytes``), immutable inputs (changed/missing/additional),
    envelope content (tampered stored envelope), public digest (tampered
    ``digest_str``), and fingerprint-envelope version (unsupported/changed).

    Raises :class:`FingerprintMismatch` with a specific reason on any
    difference. The stored envelope's internal digest is also verified against
    its own envelope bytes (catching in-place envelope tampering).
    """
    # 1. Internal envelope integrity: the stored digest_str must match a digest
    #    recomputed from the stored envelope bytes.
    try:
        record.verify_digest()
    except ValueError as e:
        raise FingerprintMismatch(f"Stored envelope digest is inconsistent: {e}") from e

    # 2. Version: the stored envelope version must be the supported one.
    if record.version != FINGERPRINT_VERSION:
        raise FingerprintMismatch(
            f"Fingerprint-envelope version {record.version} is not supported "
            f"(expected {FINGERPRINT_VERSION})."
        )

    # 3. Canonical-config digest: recompute from the supplied bytes and compare.
    recomputed_config_digest = hashlib.sha256(canonical_config_bytes).hexdigest()
    if recomputed_config_digest != record.canonical_config.digest:
        raise FingerprintMismatch(
            "Canonical configuration changed: stored config digest "
            f"{record.canonical_config.digest!r} does not match recomputed "
            f"{recomputed_config_digest!r}."
        )

    # 4. Immutable inputs: recompute the full fingerprint from the supplied
    #    canonical bytes + supplied inputs and compare the public digest.
    recomputed = specification_fingerprint(canonical_config_bytes, immutable_inputs)
    if recomputed.digest_str != record.digest_str:
        # Distinguish input-set differences for a clearer message.
        stored_names = [ii.name for ii in record.immutable_inputs]
        supplied_names = [ii.name for ii in (immutable_inputs or [])]
        if stored_names != supplied_names:
            raise FingerprintMismatch(
                f"Immutable-input set changed: stored names {stored_names!r} vs "
                f"supplied names {sorted(supplied_names)!r}."
            )
        raise FingerprintMismatch(
            "Immutable-input content changed: stored public digest "
            f"{record.digest_str!r} does not match recomputed {recomputed.digest_str!r}."
        )
