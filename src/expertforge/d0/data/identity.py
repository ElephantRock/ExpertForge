"""Pure identity functions for D0 split, training-order, and validation-order.

All formulas use the exact domain-separated byte strings frozen in the ratified
D0.0 contract (amendment #53 correction comment 5183516544):

- split: ``sha256("expertforge-d0-split-v1\\0" || utf8(document_id))``
- training order: ``sha256("expertforge-d0-order-v1\\0" || uint64_be(epoch) || uint64_be(data_seed) || utf8(document_id))``
- validation order: ``sha256("expertforge-d0-validation-v1\\0" || utf8(document_id))``

These functions are pure, deterministic, and independent of any I/O or
materialization state. Golden-vector tests verify exact SHA-256 digests and
derived values.
"""

from __future__ import annotations

import hashlib
import struct

__all__ = [
    "DATASET_MANIFEST_DIGEST_PREFIX",
    "D0DataSeed",
    "SPLIT_DOMAIN",
    "TRAINING_ORDER_DOMAIN",
    "VALIDATION_ORDER_DOMAIN",
    "split_bucket",
    "is_train",
    "is_validation",
    "training_order_key",
    "validation_order_key",
    "uint64_be",
]

# The frozen data-order seed derived from master seed 2026080200.
# From the qualification config: data_seed_u64 = 4657843784274978659
D0DataSeed: int = 4657843784274978659

# Domain separators with explicit NUL byte (\\x00), frozen by the D0 contract.
SPLIT_DOMAIN = b"expertforge-d0-split-v1\x00"
TRAINING_ORDER_DOMAIN = b"expertforge-d0-order-v1\x00"
VALIDATION_ORDER_DOMAIN = b"expertforge-d0-validation-v1\x00"

# Split constants: u mod 1000 < 995 = train, >= 995 = validation.
SPLIT_MODULUS = 1000
TRAIN_THRESHOLD = 995


def uint64_be(value: int) -> bytes:
    """Encode a non-negative integer as exactly 8 big-endian unsigned bytes.

    Raises ValueError if the value is negative or exceeds uint64 range.
    """
    if value < 0:
        raise ValueError(f"uint64_be requires non-negative value; got {value}")
    if value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"uint64_be value exceeds uint64 range; got {value}")
    return struct.pack(">Q", value)


def _sha256(data: bytes) -> bytes:
    """Raw SHA-256 digest bytes (32 bytes)."""
    return hashlib.sha256(data).digest()


def _first_u64_be(digest: bytes) -> int:
    """Extract the first 8 bytes of ``digest`` as an unsigned big-endian integer."""
    if len(digest) < 8:
        raise ValueError(f"digest must be at least 8 bytes; got {len(digest)}")
    value: int = struct.unpack(">Q", digest[:8])[0]
    return value


def split_bucket(document_id: str) -> int:
    """Derive the split bucket (0..999) for a document.

    Formula:
    ``first_u64_be(sha256("expertforge-d0-split-v1\\0" || utf8(document_id))) mod 1000``
    """
    doc_bytes = document_id.encode("utf-8")
    digest = _sha256(SPLIT_DOMAIN + doc_bytes)
    return _first_u64_be(digest) % SPLIT_MODULUS


def is_train(document_id: str) -> bool:
    """Return True if the document belongs to the training split."""
    return split_bucket(document_id) < TRAIN_THRESHOLD


def is_validation(document_id: str) -> bool:
    """Return True if the document belongs to the validation split."""
    return split_bucket(document_id) >= TRAIN_THRESHOLD


def training_order_key(document_id: str, *, epoch: int, data_seed: int = D0DataSeed) -> bytes:
    """Derive the 32-byte training-order sort key for a document.

    Formula:
    ``sha256("expertforge-d0-order-v1\\0" || uint64_be(epoch) || uint64_be(data_seed) || utf8(document_id))``

    Ascending bytes determine order; document_id is the tiebreaker.
    """
    payload = (
        TRAINING_ORDER_DOMAIN
        + uint64_be(epoch)
        + uint64_be(data_seed)
        + document_id.encode("utf-8")
    )
    return _sha256(payload)


def validation_order_key(document_id: str) -> bytes:
    """Derive the 32-byte validation-order sort key for a document.

    Formula:
    ``sha256("expertforge-d0-validation-v1\\0" || utf8(document_id))``

    Ascending bytes determine order; document_id is the tiebreaker.
    """
    return _sha256(VALIDATION_ORDER_DOMAIN + document_id.encode("utf-8"))


# Re-exported prefix used by the D0 source manifest manifest_sha256 field.
# Not a domain separator; this is the dataset manifest digest string prefix.
DATASET_MANIFEST_DIGEST_PREFIX = "sha256"
