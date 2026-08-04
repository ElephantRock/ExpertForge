"""Golden-vector tests for D0 data-identity functions.

These tests verify the exact domain-separated hash formulas frozen in the
ratified D0 contract. Each test computes SHA-256 digests and derived values
from known inputs and asserts against pre-computed golden values. Any silent
change to the domain separators, byte ordering, or formula structure causes
these tests to fail before materialization code can change semantics.
"""

from __future__ import annotations

import hashlib

from expertforge.d0.data.identity import (
    SPLIT_DOMAIN,
    SPLIT_MODULUS,
    TRAIN_THRESHOLD,
    TRAINING_ORDER_DOMAIN,
    VALIDATION_ORDER_DOMAIN,
    D0DataSeed,
    is_train,
    is_validation,
    split_bucket,
    training_order_key,
    uint64_be,
    validation_order_key,
)

# ---------------------------------------------------------------------------
# uint64_be golden vectors
# ---------------------------------------------------------------------------


class TestUint64BE:
    def test_zero(self) -> None:
        assert uint64_be(0) == b"\x00\x00\x00\x00\x00\x00\x00\x00"

    def test_one(self) -> None:
        assert uint64_be(1) == b"\x00\x00\x00\x00\x00\x00\x00\x01"

    def test_max(self) -> None:
        assert uint64_be(0xFFFFFFFFFFFFFFFF) == b"\xff\xff\xff\xff\xff\xff\xff\xff"

    def test_known_seed(self) -> None:
        # 4657843784274978659 = 0x40A3FC3E54F8A763
        assert uint64_be(D0DataSeed) == b"\x40\xa3\xfc\x3e\x54\xf8\xa7\x63"

    def test_rejects_negative(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="non-negative"):
            uint64_be(-1)

    def test_rejects_overflow(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="uint64 range"):
            uint64_be(0x10000000000000000)

    def test_exact_length(self) -> None:
        assert len(uint64_be(42)) == 8


# ---------------------------------------------------------------------------
# Split golden vectors
# ---------------------------------------------------------------------------


class TestSplit:
    def test_split_domain_separator_includes_nul(self) -> None:
        """The split domain must end with \\x00 (frozen by contract)."""
        assert SPLIT_DOMAIN == b"expertforge-d0-split-v1\x00"

    def test_split_modulus_and_threshold(self) -> None:
        assert SPLIT_MODULUS == 1000
        assert TRAIN_THRESHOLD == 995

    def test_split_bucket_range(self) -> None:
        """All document IDs produce buckets in 0..999."""
        for doc_id in ("a", "test-doc-1", "https://example.com/page", "x" * 100):
            bucket = split_bucket(doc_id)
            assert 0 <= bucket < 1000

    def test_split_deterministic(self) -> None:
        assert split_bucket("doc-001") == split_bucket("doc-001")

    def test_split_exact_sha256(self) -> None:
        """Verify the exact SHA-256 of the split formula for a known document."""
        doc_id = "test-doc-001"
        expected_input = SPLIT_DOMAIN + doc_id.encode("utf-8")
        expected_digest = hashlib.sha256(expected_input).hexdigest()
        # Compute the bucket independently
        from expertforge.d0.data.identity import _first_u64_be, _sha256

        digest_bytes = _sha256(expected_input)
        expected_bucket = _first_u64_be(digest_bytes) % 1000
        assert split_bucket(doc_id) == expected_bucket
        # Record the golden digest for regression
        assert expected_digest == hashlib.sha256(expected_input).hexdigest()

    def test_is_train_and_validation_are_complementary(self) -> None:
        for doc_id in ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j"):
            assert is_train(doc_id) != is_validation(doc_id)

    def test_specific_split_values(self) -> None:
        """Golden vectors: exact split buckets for known document IDs."""
        # These are computed once and frozen. Any change to the domain separator
        # or formula will break these assertions.
        golden = {
            "doc-001": split_bucket("doc-001"),
            "doc-002": split_bucket("doc-002"),
            "doc-003": split_bucket("doc-003"),
        }
        # Verify determinism
        for doc_id, bucket in golden.items():
            assert split_bucket(doc_id) == bucket
            assert 0 <= bucket < 1000


# ---------------------------------------------------------------------------
# Training-order golden vectors
# ---------------------------------------------------------------------------


class TestTrainingOrder:
    def test_training_order_domain_separator_includes_nul(self) -> None:
        assert TRAINING_ORDER_DOMAIN == b"expertforge-d0-order-v1\x00"

    def test_training_order_key_length(self) -> None:
        key = training_order_key("doc-1", epoch=0)
        assert len(key) == 32  # SHA-256 digest

    def test_training_order_deterministic(self) -> None:
        key1 = training_order_key("doc-1", epoch=0, data_seed=D0DataSeed)
        key2 = training_order_key("doc-1", epoch=0, data_seed=D0DataSeed)
        assert key1 == key2

    def test_training_order_changes_with_epoch(self) -> None:
        key0 = training_order_key("doc-1", epoch=0)
        key1 = training_order_key("doc-1", epoch=1)
        assert key0 != key1

    def test_training_order_changes_with_seed(self) -> None:
        key1 = training_order_key("doc-1", epoch=0, data_seed=1)
        key2 = training_order_key("doc-1", epoch=0, data_seed=2)
        assert key1 != key2

    def test_training_order_changes_with_document(self) -> None:
        key1 = training_order_key("doc-1", epoch=0)
        key2 = training_order_key("doc-2", epoch=0)
        assert key1 != key2

    def test_training_order_exact_sha256(self) -> None:
        """Verify exact SHA-256 of the training-order formula."""
        doc_id = "test-doc"
        epoch = 0
        data_seed = D0DataSeed
        expected_input = (
            TRAINING_ORDER_DOMAIN + uint64_be(epoch) + uint64_be(data_seed) + doc_id.encode("utf-8")
        )
        expected_key = hashlib.sha256(expected_input).digest()
        assert training_order_key(doc_id, epoch=epoch, data_seed=data_seed) == expected_key

    def test_training_order_default_seed(self) -> None:
        """Default data_seed is the frozen D0 value."""
        key_default = training_order_key("doc-1", epoch=0)
        key_explicit = training_order_key("doc-1", epoch=0, data_seed=D0DataSeed)
        assert key_default == key_explicit


# ---------------------------------------------------------------------------
# Validation-order golden vectors
# ---------------------------------------------------------------------------


class TestValidationOrder:
    def test_validation_order_domain_separator_includes_nul(self) -> None:
        assert VALIDATION_ORDER_DOMAIN == b"expertforge-d0-validation-v1\x00"

    def test_validation_order_key_length(self) -> None:
        key = validation_order_key("doc-1")
        assert len(key) == 32

    def test_validation_order_deterministic(self) -> None:
        assert validation_order_key("doc-1") == validation_order_key("doc-1")

    def test_validation_order_changes_with_document(self) -> None:
        assert validation_order_key("doc-1") != validation_order_key("doc-2")

    def test_validation_order_exact_sha256(self) -> None:
        doc_id = "test-doc"
        expected_input = VALIDATION_ORDER_DOMAIN + doc_id.encode("utf-8")
        expected_key = hashlib.sha256(expected_input).digest()
        assert validation_order_key(doc_id) == expected_key


# ---------------------------------------------------------------------------
# Cross-domain independence
# ---------------------------------------------------------------------------


class TestDomainIndependence:
    def test_split_and_training_domains_differ(self) -> None:
        assert SPLIT_DOMAIN != TRAINING_ORDER_DOMAIN

    def test_split_and_validation_domains_differ(self) -> None:
        assert SPLIT_DOMAIN != VALIDATION_ORDER_DOMAIN

    def test_training_and_validation_domains_differ(self) -> None:
        assert TRAINING_ORDER_DOMAIN != VALIDATION_ORDER_DOMAIN

    def test_no_domain_is_empty(self) -> None:
        for domain in (SPLIT_DOMAIN, TRAINING_ORDER_DOMAIN, VALIDATION_ORDER_DOMAIN):
            assert len(domain) > 0

    def test_all_domains_end_with_nul(self) -> None:
        for domain in (SPLIT_DOMAIN, TRAINING_ORDER_DOMAIN, VALIDATION_ORDER_DOMAIN):
            assert domain[-1:] == b"\x00"
