"""Tests for run and attempt IDs (Issue #6 decision §2, §3).

Run IDs: ``run-YYYYMMDDtHHMMSSz-<spec-prefix-12>-<20-random-hex>``.
Attempt IDs: ``attempt-YYYYMMDDtHHMMSSz-<20-random-hex>``.
The timestamp is for inspection only; uniqueness comes from 80
cryptographically secure random bits (20 hex). Generation supports injected
clock and entropy providers, collision detection, bounded retries, and a typed
collision-exhaustion error.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timezone

import pytest

from expertforge.identity.ids import (
    IdentityCollisionError,
    attempt_id,
    run_id,
)

RUN_ID_RE = re.compile(r"^run-\d{8}t\d{6}z-[0-9a-f]{12}-[0-9a-f]{20}$")
ATTEMPT_ID_RE = re.compile(r"^attempt-\d{8}t\d{6}z-[0-9a-f]{20}$")


# --- format ----------------------------------------------------------------


class TestRunIdFormat:
    def test_run_id_matches_required_shape(self) -> None:
        rid = run_id(spec_prefix="a" * 12)
        assert RUN_ID_RE.match(rid), rid

    def test_run_id_includes_spec_prefix(self) -> None:
        rid = run_id(spec_prefix="0123456789ab")
        assert "0123456789ab" in rid

    def test_run_id_timestamp_is_utc_zulu(self) -> None:
        fixed = datetime(2026, 7, 29, 14, 30, 12, tzinfo=UTC)
        rid = run_id(spec_prefix="abcdef012345", clock=lambda: fixed)
        assert rid.startswith("run-20260729t143012z-")

    def test_attempt_id_matches_required_shape(self) -> None:
        aid = attempt_id()
        assert ATTEMPT_ID_RE.match(aid), aid


# --- injectable providers --------------------------------------------------


class TestInjectableProviders:
    def test_clock_injection_determines_timestamp(self) -> None:
        fixed = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        rid = run_id(spec_prefix="0123456789ab", clock=lambda: fixed)
        assert rid.startswith("run-20260102t030405z-")

    def test_entropy_injection_determines_suffix(self) -> None:
        # Fixed entropy -> fixed suffix.
        rid_a = run_id(spec_prefix="0123456789ab", entropy=lambda n: bytes(n))
        rid_b = run_id(spec_prefix="0123456789ab", entropy=lambda n: bytes(n))
        assert rid_a == rid_b

    def test_default_entropy_is_random(self) -> None:
        # Default entropy (secrets) -> distinct suffixes across calls.
        fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        ids = {run_id(spec_prefix="0123456789ab", clock=lambda: fixed) for _ in range(50)}
        assert len(ids) == 50  # all distinct

    def test_entropy_provides_requested_byte_count(self) -> None:
        captured: list[int] = []

        def cap_entropy(nbytes: int) -> bytes:
            captured.append(nbytes)
            return bytes(nbytes)

        run_id(spec_prefix="0123456789ab", entropy=cap_entropy)
        # 20 hex chars = 80 bits = 10 bytes.
        assert captured == [10]


# --- collision detection ---------------------------------------------------


class TestCollisionDetection:
    def test_collision_triggers_retry_and_eventually_raises(self) -> None:
        # Force the entropy provider to always return zeros -> every generated
        # suffix collides under a fixed clock.
        fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        with pytest.raises(IdentityCollisionError):
            run_id(
                spec_prefix="0123456789ab",
                clock=lambda: fixed,
                entropy=lambda n: bytes(n),
                max_retries=5,
                exists=lambda _id: True,  # every candidate already exists
            )

    def test_collision_resolves_when_exists_becomes_false(self) -> None:
        fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        attempts = {"n": 0}

        def exists(_id: str) -> bool:
            attempts["n"] += 1
            return attempts["n"] < 3  # first two "exist", third is free

        rid = run_id(
            spec_prefix="0123456789ab",
            clock=lambda: fixed,
            entropy=lambda n: bytes(n),  # constant -> same candidate each retry
            exists=exists,
            max_retries=10,
        )
        assert rid.startswith("run-20260101t000000z-")

    def test_bounded_retries_respected(self) -> None:
        fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        n = {"calls": 0}

        def counting_exists(_id: str) -> bool:
            n["calls"] += 1
            return True

        with pytest.raises(IdentityCollisionError):
            run_id(
                spec_prefix="0123456789ab",
                clock=lambda: fixed,
                entropy=lambda n: bytes(n),
                exists=counting_exists,
                max_retries=4,
            )
        # max_retries candidate generations attempted.
        assert n["calls"] == 4

    def test_no_exists_check_by_default(self) -> None:
        # Default exists=None means no collision check; generation never raises.
        fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        rid = run_id(spec_prefix="0123456789ab", clock=lambda: fixed, entropy=lambda n: bytes(n))
        assert RUN_ID_RE.match(rid)


# --- 80-bit entropy floor --------------------------------------------------


class TestEntropyFloor:
    def test_run_suffix_is_80_bits(self) -> None:
        rid = run_id(spec_prefix="0123456789ab")
        suffix = rid.split("-")[-1]
        assert len(suffix) == 20  # 20 hex = 80 bits

    def test_attempt_suffix_is_80_bits(self) -> None:
        aid = attempt_id()
        suffix = aid.split("-")[-1]
        assert len(suffix) == 20


# --- spec-prefix hex + UTC invariants (review item 5) ---------------------


class TestTrailingNewlineRejection:
    """Regression: `.fullmatch()` (not `.match()` with `$`) must reject values
    ending in a newline, since Python's `$` matches before a final newline."""

    @pytest.mark.parametrize(
        "valid,pattern_call",
        [
            ("run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb", "validate_run_id"),
            ("attempt-20260101t000000z-cccccccccccccccccccc", "validate_attempt_id"),
        ],
    )
    def test_newline_suffix_rejected(self, valid: str, pattern_call: str) -> None:
        from expertforge.identity import ids as ids_mod

        fn = getattr(ids_mod, pattern_call)
        fn(valid)  # base accepts
        with pytest.raises(ValueError):
            fn(valid + "\n")

    def test_spec_prefix_newline_rejected(self) -> None:
        with pytest.raises(ValueError):
            run_id(spec_prefix="0123456789ab\n")

    def test_spec_prefix_carriage_return_rejected(self) -> None:
        with pytest.raises(ValueError):
            run_id(spec_prefix="0123456789ab\r")


class TestSpecPrefixAndUtcInvariants:
    @pytest.mark.parametrize("bad_prefix", ["A" * 12, "g" * 12, "a" * 11, "a" * 13, "", "z" * 12])
    def test_invalid_spec_prefix_rejected(self, bad_prefix: str) -> None:
        with pytest.raises(ValueError):
            run_id(spec_prefix=bad_prefix)

    def test_uppercase_spec_prefix_rejected(self) -> None:
        with pytest.raises(ValueError):
            run_id(spec_prefix="ABCDEF012345")

    def test_naive_clock_rejected(self) -> None:
        from datetime import datetime

        naive = datetime(2026, 1, 1, 0, 0, 0)  # no tzinfo
        with pytest.raises(ValueError):
            run_id(spec_prefix="0123456789ab", clock=lambda: naive)

    def test_naive_clock_rejected_for_attempt(self) -> None:
        from datetime import datetime

        with pytest.raises(ValueError):
            attempt_id(clock=lambda: datetime(2026, 1, 1, 0, 0, 0))

    def test_non_utc_timezone_normalized_to_utc_zulu(self) -> None:
        from datetime import datetime, timedelta

        # +02:00 wall time -> 00:00:00Z; the zulu timestamp must reflect UTC.
        tz_plus2 = timezone(timedelta(hours=2))
        local_midnight = datetime(2026, 1, 1, 2, 0, 0, tzinfo=tz_plus2)
        rid = run_id(spec_prefix="0123456789ab", clock=lambda: local_midnight)
        assert rid.startswith("run-20260101t000000z-")


# --- validate_run_id / validate_attempt_id --------------------------------


class TestIdValidators:
    def test_validate_run_id_accepts_valid(self) -> None:
        from expertforge.identity.ids import validate_run_id

        validate_run_id("run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb")  # no raise

    def test_validate_run_id_rejects_invalid(self) -> None:
        from expertforge.identity.ids import validate_run_id

        with pytest.raises(ValueError):
            validate_run_id("not-a-run-id")

    def test_validate_attempt_id_accepts_valid(self) -> None:
        from expertforge.identity.ids import validate_attempt_id

        validate_attempt_id("attempt-20260101t000000z-cccccccccccccccccccc")  # no raise

    def test_validate_attempt_id_rejects_invalid(self) -> None:
        from expertforge.identity.ids import validate_attempt_id

        with pytest.raises(ValueError):
            validate_attempt_id("run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb")
