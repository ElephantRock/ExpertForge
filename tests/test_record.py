"""Tests for resume lineage and the per-attempt identity record
(Issue #6 decision §4, §5; review items 2, 5)."""

from __future__ import annotations

from datetime import UTC, datetime, timezone

import pytest
from pydantic import ValidationError

from expertforge.identity.fingerprint import (
    SpecificationFingerprintRecord,
    specification_fingerprint,
)
from expertforge.identity.lineage import ResumeLineage
from expertforge.identity.record import (
    IDENTITY_SCHEMA_VERSION,
    AttemptIdentityRecord,
)

_VALID_RUN = "run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb"
_VALID_ATTEMPT = "attempt-20260101t000000z-cccccccccccccccccccc"
_VALID_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _fingerprint() -> SpecificationFingerprintRecord:
    return specification_fingerprint(b'{"x":1}')


# --- resume lineage --------------------------------------------------------


class TestResumeLineage:
    def test_native_resume_has_all_three_parents(self) -> None:
        lin = ResumeLineage(
            parent_run_id=_VALID_RUN,
            parent_attempt_id=_VALID_ATTEMPT,
            parent_checkpoint_id="ckpt-001",
        )
        assert lin.parent_run_id == _VALID_RUN
        assert lin.parent_attempt_id == _VALID_ATTEMPT
        assert lin.parent_checkpoint_id == "ckpt-001"

    def test_parent_attempt_id_optional_for_imported_legacy(self) -> None:
        lin = ResumeLineage(
            parent_run_id=_VALID_RUN,
            parent_attempt_id=None,
            parent_checkpoint_id="legacy-ckpt",
        )
        assert lin.parent_attempt_id is None

    def test_lineage_is_frozen(self) -> None:
        lin = ResumeLineage(
            parent_run_id=_VALID_RUN,
            parent_attempt_id=_VALID_ATTEMPT,
            parent_checkpoint_id="ckpt-z",
        )
        with pytest.raises(ValidationError):
            lin.parent_run_id = "other"  # type: ignore[misc]

    def test_parent_run_id_required(self) -> None:
        with pytest.raises(ValidationError):
            ResumeLineage(  # type: ignore[call-arg]
                parent_attempt_id=_VALID_ATTEMPT,
                parent_checkpoint_id="ckpt-z",
            )

    @pytest.mark.parametrize(
        "bad_run", ["run-x", "Run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb", "x"]
    )
    def test_invalid_parent_run_id_rejected(self, bad_run: str) -> None:
        with pytest.raises(ValidationError):
            ResumeLineage(
                parent_run_id=bad_run, parent_attempt_id=_VALID_ATTEMPT, parent_checkpoint_id="c"
            )

    def test_invalid_parent_attempt_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResumeLineage(
                parent_run_id=_VALID_RUN, parent_attempt_id="bad", parent_checkpoint_id="c"
            )


# --- identity record -------------------------------------------------------


def _make_record(
    *,
    run_id: str = _VALID_RUN,
    attempt_id: str = _VALID_ATTEMPT,
    lineage: ResumeLineage | None = None,
    created_at: datetime = _VALID_TS,
) -> AttemptIdentityRecord:
    return AttemptIdentityRecord(
        specification_fingerprint=_fingerprint(),
        run_id=run_id,
        attempt_id=attempt_id,
        created_at_utc=created_at,
        lineage=lineage,
    )


class TestAttemptIdentityRecord:
    def test_record_carries_schema_version(self) -> None:
        assert _make_record().identity_schema_version == IDENTITY_SCHEMA_VERSION

    def test_record_carries_typed_fingerprint(self) -> None:
        rec = _make_record()
        # The fingerprint is the full reconstructable record, not a string.
        assert isinstance(rec.specification_fingerprint, SpecificationFingerprintRecord)
        assert rec.fingerprint_digest_str().startswith("spec-v1-sha256-")

    def test_record_null_lineage_for_first_attempt(self) -> None:
        assert _make_record(lineage=None).lineage is None

    def test_record_with_lineage(self) -> None:
        lin = ResumeLineage(
            parent_run_id=_VALID_RUN,
            parent_attempt_id=_VALID_ATTEMPT,
            parent_checkpoint_id="ckpt-parent",
        )
        rec = _make_record(lineage=lin)
        assert rec.lineage is not None
        assert rec.lineage.parent_run_id == _VALID_RUN

    def test_record_is_frozen(self) -> None:
        rec = _make_record()
        with pytest.raises(ValidationError):
            rec.run_id = "other"  # type: ignore[misc]

    def test_record_serializes_to_deterministic_json(self) -> None:
        a = _make_record().to_deterministic_json()
        b = _make_record().to_deterministic_json()
        assert a == b
        assert b", " not in a
        assert b": " not in a

    def test_record_round_trips_through_json(self) -> None:
        rec = _make_record()
        restored = AttemptIdentityRecord.model_validate_json(rec.to_deterministic_json())
        assert restored == rec

    @pytest.mark.parametrize(
        "bad_run", ["", "run-x", "Run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb"]
    )
    def test_invalid_run_id_rejected(self, bad_run: str) -> None:
        with pytest.raises(ValidationError):
            _make_record(run_id=bad_run)

    @pytest.mark.parametrize("bad_attempt", ["", "attempt-x", "x"])
    def test_invalid_attempt_id_rejected(self, bad_attempt: str) -> None:
        with pytest.raises(ValidationError):
            _make_record(attempt_id=bad_attempt)

    def test_naive_created_at_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_record(created_at=datetime(2026, 1, 1, 0, 0, 0))  # naive

    def test_non_utc_aware_created_at_rejected(self) -> None:
        # A timezone-aware but non-UTC offset must be rejected, not silently
        # normalized to UTC.
        from datetime import timedelta

        non_utc = timezone(timedelta(hours=2))
        with pytest.raises(ValidationError):
            _make_record(created_at=datetime(2026, 1, 1, 0, 0, 0, tzinfo=non_utc))

    def test_utc_created_at_accepted(self) -> None:
        rec = _make_record(created_at=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC))
        assert rec.created_at_utc == datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_unknown_schema_version_rejected_on_construction(self) -> None:
        rec = _make_record()
        bad = rec.model_dump()
        bad["identity_schema_version"] = 999
        with pytest.raises((ValidationError, ValueError)):
            AttemptIdentityRecord.model_validate(bad, strict=False)

    def test_lineage_does_not_alter_fingerprint(self) -> None:
        lin = ResumeLineage(
            parent_run_id=_VALID_RUN,
            parent_attempt_id=_VALID_ATTEMPT,
            parent_checkpoint_id="ckpt-parent",
        )
        assert (
            _make_record().specification_fingerprint
            == _make_record(lineage=lin).specification_fingerprint
        )
