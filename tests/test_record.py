"""Tests for resume lineage and the per-attempt identity record
(Issue #6 decision §4, §5)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from expertforge.identity.fingerprint import specification_fingerprint
from expertforge.identity.lineage import ResumeLineage
from expertforge.identity.record import (
    IDENTITY_SCHEMA_VERSION,
    AttemptIdentityRecord,
)


def _fingerprint_digest() -> str:
    # Any stable canonical bytes; reuse a small constant for the record tests.
    return specification_fingerprint(b'{"x":1}').digest_str


# --- resume lineage --------------------------------------------------------


class TestResumeLineage:
    def test_native_resume_has_all_three_parents(self) -> None:
        lin = ResumeLineage(
            parent_run_id="run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb",
            parent_attempt_id="attempt-20260101t000000z-aaaaaaaaaaaaaaaaaaaa",
            parent_checkpoint_id="ckpt-001",
        )
        assert lin.parent_run_id.startswith("run-")
        assert lin.parent_attempt_id is not None
        assert lin.parent_attempt_id.startswith("attempt-")
        assert lin.parent_checkpoint_id == "ckpt-001"

    def test_parent_attempt_id_optional_for_imported_legacy(self) -> None:
        lin = ResumeLineage(
            parent_run_id="run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb",
            parent_attempt_id=None,
            parent_checkpoint_id="legacy-ckpt",
        )
        assert lin.parent_attempt_id is None

    def test_lineage_is_frozen(self) -> None:
        lin = ResumeLineage(
            parent_run_id="run-x",
            parent_attempt_id="attempt-y",
            parent_checkpoint_id="ckpt-z",
        )
        with pytest.raises(ValidationError):
            lin.parent_run_id = "other"  # type: ignore[misc]

    def test_parent_run_id_required(self) -> None:
        with pytest.raises(ValidationError):
            ResumeLineage(  # type: ignore[call-arg]
                parent_attempt_id="attempt-y",
                parent_checkpoint_id="ckpt-z",
            )


# --- identity record -------------------------------------------------------


class TestAttemptIdentityRecord:
    def _make(self, lineage: ResumeLineage | None = None) -> AttemptIdentityRecord:
        return AttemptIdentityRecord(
            specification_fingerprint=_fingerprint_digest(),
            run_id="run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb",
            attempt_id="attempt-20260101t000000z-cccccccccccccccccccc",
            created_at_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
            lineage=lineage,
        )

    def test_record_carries_schema_version(self) -> None:
        rec = self._make()
        assert rec.identity_schema_version == IDENTITY_SCHEMA_VERSION

    def test_record_null_lineage_for_first_attempt(self) -> None:
        rec = self._make(lineage=None)
        assert rec.lineage is None

    def test_record_with_lineage(self) -> None:
        lin = ResumeLineage(
            parent_run_id="run-parent",
            parent_attempt_id="attempt-parent",
            parent_checkpoint_id="ckpt-parent",
        )
        rec = self._make(lineage=lin)
        assert rec.lineage is not None
        assert rec.lineage.parent_run_id == "run-parent"

    def test_record_is_frozen(self) -> None:
        rec = self._make()
        with pytest.raises(ValidationError):
            rec.run_id = "other"  # type: ignore[misc]

    def test_record_serializes_to_deterministic_json(self) -> None:
        rec = self._make()
        a = rec.to_deterministic_json()
        b = self._make().to_deterministic_json()
        assert a == b
        # Compact, sorted.
        assert b", " not in a
        assert b": " not in a

    def test_lineage_does_not_alter_fingerprint(self) -> None:
        # Same spec + run + attempt but different lineage -> fingerprint unchanged.
        base = self._make()
        lin = ResumeLineage(
            parent_run_id="run-parent",
            parent_attempt_id="attempt-parent",
            parent_checkpoint_id="ckpt-parent",
        )
        with_lineage = self._make(lineage=lin)
        assert base.specification_fingerprint == with_lineage.specification_fingerprint
