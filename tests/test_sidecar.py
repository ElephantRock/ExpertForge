"""Tests for the identity sidecar (Issue #6 decision §5).

Deterministic JSON written to
``<artifact-root>/<run-id>/attempts/<attempt-id>/run-identity.json`` with
exclusive no-overwrite creation, typed loading, version rejection, and
fingerprint verification. The complete sidecar is rendered and UTF-8-validated
before any byte is written.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from expertforge.identity.fingerprint import specification_fingerprint
from expertforge.identity.record import IDENTITY_SCHEMA_VERSION, AttemptIdentityRecord
from expertforge.identity.sidecar import (
    IdentitySidecarError,
    load_identity_sidecar,
    sidecar_path,
    write_identity_sidecar,
)


def _record(
    run_id: str = "run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb",
    attempt_id: str = "attempt-20260101t000000z-cccccccccccccccccccc",
    fingerprint: str | None = None,
) -> AttemptIdentityRecord:
    fp = fingerprint or specification_fingerprint(b'{"x":1}').digest_str
    return AttemptIdentityRecord(
        specification_fingerprint=fp,
        run_id=run_id,
        attempt_id=attempt_id,
        created_at_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
    )


# --- path layout -----------------------------------------------------------


class TestSidecarPath:
    def test_path_layout_under_artifact_root(self, tmp_path: Path) -> None:
        p = sidecar_path(tmp_path, "run-x", "attempt-y")
        assert p == tmp_path / "run-x" / "attempts" / "attempt-y" / "run-identity.json"


# --- write: no-overwrite + deterministic -----------------------------------


class TestSidecarWrite:
    def test_write_creates_file_at_correct_path(self, tmp_path: Path) -> None:
        rec = _record()
        p = write_identity_sidecar(tmp_path, rec)
        assert p == sidecar_path(tmp_path, rec.run_id, rec.attempt_id)
        assert p.exists()

    def test_write_is_deterministic_compact_sorted(self, tmp_path: Path) -> None:
        rec = _record()
        p = write_identity_sidecar(tmp_path, rec)
        text = p.read_text(encoding="utf-8")
        assert ", " not in text
        assert ": " not in text
        # Re-serialize the record and compare byte-for-byte.
        assert p.read_bytes() == rec.to_deterministic_json()

    def test_write_refuses_to_overwrite_existing(self, tmp_path: Path) -> None:
        rec = _record()
        write_identity_sidecar(tmp_path, rec)
        with pytest.raises(IdentitySidecarError):
            write_identity_sidecar(tmp_path, rec)

    def test_write_rejects_path_traversal_in_ids(self, tmp_path: Path) -> None:
        # run_id / attempt_id must not escape the artifact root.
        bad = _record(run_id="run-..-evil")
        with pytest.raises(IdentitySidecarError):
            write_identity_sidecar(tmp_path, bad)


# --- load: typed, version rejection, fingerprint verify --------------------


class TestSidecarLoad:
    def test_load_round_trips_record(self, tmp_path: Path) -> None:
        rec = _record()
        write_identity_sidecar(tmp_path, rec)
        loaded = load_identity_sidecar(sidecar_path(tmp_path, rec.run_id, rec.attempt_id))
        assert loaded == rec

    def test_load_rejects_unknown_schema_version(self, tmp_path: Path) -> None:
        rec = _record()
        p = sidecar_path(tmp_path, rec.run_id, rec.attempt_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        bad = json.loads(rec.to_deterministic_json())
        bad["identity_schema_version"] = 999
        # Render fully before writing (mirrors production discipline).
        p.write_text(json.dumps(bad, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        with pytest.raises(IdentitySidecarError):
            load_identity_sidecar(p)

    def test_load_rejects_malformed_json(self, tmp_path: Path) -> None:
        rec = _record()
        p = sidecar_path(tmp_path, rec.run_id, rec.attempt_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(IdentitySidecarError):
            load_identity_sidecar(p)

    def test_load_rejects_validation_failure(self, tmp_path: Path) -> None:
        rec = _record()
        p = sidecar_path(tmp_path, rec.run_id, rec.attempt_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        bad = json.loads(rec.to_deterministic_json())
        bad["run_id"] = ""  # invalid (min_length=1)
        p.write_text(json.dumps(bad, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        with pytest.raises(IdentitySidecarError):
            load_identity_sidecar(p)

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(IdentitySidecarError):
            load_identity_sidecar(tmp_path / "nope.json")


# --- versioning ------------------------------------------------------------


class TestVersioning:
    def test_two_independent_versions(self) -> None:
        # The record carries identity_schema_version; the fingerprint carries its
        # own envelope version. They are independent integers.
        from expertforge.identity.fingerprint import FINGERPRINT_VERSION

        assert IDENTITY_SCHEMA_VERSION == 1
        assert FINGERPRINT_VERSION == 1
        # They are conceptually independent even if numerically equal today.
        assert isinstance(IDENTITY_SCHEMA_VERSION, int)
        assert isinstance(FINGERPRINT_VERSION, int)
