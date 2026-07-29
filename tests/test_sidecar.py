"""Tests for the identity sidecar (Issue #6 decision §5; review item 6).

Deterministic JSON written to
``<artifact-root>/<run-id>/attempts/<attempt-id>/run-identity.json`` with
exclusive no-overwrite creation, full-write loop + fsync, partial-file cleanup on
failure, typed loading, version rejection, and fingerprint verification. The
complete sidecar is rendered and UTF-8-validated before any byte is written.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from expertforge.identity.fingerprint import specification_fingerprint
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.identity.sidecar import (
    IdentitySidecarError,
    load_identity_sidecar,
    sidecar_path,
    write_identity_sidecar,
)

_VALID_RUN = "run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb"
_VALID_ATTEMPT = "attempt-20260101t000000z-cccccccccccccccccccc"
_VALID_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _record(
    run_id: str = _VALID_RUN,
    attempt_id: str = _VALID_ATTEMPT,
) -> AttemptIdentityRecord:
    return AttemptIdentityRecord(
        specification_fingerprint=specification_fingerprint(b'{"x":1}'),
        run_id=run_id,
        attempt_id=attempt_id,
        created_at_utc=_VALID_TS,
    )


# --- path layout -----------------------------------------------------------


class TestSidecarPath:
    def test_path_layout_under_artifact_root(self, tmp_path: Path) -> None:
        p = sidecar_path(tmp_path, _VALID_RUN, _VALID_ATTEMPT)
        assert p == tmp_path / _VALID_RUN / "attempts" / _VALID_ATTEMPT / "run-identity.json"


# --- write: no-overwrite + deterministic + durability ---------------------


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
        assert p.read_bytes() == rec.to_deterministic_json()

    def test_write_refuses_to_overwrite_existing(self, tmp_path: Path) -> None:
        rec = _record()
        write_identity_sidecar(tmp_path, rec)
        with pytest.raises(IdentitySidecarError):
            write_identity_sidecar(tmp_path, rec)

    def test_write_rejects_path_traversal_in_ids(self, tmp_path: Path) -> None:
        with pytest.raises(IdentitySidecarError):
            sidecar_path(tmp_path, "run-..-evil", _VALID_ATTEMPT)

    def test_short_write_cleanup_allows_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force os.write to fail; the partial file must be removed so a retry is
        # not permanently blocked by O_EXCL.
        original_write = os.write
        calls = {"n": 0}

        def flaky_write(fd: int, data: bytes) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("simulated write failure")
            return original_write(fd, data)

        monkeypatch.setattr("os.write", flaky_write)
        rec = _record()
        with pytest.raises(IdentitySidecarError):
            write_identity_sidecar(tmp_path, rec)
        # Partial file removed -> a fresh write succeeds.
        monkeypatch.setattr("os.write", original_write)
        p = write_identity_sidecar(tmp_path, rec)
        assert p.exists()


# --- load: typed, version rejection, fingerprint verify, error boundary ----


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
        p.write_text(json.dumps(bad, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        with pytest.raises(IdentitySidecarError):
            load_identity_sidecar(p)

    def test_load_rejects_non_object_json(self, tmp_path: Path) -> None:
        rec = _record()
        p = sidecar_path(tmp_path, rec.run_id, rec.attempt_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        for bad in ["[]", "null", "42", '"string"', "true"]:
            p.write_text(bad, encoding="utf-8")
            with pytest.raises(IdentitySidecarError):
                load_identity_sidecar(p)

    def test_load_rejects_invalid_utf8(self, tmp_path: Path) -> None:
        rec = _record()
        p = sidecar_path(tmp_path, rec.run_id, rec.attempt_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xfe not utf-8")
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
        bad["run_id"] = "invalid-id"  # fails regex
        p.write_text(json.dumps(bad, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        with pytest.raises(IdentitySidecarError):
            load_identity_sidecar(p)

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(IdentitySidecarError):
            load_identity_sidecar(tmp_path / "nope.json")

    def test_load_fingerprint_mismatch_rejected(self, tmp_path: Path) -> None:
        rec = _record()
        write_identity_sidecar(tmp_path, rec)
        with pytest.raises(IdentitySidecarError):
            load_identity_sidecar(
                sidecar_path(tmp_path, rec.run_id, rec.attempt_id),
                expected_fingerprint="spec-v1-sha256-" + "0" * 64,
            )
