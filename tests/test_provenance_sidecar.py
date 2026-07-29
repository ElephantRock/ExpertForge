"""Tests for the provenance sidecar (Issue #7 decision: artifact ownership).

Deterministic JSON written to
``<artifact-root>/<run-id>/attempts/<attempt-id>/run-provenance.json`` with
exclusive publication, full-write handling, fsync, failure cleanup, typed load
and verification. Mirrors the #6 identity sidecar discipline but for provenance.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from expertforge.config.resolve import resolve_config
from expertforge.identity.emit import emit_attempt_identity
from expertforge.provenance.record import PROVENANCE_SCHEMA_VERSION, ProvenanceRecord
from expertforge.provenance.sidecar import (
    ProvenanceSidecarError,
    load_provenance_sidecar,
    provenance_sidecar_path,
    write_provenance_sidecar,
)
from expertforge.provenance.software import capture_software_environment

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
_FIXED = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _record(tmp_path: Path, **sections: object) -> ProvenanceRecord:
    ident, _ = emit_attempt_identity(
        artifact_root=tmp_path,
        config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
        clock=lambda: _FIXED,
        entropy=lambda n: bytes(n),
    )
    return ProvenanceRecord.from_identity(ident, **sections)  # type: ignore[arg-type]


# --- path layout ----------------------------------------------------------


class TestProvenanceSidecarPath:
    def test_path_layout(self, tmp_path: Path) -> None:
        p = provenance_sidecar_path(
            tmp_path,
            "run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb",
            "attempt-20260101t000000z-cccccccccccccccccccc",
        )
        assert p.name == "run-provenance.json"
        assert "attempts" in p.parts

    def test_path_rejects_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(ProvenanceSidecarError):
            provenance_sidecar_path(tmp_path, "run-..-evil", "attempt-x")


# --- write: deterministic + exclusive + durable ---------------------------


class TestProvenanceSidecarWrite:
    def test_write_creates_file_at_correct_path(self, tmp_path: Path) -> None:
        rec = _record(tmp_path)
        p = write_provenance_sidecar(tmp_path, rec)
        assert p == provenance_sidecar_path(tmp_path, rec.run_id, rec.attempt_id)
        assert p.exists()

    def test_write_is_deterministic_compact_sorted(self, tmp_path: Path) -> None:
        rec = _record(tmp_path, software=capture_software_environment())
        p = write_provenance_sidecar(tmp_path, rec)
        text = p.read_text(encoding="utf-8")
        assert ", " not in text
        assert ": " not in text
        assert p.read_bytes() == rec.to_deterministic_json()

    def test_write_refuses_to_overwrite(self, tmp_path: Path) -> None:
        rec = _record(tmp_path)
        write_provenance_sidecar(tmp_path, rec)
        with pytest.raises(ProvenanceSidecarError):
            write_provenance_sidecar(tmp_path, rec)

    def test_close_failure_cleans_up_and_allows_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Build the record first (writes the identity sidecar with the real
        # os.close), then patch os.close only for the provenance write.
        rec = _record(tmp_path)
        original_close = os.close
        calls = {"n": 0}

        def flaky_close(fd: int) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("simulated close failure")
            original_close(fd)

        monkeypatch.setattr("os.close", flaky_close)
        target = provenance_sidecar_path(tmp_path, rec.run_id, rec.attempt_id)
        with pytest.raises(ProvenanceSidecarError):
            write_provenance_sidecar(tmp_path, rec)
        assert not target.exists()
        monkeypatch.setattr("os.close", original_close)
        p = write_provenance_sidecar(tmp_path, rec)
        assert p.exists()
        assert p.exists()


# --- load: typed, version rejection, error boundary -----------------------


class TestProvenanceSidecarLoad:
    def test_load_round_trips(self, tmp_path: Path) -> None:
        rec = _record(tmp_path, software=capture_software_environment())
        write_provenance_sidecar(tmp_path, rec)
        loaded = load_provenance_sidecar(
            provenance_sidecar_path(tmp_path, rec.run_id, rec.attempt_id)
        )
        assert loaded == rec

    def test_load_rejects_unknown_schema_version(self, tmp_path: Path) -> None:
        rec = _record(tmp_path)
        p = provenance_sidecar_path(tmp_path, rec.run_id, rec.attempt_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        bad = json.loads(rec.to_deterministic_json())
        bad["provenance_schema_version"] = 999
        p.write_text(json.dumps(bad, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        with pytest.raises(ProvenanceSidecarError):
            load_provenance_sidecar(p)

    def test_load_rejects_non_object_json(self, tmp_path: Path) -> None:
        rec = _record(tmp_path)
        p = provenance_sidecar_path(tmp_path, rec.run_id, rec.attempt_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        for bad in ["[]", "null", "42", '"x"']:
            p.write_text(bad, encoding="utf-8")
            with pytest.raises(ProvenanceSidecarError):
                load_provenance_sidecar(p)

    def test_load_rejects_invalid_utf8(self, tmp_path: Path) -> None:
        rec = _record(tmp_path)
        p = provenance_sidecar_path(tmp_path, rec.run_id, rec.attempt_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xfe bad")
        with pytest.raises(ProvenanceSidecarError):
            load_provenance_sidecar(p)

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ProvenanceSidecarError):
            load_provenance_sidecar(tmp_path / "nope.json")


# --- two version fields ---------------------------------------------------


class TestProvenanceVersioning:
    def test_record_carries_provenance_schema_version(self, tmp_path: Path) -> None:
        rec = _record(tmp_path)
        assert rec.provenance_schema_version == PROVENANCE_SCHEMA_VERSION
