"""Tests for the human-readable provenance summary (Issue #7).

A human-readable summary is available without replacing the machine-readable
artifact. It must surface the key facts (identity, source, software, hardware)
without leaking secrets, and degrade gracefully when optional sections are absent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from expertforge.config.resolve import resolve_config
from expertforge.identity.emit import emit_attempt_identity
from expertforge.provenance.record import ProvenanceRecord
from expertforge.provenance.software import capture_software_environment
from expertforge.provenance.summary import format_provenance_summary

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


class TestProvenanceSummary:
    def test_summary_includes_run_and_attempt(self, tmp_path: Path) -> None:
        rec = _record(tmp_path)
        text = format_provenance_summary(rec)
        assert rec.run_id in text
        assert rec.attempt_id in text

    def test_summary_includes_fingerprint_digest(self, tmp_path: Path) -> None:
        rec = _record(tmp_path)
        text = format_provenance_summary(rec)
        # The public fingerprint digest prefix is human-inspectable.
        assert "spec-v1-sha256-" in text

    def test_summary_includes_start_time(self, tmp_path: Path) -> None:
        rec = _record(tmp_path)
        text = format_provenance_summary(rec)
        assert "2026" in text

    def test_summary_with_software_and_hardware(self, tmp_path: Path) -> None:
        from expertforge.provenance.hardware import capture_accelerator, capture_topology

        rec = _record(
            tmp_path,
            software=capture_software_environment(),
            hardware=capture_accelerator(nvidia_smi="/nonexistent/nvidia-smi"),
            topology=capture_topology(),
        )
        text = format_provenance_summary(rec)
        assert "python" in text.lower()
        assert "accelerator" in text.lower()

    def test_summary_degrades_when_optional_sections_absent(self, tmp_path: Path) -> None:
        rec = _record(tmp_path)  # no source/software/hardware/topology
        text = format_provenance_summary(rec)
        # Must not raise; surfaces the absence gracefully.
        assert "not captured" in text.lower() or "—" in text or "n/a" in text.lower()

    def test_summary_does_not_leak_token_patterns(self, tmp_path: Path) -> None:
        rec = _record(tmp_path)
        text = format_provenance_summary(rec)
        # Defense-in-depth: no token-like patterns.
        import re

        for pat in [r"gh[pousr]_[A-Za-z0-9]{36}", r"AKIA[0-9A-Z]{16}"]:
            assert not re.search(pat, text)
