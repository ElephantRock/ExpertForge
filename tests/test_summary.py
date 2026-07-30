"""Tests for the human-readable provenance summary (Issue #7).

A human-readable summary is available without replacing the machine-readable
artifact. It must surface the key facts (identity, source, software, hardware)
without leaking secrets, and degrade gracefully when optional sections are absent.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from expertforge.config.resolve import resolve_config
from expertforge.identity.emit import emit_attempt_identity
from expertforge.provenance.record import ProvenanceRecord
from expertforge.provenance.software import capture_software_environment
from expertforge.provenance.source_snapshot import (
    capture_source_snapshot,
    source_snapshot_immutable_input,
)
from expertforge.provenance.summary import format_provenance_summary

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
_FIXED = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _init_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)
    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


def _record(tmp_path: Path, **sections: object) -> ProvenanceRecord:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _init_repo(repo)
    snap = capture_source_snapshot(repo)
    ident, _ = emit_attempt_identity(
        artifact_root=tmp_path,
        config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
        immutable_inputs=[source_snapshot_immutable_input(snap)],
        clock=lambda: _FIXED,
        entropy=lambda n: bytes(n),
    )
    return ProvenanceRecord.from_identity(ident, source=snap, **sections)  # type: ignore[arg-type]


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
        rec = _record(tmp_path)  # source is mandatory; software/hardware/topology absent
        text = format_provenance_summary(rec)
        # Must not raise; surfaces the absence gracefully. With the typed
        # schema, absent sections render as honest unavailable/unknown markers
        # rather than a literal "not captured" line.
        lower = text.lower()
        assert "unavailable" in lower or "unknown" in lower or "not_applicable" in lower

    def test_summary_does_not_leak_token_patterns(self, tmp_path: Path) -> None:
        rec = _record(tmp_path)
        text = format_provenance_summary(rec)
        # Defense-in-depth: no token-like patterns.
        import re

        for pat in [r"gh[pousr]_[A-Za-z0-9]{36}", r"AKIA[0-9A-Z]{16}"]:
            assert not re.search(pat, text)
