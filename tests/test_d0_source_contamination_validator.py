from __future__ import annotations

import json
from pathlib import Path

import pytest

from expertforge.d0.errors import D0PreflightError
from expertforge.d0.scan_report import build_scan_report, canonical_report_bytes
from scripts.validate_d0_source_contamination import (
    main,
    validate_actual_report,
    validate_portable,
)


def test_portable_validator_never_claims_actual_scan_acceptance() -> None:
    report = validate_portable()

    assert report["status"] == "valid_d0_source_contamination_implementation"
    assert report["actual_corpus_scan_completed"] is False
    assert report["actual_corpus_scan_accepted"] is False
    assert report["d0_1_accepted"] is False
    assert report["dataset_shard_count"] == 14
    assert report["dataset_total_size_bytes"] == 28_518_193_415


def test_fixture_report_cannot_satisfy_actual_validator(tmp_path: Path) -> None:
    report = build_scan_report(
        dataset_manifest_sha256=(
            "85fab524b13c49da78264eb03124ce113395657d6403627e53008aade7eea7d7"
        ),
        tokenizer_manifest_sha256=(
            "eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c"
        ),
        prompt_manifest_sha256=(
            "c748706d9706e42bc0147d62c80f8f486ebb6aaa72fb53140b491ce7706e18c1"
        ),
        prompt_payload_sha256=(
            "c52ef9f4420ff8160fd5f212370f46cefde71033ac38431bdad2878640070e51"
        ),
        scanner_source_commit="1" * 40,
        scanner_algorithm_version="d0-contamination-aho-corasick-v1",
        runtime_versions={"expertforge": "fixture", "pyarrow": "25.0.0", "python": "3.11"},
        expected_shard_count=0,
        expected_total_size_bytes=0,
        shards=(),
        hits=(),
        scan_complete=True,
        terminal_diagnostic="synthetic fixture",
    )
    path = tmp_path / "fixture-report.json"
    path.write_bytes(canonical_report_bytes(report))

    with pytest.raises(D0PreflightError, match="fixture reports"):
        validate_actual_report(path)


def test_actual_validator_rejects_noncanonical_report_bytes(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"report_sha256": "0" * 64}, indent=2), encoding="utf-8")

    with pytest.raises(D0PreflightError):
        validate_actual_report(path)


def test_portable_command_succeeds_without_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "valid_d0_source_contamination_implementation" in captured.out
    assert captured.err == ""
