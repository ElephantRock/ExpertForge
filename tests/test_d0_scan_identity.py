from __future__ import annotations

import json
from pathlib import Path

import pytest

from expertforge.d0.errors import SourceOrderError
from expertforge.d0.scan_identity import ScanIdentity, bind_scan_identity


def _identity(*, source_commit: str = "6" * 40) -> ScanIdentity:
    return ScanIdentity(
        dataset_manifest_sha256="1" * 64,
        tokenizer_manifest_sha256="2" * 64,
        prompt_manifest_sha256="3" * 64,
        prompt_payload_sha256="4" * 64,
        scanner_algorithm_version="d0-contamination-aho-corasick-v1",
        scanner_source_commit=source_commit,
    )


def test_scan_identity_is_created_and_exactly_reusable(tmp_path: Path) -> None:
    path = tmp_path / "scan-identity.json"
    first = bind_scan_identity(path, _identity())
    second = bind_scan_identity(path, _identity())

    assert first == second
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["scan_identity_sha256"] == first
    assert not list(tmp_path.glob("*.part"))


def test_scan_identity_rejects_restart_under_different_source_commit(tmp_path: Path) -> None:
    path = tmp_path / "scan-identity.json"
    bind_scan_identity(path, _identity())

    with pytest.raises(SourceOrderError, match="different scan identity"):
        bind_scan_identity(path, _identity(source_commit="7" * 40))


def test_scan_identity_rejects_semantic_mutation(tmp_path: Path) -> None:
    path = tmp_path / "scan-identity.json"
    bind_scan_identity(path, _identity())
    value = json.loads(path.read_text(encoding="utf-8"))
    value["scanner_algorithm_version"] = "changed"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(SourceOrderError, match="digest mismatch"):
        bind_scan_identity(path, _identity())
