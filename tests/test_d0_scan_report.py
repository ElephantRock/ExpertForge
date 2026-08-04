from __future__ import annotations

import copy

import pytest

from expertforge.d0.contamination import HitRecord
from expertforge.d0.errors import ScanReportError
from expertforge.d0.scan_report import (
    ShardScanStats,
    build_scan_report,
    canonical_report_bytes,
)


def _shard(*, complete: bool = True) -> ShardScanStats:
    return ShardScanStats(
        path="sample/000.parquet",
        sha256="1" * 64,
        size_bytes=4,
        physical_rows_visited=3,
        accepted_normalized_documents=2,
        rejected_documents=1,
        duplicates_suppressed=1,
        unique_documents_scanned=1,
        normalized_utf8_bytes_scanned=5,
        normalized_codepoints_scanned=5,
        complete=complete,
    )


def _build(
    *,
    shards: tuple[ShardScanStats, ...],
    hits: tuple[HitRecord, ...] = (),
    scan_complete: bool,
) -> dict[str, object]:
    return build_scan_report(
        dataset_manifest_sha256="2" * 64,
        tokenizer_manifest_sha256="3" * 64,
        prompt_manifest_sha256="4" * 64,
        prompt_payload_sha256="5" * 64,
        scanner_source_commit="6" * 40,
        scanner_algorithm_version="d0-contamination-aho-corasick-v1",
        runtime_versions={"expertforge": "0.0.1", "python": "3.11.13"},
        expected_shard_count=1,
        expected_total_size_bytes=4,
        shards=shards,
        hits=hits,
        scan_complete=scan_complete,
        terminal_diagnostic="scan completed" if scan_complete else "scan interrupted",
    )


def test_complete_zero_hit_report_is_accepted_and_content_addressed() -> None:
    report = _build(shards=(_shard(),), scan_complete=True)
    assert report["status"] == "complete_zero_hit"
    assert report["accepted"] is True
    assert canonical_report_bytes(report)


def test_incomplete_report_cannot_claim_acceptance() -> None:
    report = _build(shards=(_shard(complete=False),), scan_complete=False)
    assert report["status"] == "incomplete"
    assert report["accepted"] is False


def test_complete_hit_report_is_truthful_failure_evidence() -> None:
    hit = HitRecord(
        prompt_id="d0-gen-01",
        check="exact_normalized_probe_substring",
        document_id="doc-1",
        source_file_path="sample/000.parquet",
        physical_row_index=0,
        match_start_codepoint=10,
        match_end_codepoint=74,
    )
    report = _build(shards=(_shard(),), hits=(hit,), scan_complete=True)
    assert report["status"] == "complete_hits_detected"
    assert report["accepted"] is False


def test_scan_complete_rejects_partial_inventory() -> None:
    with pytest.raises(ScanReportError, match="entire frozen inventory"):
        _build(shards=(_shard(complete=False),), scan_complete=True)


def test_report_digest_rejects_semantic_mutation() -> None:
    report = _build(shards=(_shard(),), scan_complete=True)
    mutated = copy.deepcopy(report)
    mutated["terminal_diagnostic"] = "changed"
    with pytest.raises(ScanReportError, match="digest mismatch"):
        canonical_report_bytes(mutated)
