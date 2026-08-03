"""Canonical, content-addressed D0 contamination scan reports."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from expertforge.d0.contamination import CheckName, HitRecord, sort_hit_records
from expertforge.d0.errors import ScanReportError
from expertforge.d0.source_manifest import canonical_json_bytes

ScanStatus = Literal["incomplete", "complete_zero_hit", "complete_hits_detected"]

_HEX = frozenset("0123456789abcdef")
_CHECKS: tuple[CheckName, ...] = (
    "exact_normalized_prompt_substring",
    "exact_normalized_probe_substring",
    "any_exact_contiguous_64_codepoint_prompt_window",
)


def _non_negative(value: int, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ScanReportError(f"{field_name} must be a non-negative exact integer")
    return value


def _digest(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise ScanReportError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _commit(value: str, field_name: str) -> str:
    if len(value) != 40 or any(character not in _HEX for character in value):
        raise ScanReportError(f"{field_name} must be a full lowercase Git commit SHA")
    return value


@dataclass(frozen=True, slots=True)
class ShardScanStats:
    """Deterministic evidence accumulated for one frozen source shard."""

    path: str
    sha256: str
    size_bytes: int
    physical_rows_visited: int
    accepted_normalized_documents: int
    rejected_documents: int
    duplicates_suppressed: int
    unique_documents_scanned: int
    normalized_utf8_bytes_scanned: int
    normalized_codepoints_scanned: int
    complete: bool

    def as_dict(self) -> dict[str, object]:
        """Return the machine-record representation after invariant checks."""

        if not self.path:
            raise ScanReportError("shard path must not be blank")
        _digest(self.sha256, f"{self.path}.sha256")
        counts = {
            "size_bytes": self.size_bytes,
            "physical_rows_visited": self.physical_rows_visited,
            "accepted_normalized_documents": self.accepted_normalized_documents,
            "rejected_documents": self.rejected_documents,
            "duplicates_suppressed": self.duplicates_suppressed,
            "unique_documents_scanned": self.unique_documents_scanned,
            "normalized_utf8_bytes_scanned": self.normalized_utf8_bytes_scanned,
            "normalized_codepoints_scanned": self.normalized_codepoints_scanned,
        }
        for field_name, value in counts.items():
            _non_negative(value, f"{self.path}.{field_name}")
        if self.accepted_normalized_documents != (
            self.duplicates_suppressed + self.unique_documents_scanned
        ):
            raise ScanReportError(
                f"{self.path}: accepted documents must equal duplicates plus unique scans"
            )
        if self.physical_rows_visited < (
            self.accepted_normalized_documents + self.rejected_documents
        ):
            raise ScanReportError(
                f"{self.path}: row count is smaller than accepted plus rejected documents"
            )
        return {
            "path": self.path,
            "sha256": self.sha256,
            **counts,
            "complete": self.complete,
        }


def report_sha256(report: Mapping[str, object]) -> str:
    """Compute the report self-digest excluding ``report_sha256``."""

    payload = dict(report)
    payload.pop("report_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def canonical_report_bytes(report: Mapping[str, object]) -> bytes:
    """Return canonical bytes after validating the embedded self-digest."""

    expected = report.get("report_sha256")
    if not isinstance(expected, str):
        raise ScanReportError("report_sha256 is missing")
    actual = report_sha256(report)
    if expected != actual:
        raise ScanReportError(
            f"scan report digest mismatch: expected {expected}, actual {actual}"
        )
    return canonical_json_bytes(dict(report))


def build_scan_report(
    *,
    dataset_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    prompt_manifest_sha256: str,
    prompt_payload_sha256: str,
    scanner_source_commit: str,
    scanner_algorithm_version: str,
    runtime_versions: Mapping[str, str],
    expected_shard_count: int,
    expected_total_size_bytes: int,
    shards: Sequence[ShardScanStats],
    hits: Sequence[HitRecord],
    scan_complete: bool,
    terminal_diagnostic: str,
) -> dict[str, object]:
    """Build a truthful complete or partial content-addressed scan report."""

    bindings = {
        "dataset_manifest_sha256": _digest(
            dataset_manifest_sha256, "dataset_manifest_sha256"
        ),
        "tokenizer_manifest_sha256": _digest(
            tokenizer_manifest_sha256, "tokenizer_manifest_sha256"
        ),
        "prompt_manifest_sha256": _digest(
            prompt_manifest_sha256, "prompt_manifest_sha256"
        ),
        "prompt_payload_sha256": _digest(
            prompt_payload_sha256, "prompt_payload_sha256"
        ),
        "scanner_source_commit": _commit(scanner_source_commit, "scanner_source_commit"),
        "scanner_algorithm_version": scanner_algorithm_version,
    }
    if not scanner_algorithm_version:
        raise ScanReportError("scanner_algorithm_version must not be blank")
    if not terminal_diagnostic:
        raise ScanReportError("terminal_diagnostic must not be blank")
    expected_shard_count = _non_negative(expected_shard_count, "expected_shard_count")
    expected_total_size_bytes = _non_negative(
        expected_total_size_bytes, "expected_total_size_bytes"
    )

    serialized_shards = [shard.as_dict() for shard in shards]
    shard_paths = [shard.path for shard in shards]
    if len(set(shard_paths)) != len(shard_paths):
        raise ScanReportError("scan report contains duplicate shard paths")
    if shard_paths != sorted(shard_paths):
        raise ScanReportError("scan report shard paths are not lexicographically sorted")

    verified_bytes = sum(shard.size_bytes for shard in shards if shard.complete)
    completed_shards = sum(1 for shard in shards if shard.complete)
    physical_rows_visited = sum(shard.physical_rows_visited for shard in shards)
    accepted_documents = sum(shard.accepted_normalized_documents for shard in shards)
    rejected_documents = sum(shard.rejected_documents for shard in shards)
    duplicates_suppressed = sum(shard.duplicates_suppressed for shard in shards)
    unique_documents_scanned = sum(shard.unique_documents_scanned for shard in shards)
    normalized_utf8_bytes_scanned = sum(
        shard.normalized_utf8_bytes_scanned for shard in shards
    )
    normalized_codepoints_scanned = sum(
        shard.normalized_codepoints_scanned for shard in shards
    )

    ordered_hits = sort_hit_records(hits)
    per_tier_match_counts: dict[str, int] = {check: 0 for check in _CHECKS}
    for hit in ordered_hits:
        per_tier_match_counts[hit.check] += 1

    inventory_complete = (
        completed_shards == expected_shard_count
        and len(shards) == expected_shard_count
        and verified_bytes == expected_total_size_bytes
        and all(shard.complete for shard in shards)
    )
    if scan_complete and not inventory_complete:
        raise ScanReportError(
            "scan_complete cannot be true until the entire frozen inventory is complete"
        )

    hit_count = len(ordered_hits)
    if not scan_complete:
        status: ScanStatus = "incomplete"
    elif hit_count == 0:
        status = "complete_zero_hit"
    else:
        status = "complete_hits_detected"
    accepted = status == "complete_zero_hit"

    versions: dict[str, str] = {}
    for name in sorted(runtime_versions):
        value = runtime_versions[name]
        if not name or not value:
            raise ScanReportError("runtime version keys and values must not be blank")
        versions[name] = value

    report: dict[str, object] = {
        "schema_version": "expertforge-d0-contamination-scan-report/1",
        "status": status,
        "issue": 46,
        "parent_issue": 41,
        "bindings": bindings,
        "runtime_versions": versions,
        "expected_inventory": {
            "shard_count": expected_shard_count,
            "total_size_bytes": expected_total_size_bytes,
        },
        "observed_inventory": {
            "reported_shards": len(shards),
            "completed_shards": completed_shards,
            "verified_bytes": verified_bytes,
        },
        "counts": {
            "physical_rows_visited": physical_rows_visited,
            "accepted_normalized_documents": accepted_documents,
            "rejected_documents": rejected_documents,
            "duplicates_suppressed": duplicates_suppressed,
            "unique_documents_scanned": unique_documents_scanned,
            "normalized_utf8_bytes_scanned": normalized_utf8_bytes_scanned,
            "normalized_codepoints_scanned": normalized_codepoints_scanned,
            "hit_count": hit_count,
            "per_tier_match_counts": per_tier_match_counts,
        },
        "shards": serialized_shards,
        "hits": [hit.as_dict() for hit in ordered_hits],
        "scan_complete": scan_complete,
        "accepted": accepted,
        "terminal_diagnostic": terminal_diagnostic,
    }
    report["report_sha256"] = report_sha256(report)
    return report
