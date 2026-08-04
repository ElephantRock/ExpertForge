"""Validate D0.1 source/contamination implementation or actual scan evidence."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from expertforge.d0.contamination import ContaminationMatcher
from expertforge.d0.errors import D0PreflightError, ScanReportError
from expertforge.d0.scan_report import canonical_report_bytes
from expertforge.d0.scan_runner import SCANNER_ALGORITHM_VERSION, load_prompt_manifest
from expertforge.d0.source_manifest import SourceManifest, load_source_manifest
from scripts.run_d0_contamination_scan import (
    DATASET_MANIFEST_PATH,
    DATASET_MANIFEST_SHA256,
    PROMPT_MANIFEST_PATH,
    PROMPT_MANIFEST_SHA256,
    PROMPT_PAYLOAD_SHA256,
    TOKENIZER_MANIFEST_PATH,
    TOKENIZER_MANIFEST_SHA256,
)

_CHECKS = (
    "exact_normalized_prompt_substring",
    "exact_normalized_probe_substring",
    "any_exact_contiguous_64_codepoint_prompt_window",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D0PreflightError(message)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise D0PreflightError(f"{field} must be an object")
    return value


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise D0PreflightError(f"{field} must be an array")
    return value


def _exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise D0PreflightError(f"{field} must be an integer >= {minimum}")
    return value


def _load_frozen_inputs() -> tuple[SourceManifest, SourceManifest, Mapping[str, Any]]:
    dataset_manifest = load_source_manifest(
        DATASET_MANIFEST_PATH,
        expected_manifest_sha256=DATASET_MANIFEST_SHA256,
    )
    tokenizer_manifest = load_source_manifest(
        TOKENIZER_MANIFEST_PATH,
        expected_manifest_sha256=TOKENIZER_MANIFEST_SHA256,
    )
    prompt_manifest, prompt_payload_sha256 = load_prompt_manifest(
        PROMPT_MANIFEST_PATH,
        expected_sha256=PROMPT_MANIFEST_SHA256,
    )
    _require(prompt_payload_sha256 == PROMPT_PAYLOAD_SHA256, "prompt payload identity changed")
    ContaminationMatcher.from_manifest(prompt_manifest)
    return dataset_manifest, tokenizer_manifest, prompt_manifest


def validate_portable() -> dict[str, object]:
    """Validate implementation bindings without claiming an actual corpus scan."""

    dataset_manifest, tokenizer_manifest, prompt_manifest = _load_frozen_inputs()
    try:
        pyarrow_version = importlib.metadata.version("pyarrow")
    except importlib.metadata.PackageNotFoundError as exc:
        raise D0PreflightError("portable D0.1 validation requires the 'd0-data' extra") from exc
    _require(pyarrow_version.startswith("25."), "locked PyArrow major version changed")
    _require(len(dataset_manifest.files) == 14, "frozen dataset shard count changed")
    _require(
        dataset_manifest.total_size_bytes == 28_518_193_415,
        "frozen dataset byte total changed",
    )
    _require(len(tokenizer_manifest.files) == 5, "frozen tokenizer file count changed")
    _require(prompt_manifest.get("prompt_count") == 8, "frozen prompt count changed")
    return {
        "status": "valid_d0_source_contamination_implementation",
        "scanner_algorithm_version": SCANNER_ALGORITHM_VERSION,
        "dataset_manifest_sha256": dataset_manifest.manifest_sha256,
        "dataset_shard_count": len(dataset_manifest.files),
        "dataset_total_size_bytes": dataset_manifest.total_size_bytes,
        "tokenizer_manifest_sha256": tokenizer_manifest.manifest_sha256,
        "tokenizer_file_count": len(tokenizer_manifest.files),
        "prompt_manifest_sha256": PROMPT_MANIFEST_SHA256,
        "prompt_payload_sha256": PROMPT_PAYLOAD_SHA256,
        "pyarrow_version": pyarrow_version,
        "actual_corpus_scan_completed": False,
        "actual_corpus_scan_accepted": False,
        "d0_1_accepted": False,
        "tokenization_performed": False,
        "splitting_performed": False,
        "packing_performed": False,
        "training_performed": False,
    }


def _validate_shards(report: Mapping[str, Any], dataset_manifest: SourceManifest) -> None:
    shards = _sequence(report.get("shards"), "shards")
    _require(len(shards) == len(dataset_manifest.files), "actual report shard count changed")
    for index, identity in enumerate(dataset_manifest.files):
        shard = _mapping(shards[index], f"shards[{index}]")
        _require(shard.get("path") == identity.path, f"shards[{index}] path changed")
        _require(shard.get("sha256") == identity.sha256, f"shards[{index}] digest changed")
        _require(shard.get("size_bytes") == identity.size_bytes, f"shards[{index}] size changed")
        _require(shard.get("complete") is True, f"shards[{index}] is incomplete")
        rows = _exact_int(
            shard.get("physical_rows_visited"),
            f"shards[{index}].physical_rows_visited",
            minimum=1,
        )
        accepted = _exact_int(
            shard.get("accepted_normalized_documents"),
            f"shards[{index}].accepted_normalized_documents",
        )
        rejected = _exact_int(
            shard.get("rejected_documents"),
            f"shards[{index}].rejected_documents",
        )
        duplicates = _exact_int(
            shard.get("duplicates_suppressed"),
            f"shards[{index}].duplicates_suppressed",
        )
        unique = _exact_int(
            shard.get("unique_documents_scanned"),
            f"shards[{index}].unique_documents_scanned",
        )
        _require(accepted + rejected <= rows, f"shards[{index}] document counts exceed rows")
        _require(accepted == duplicates + unique, f"shards[{index}] dedup counts changed")
        reasons = _mapping(
            shard.get("rejected_documents_by_reason"),
            f"shards[{index}].rejected_documents_by_reason",
        )
        reason_total = 0
        for reason, count in reasons.items():
            _require(isinstance(reason, str) and bool(reason), "rejection reason is invalid")
            reason_total += _exact_int(count, f"shards[{index}].reason.{reason}", minimum=1)
        _require(reason_total == rejected, f"shards[{index}] rejection reason total changed")


def validate_actual_report(path: Path) -> dict[str, object]:
    """Validate complete zero-hit evidence over the exact frozen source inventory."""

    dataset_manifest, tokenizer_manifest, _ = _load_frozen_inputs()
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D0PreflightError(f"cannot read actual scan report {path}: {exc}") from exc
    report = _mapping(value, "report")
    try:
        canonical = canonical_report_bytes(report)
    except ScanReportError as exc:
        raise D0PreflightError(f"actual scan report is not canonical: {exc}") from exc
    _require(raw == canonical, "actual scan report bytes are not canonical")

    _require(
        report.get("schema_version") == "expertforge-d0-contamination-scan-report/1",
        "actual report schema changed",
    )
    # Accept frozen-inventory evidence either from an explicit execution_scope field
    # or by inferring it from verified bytes matching the frozen total. The scanner
    # at commit 75f45d7 does not write execution_scope; the objective evidence
    # (28,518,193,415 verified bytes across 14 shards) is authoritative.
    expected_inv = _mapping(report.get("expected_inventory"), "expected_inventory")
    observed_inv = _mapping(report.get("observed_inventory"), "observed_inventory")
    is_frozen = report.get("execution_scope") == "frozen_d0_source_inventory" or (
        _exact_int(expected_inv.get("total_size_bytes"), "expected_inventory.total_size_bytes")
        == 28_518_193_415
        and _exact_int(observed_inv.get("verified_bytes"), "observed_inventory.verified_bytes")
        == 28_518_193_415
    )
    _require(
        is_frozen,
        "fixture reports cannot satisfy the actual-corpus validator: "
        "verified bytes do not match the frozen 28,518,193,415-byte inventory",
    )
    _require(report.get("status") == "complete_zero_hit", "actual report is not zero-hit")
    _require(report.get("scan_complete") is True, "actual report is incomplete")
    _require(report.get("accepted") is True, "actual report is not accepted")
    _require(
        report.get("issue") == 46 and report.get("parent_issue") == 41, "issue binding changed"
    )

    bindings = _mapping(report.get("bindings"), "bindings")
    _require(
        bindings.get("dataset_manifest_sha256") == DATASET_MANIFEST_SHA256,
        "dataset binding changed",
    )
    _require(
        bindings.get("tokenizer_manifest_sha256") == TOKENIZER_MANIFEST_SHA256,
        "tokenizer binding changed",
    )
    _require(
        bindings.get("prompt_manifest_sha256") == PROMPT_MANIFEST_SHA256,
        "prompt-manifest binding changed",
    )
    _require(
        bindings.get("prompt_payload_sha256") == PROMPT_PAYLOAD_SHA256,
        "prompt-payload binding changed",
    )
    _require(
        bindings.get("scanner_algorithm_version") == SCANNER_ALGORITHM_VERSION,
        "scanner algorithm changed",
    )
    source_commit = bindings.get("scanner_source_commit")
    _require(
        isinstance(source_commit, str) and len(source_commit) == 40,
        "scanner source commit is invalid",
    )

    expected = _mapping(report.get("expected_inventory"), "expected_inventory")
    observed = _mapping(report.get("observed_inventory"), "observed_inventory")
    _require(expected.get("shard_count") == 14, "expected shard count changed")
    _require(expected.get("total_size_bytes") == 28_518_193_415, "expected bytes changed")
    _require(observed.get("reported_shards") == 14, "reported shard count changed")
    _require(observed.get("completed_shards") == 14, "completed shard count changed")
    _require(observed.get("verified_bytes") == 28_518_193_415, "verified bytes changed")

    counts = _mapping(report.get("counts"), "counts")
    _require(
        _exact_int(counts.get("physical_rows_visited"), "physical_rows_visited", minimum=1) > 0,
        "no rows visited",
    )
    _require(
        _exact_int(counts.get("unique_documents_scanned"), "unique_documents_scanned", minimum=1)
        > 0,
        "no documents scanned",
    )
    _require(counts.get("hit_count") == 0, "actual report contains hits")
    tier_counts = _mapping(counts.get("per_tier_match_counts"), "per_tier_match_counts")
    _require(
        dict(tier_counts) == {check: 0 for check in _CHECKS},
        "actual report per-tier match counts changed",
    )
    _require(_sequence(report.get("hits"), "hits") == [], "actual report hit records are non-empty")
    _validate_shards(report, dataset_manifest)

    runtime = _mapping(report.get("runtime_versions"), "runtime_versions")
    for name in ("expertforge", "pyarrow", "python"):
        _require(
            isinstance(runtime.get(name), str) and bool(runtime.get(name)),
            f"runtime {name} is missing",
        )

    return {
        "status": "accepted_d0_source_contamination_evidence",
        "report_path": path.as_posix(),
        "report_sha256": report.get("report_sha256"),
        "scanner_source_commit": source_commit,
        "dataset_manifest_sha256": dataset_manifest.manifest_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest.manifest_sha256,
        "actual_corpus_scan_completed": True,
        "actual_corpus_scan_accepted": True,
        "zero_hit": True,
        "d0_1_accepted": True,
        "tokenization_performed": False,
        "splitting_performed": False,
        "packing_performed": False,
        "training_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Validate actual frozen-corpus evidence instead of portable implementation readiness.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_actual_report(args.report) if args.report else validate_portable()
    except (
        D0PreflightError,
        KeyError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        sys.stderr.write(f"D0 SOURCE/CONTAMINATION INVALID: {exc}\n")
        return 1
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
