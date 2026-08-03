"""Validate the content-addressed D0.0 final review report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.validate_d0_config_binding import FINGERPRINT_PATHS, ROOT
from scripts.validate_d0_generation_prompts import PROMPT_MANIFEST_PATH, load_prompt_manifest
from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    ContractValidationError,
    load_json_object,
)

MANIFEST_PATH = ROOT / "experiments/d0/final-review-report-v1.json"
_REQUIRED_HEADINGS = (
    "# D0.0 Dense Baseline Contract — Final Review Report",
    "## Executive determination",
    "## Review scope",
    "## Frozen scientific object",
    "## Immutable source identities",
    "## Configuration and specification identity",
    "## Formal experiment contract",
    "## Independent parameter accounting",
    "## Generation and contamination contract",
    "## Acceptance, failure, and kill boundaries",
    "## Permanent validation and CI",
    "## Evidence-to-claim matrix",
    "## Review findings",
    "## Alignment with the ExpertForge operating model",
    "## Ratification state",
    "## Final review disposition",
)
_FORBIDDEN_CLAIMS = (
    "corpus scan completed: true",
    "material execution authorized: true",
    "qualification attempt exists: true",
    "canonical attempt exists: true",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    _require(set(value) == expected, f"{field} keys changed")


def _fingerprint_digest(path: Path) -> str:
    record = load_json_object(path)
    digest = record.get("digest")
    if not isinstance(digest, str):
        raise ContractValidationError(f"{path.name}: fingerprint digest missing")
    return digest


def validate_final_review_report(
    manifest: Mapping[str, Any],
    report_bytes: bytes,
    contract: Mapping[str, Any],
) -> dict[str, object]:
    """Validate final review identity, required content, and claim boundary."""

    expected_keys = {
        "schema_version",
        "status",
        "issue",
        "parent_issue",
        "pull_request",
        "report_path",
        "report_sha256",
        "review_disposition",
        "remaining_ratification_blockers",
        "material_execution_authorized",
        "actual_corpus_scan_completed",
        "qualification_attempt_exists",
        "canonical_attempt_exists",
        "dataset_manifest_sha256",
        "tokenizer_manifest_sha256",
        "qualification_specification_fingerprint",
        "canonical_specification_fingerprint",
        "generation_prompt_manifest_sha256",
        "generation_prompt_payload_sha256",
    }
    _require_exact_keys(manifest, expected_keys, "final review manifest")
    _require(
        manifest["schema_version"] == "expertforge-d0-final-review-report/1",
        "final review schema changed",
    )
    _require(
        manifest["status"] == "prospective_contract_review_complete",
        "final review status changed",
    )
    _require(manifest["issue"] == 42, "final review issue changed")
    _require(manifest["parent_issue"] == 41, "final review parent issue changed")
    _require(manifest["pull_request"] == 43, "final review pull request changed")
    _require(
        manifest["report_path"] == "reports/d0-baseline-contract-final-review.md",
        "final review report path changed",
    )
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    _require(
        manifest["report_sha256"] == report_sha256,
        f"final review report SHA-256 mismatch: actual {report_sha256}",
    )
    _require(
        manifest["review_disposition"] == "accept_for_PROJECT_STATE_synchronization",
        "final review disposition changed",
    )
    _require(
        manifest["remaining_ratification_blockers"] == ["PROJECT_STATE_synchronization"],
        "final review remaining blocker changed",
    )
    for field in (
        "material_execution_authorized",
        "actual_corpus_scan_completed",
        "qualification_attempt_exists",
        "canonical_attempt_exists",
    ):
        _require(manifest[field] is False, f"{field} must remain false")

    _require(
        contract["ratification_blockers"] == ["PROJECT_STATE_synchronization"],
        "contract blocker state disagrees with final review",
    )
    dataset = _require_mapping(contract["dataset"], "dataset")
    tokenizer = _require_mapping(contract["tokenizer"], "tokenizer")
    prompts = _require_mapping(contract["generation_prompt_contract"], "generation_prompt_contract")
    _require(
        manifest["dataset_manifest_sha256"] == dataset["source_manifest_sha256"],
        "final review dataset identity drift",
    )
    _require(
        manifest["tokenizer_manifest_sha256"] == tokenizer["source_manifest_sha256"],
        "final review tokenizer identity drift",
    )
    _require(
        manifest["qualification_specification_fingerprint"]
        == _fingerprint_digest(FINGERPRINT_PATHS["qualification"]),
        "final review qualification fingerprint drift",
    )
    _require(
        manifest["canonical_specification_fingerprint"]
        == _fingerprint_digest(FINGERPRINT_PATHS["canonical"]),
        "final review canonical fingerprint drift",
    )
    prompt_manifest, prompt_raw = load_prompt_manifest(PROMPT_MANIFEST_PATH)
    _require(
        manifest["generation_prompt_manifest_sha256"]
        == hashlib.sha256(prompt_raw).hexdigest()
        == prompts["manifest_sha256"],
        "final review prompt manifest identity drift",
    )
    _require(
        manifest["generation_prompt_payload_sha256"]
        == prompt_manifest["prompt_payload_sha256"]
        == prompts["prompt_payload_sha256"],
        "final review prompt payload identity drift",
    )

    report = report_bytes.decode("utf-8")
    positions = [report.find(heading) for heading in _REQUIRED_HEADINGS]
    _require(all(position >= 0 for position in positions), "final review required heading missing")
    _require(positions == sorted(positions), "final review heading order changed")
    for claim in _FORBIDDEN_CLAIMS:
        _require(claim not in report.casefold(), f"final review contains forbidden claim: {claim}")
    required_phrases: Sequence[str] = (
        "PROJECT_STATE_synchronization",
        "actual corpus scan has not run",
        "does not authorize D0.1 or D0.2",
        "accept the D0.0 prospective contract for project-state synchronization",
    )
    lowered = report.casefold()
    for phrase in required_phrases:
        _require(phrase.casefold() in lowered, f"final review required claim missing: {phrase}")

    return {
        "status": "valid_d0_final_review_report",
        "schema_version": manifest["schema_version"],
        "report_path": manifest["report_path"],
        "report_sha256": report_sha256,
        "report_bytes": len(report_bytes),
        "required_heading_count": len(_REQUIRED_HEADINGS),
        "remaining_ratification_blockers": ["PROJECT_STATE_synchronization"],
        "material_execution_authorized": False,
    }


def validate_all() -> dict[str, object]:
    manifest = load_json_object(MANIFEST_PATH)
    report_path = ROOT / str(manifest["report_path"])
    return validate_final_review_report(
        manifest,
        report_path.read_bytes(),
        load_json_object(ROOT / CONTRACT_PATH),
    )


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Validate the D0.0 final review report")


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    try:
        report = validate_all()
    except (
        ContractValidationError,
        KeyError,
        OSError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        sys.stderr.write(f"D0 FINAL REVIEW REPORT INVALID: {error}\n")
        return 1
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
