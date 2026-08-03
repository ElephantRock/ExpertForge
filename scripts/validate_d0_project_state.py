"""Validate the post-ratification D0 project-state index."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.validate_d0_config_binding import ROOT
from scripts.validate_d0_final_review_report import MANIFEST_PATH as FINAL_REVIEW_MANIFEST_PATH
from scripts.validate_d0_ratification_record import (
    RATIFICATION_PATH,
    validate_ratification_record,
)
from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    ContractValidationError,
    load_json_object,
)

MANIFEST_PATH = ROOT / "experiments/d0/project-state-v2.json"
PROJECT_STATE_PATH = ROOT / "PROJECT_STATE.md"
_REQUIRED_HEADINGS = (
    "# PROJECT_STATE.md",
    "## Current milestone",
    "## D0 work-package state",
    "## Frozen D0.0 prospective contract",
    "## Permanent conformance gate",
    "## Claim and execution boundary",
    "## Active issues and pull requests",
    "## Known blockers",
    "## Latest accepted experiment",
    "## Next recommended action",
    "## Critical path",
    "## Relationship to ExpertOS",
)
_FORBIDDEN_CLAIMS = (
    "material execution authorized: true",
    "actual corpus scan completed: true",
    "qualification attempt exists: true",
    "canonical attempt exists: true",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    _require(set(value) == expected, f"{field} keys changed")


def validate_project_state(
    manifest: Mapping[str, Any],
    project_state_bytes: bytes,
    contract: Mapping[str, Any],
    final_review_manifest: Mapping[str, Any],
    ratification_record: Mapping[str, Any],
) -> dict[str, object]:
    """Validate current project-state identity and authorization boundaries."""

    ratification_report = validate_ratification_record(
        ratification_record,
        contract,
        final_review_manifest,
    )
    expected_keys = {
        "schema_version",
        "status",
        "issue",
        "ratified_issue",
        "parent_issue",
        "pull_request",
        "merge_commit",
        "ratification_record_path",
        "ratification_record_sha256",
        "project_state_path",
        "project_state_sha256",
        "current_milestone",
        "d0_0_state",
        "remaining_ratification_blockers",
        "material_execution_authorized",
        "d0_1_authorized",
        "d0_2_authorized",
        "d0_3_authorized",
        "qualification_attempt_authorized",
        "canonical_attempt_authorized",
        "actual_corpus_scan_completed",
        "dataset_manifest_sha256",
        "tokenizer_manifest_sha256",
        "qualification_specification_fingerprint",
        "canonical_specification_fingerprint",
        "generation_prompt_manifest_sha256",
        "generation_prompt_payload_sha256",
        "final_review_report_sha256",
    }
    _require_exact_keys(manifest, expected_keys, "project-state manifest")
    _require(
        manifest["schema_version"] == "expertforge-d0-project-state/2",
        "project-state schema changed",
    )
    _require(manifest["status"] == "synchronized_ratified", "project-state status changed")
    _require(manifest["issue"] == 44, "state-transition issue changed")
    _require(manifest["ratified_issue"] == 42, "ratified issue changed")
    _require(manifest["parent_issue"] == 41, "parent issue changed")
    _require(manifest["pull_request"] == 43, "accepted pull request changed")
    _require(manifest["merge_commit"] == ratification_record["merge_commit"], "merge commit drift")
    _require(
        manifest["ratification_record_path"] == "experiments/d0/ratification-v1.json",
        "ratification record path changed",
    )
    _require(
        manifest["ratification_record_sha256"] == ratification_report["record_sha256"],
        "ratification record identity drift",
    )
    _require(manifest["project_state_path"] == "PROJECT_STATE.md", "project-state path changed")
    project_state_sha256 = hashlib.sha256(project_state_bytes).hexdigest()
    _require(
        manifest["project_state_sha256"] == project_state_sha256,
        f"PROJECT_STATE.md SHA-256 mismatch: actual {project_state_sha256}",
    )
    _require(
        manifest["current_milestone"] == "D0_controlled_dense_baseline",
        "current milestone changed",
    )
    _require(manifest["d0_0_state"] == "ratified", "D0.0 state changed")
    _require(manifest["remaining_ratification_blockers"] == [], "ratification blockers remain")
    _require(manifest["d0_1_authorized"] is True, "D0.1 must be authorized")
    _require(manifest["d0_2_authorized"] is True, "D0.2 must be authorized")
    for field in (
        "material_execution_authorized",
        "d0_3_authorized",
        "qualification_attempt_authorized",
        "canonical_attempt_authorized",
        "actual_corpus_scan_completed",
    ):
        _require(manifest[field] is False, f"{field} must remain false")

    for field in (
        "dataset_manifest_sha256",
        "tokenizer_manifest_sha256",
        "qualification_specification_fingerprint",
        "canonical_specification_fingerprint",
        "generation_prompt_manifest_sha256",
        "generation_prompt_payload_sha256",
        "final_review_report_sha256",
    ):
        _require(manifest[field] == ratification_record[field], f"project-state {field} drift")

    project_state = project_state_bytes.decode("utf-8")
    positions = [project_state.find(heading) for heading in _REQUIRED_HEADINGS]
    _require(all(position >= 0 for position in positions), "project-state required heading missing")
    _require(positions == sorted(positions), "project-state heading order changed")
    lowered = project_state.casefold()
    for claim in _FORBIDDEN_CLAIMS:
        _require(claim not in lowered, f"PROJECT_STATE.md contains forbidden claim: {claim}")
    required_phrases: Sequence[str] = (
        "D0.0 is ratified",
        "D0.1 and D0.2 implementation work is authorized",
        "The corpus-wide contamination scan has not run",
        "No production dense model has been instantiated",
        "No material D0 execution is authorized",
        "D0.3 remains blocked",
        "D0.5 remains the first material engineering-training authorization",
        "uv run --locked python -m scripts.validate_d0_ratification",
    )
    for phrase in required_phrases:
        _require(phrase.casefold() in lowered, f"project-state required claim missing: {phrase}")

    return {
        "status": "valid_d0_project_state",
        "schema_version": manifest["schema_version"],
        "project_state_path": manifest["project_state_path"],
        "project_state_sha256": project_state_sha256,
        "project_state_bytes": len(project_state_bytes),
        "required_heading_count": len(_REQUIRED_HEADINGS),
        "remaining_ratification_blockers": [],
        "remaining_ratification_blocker_count": 0,
        "d0_1_authorized": True,
        "d0_2_authorized": True,
        "d0_3_authorized": False,
        "material_execution_authorized": False,
    }


def validate_all() -> dict[str, object]:
    return validate_project_state(
        load_json_object(MANIFEST_PATH),
        PROJECT_STATE_PATH.read_bytes(),
        load_json_object(ROOT / CONTRACT_PATH),
        load_json_object(FINAL_REVIEW_MANIFEST_PATH),
        load_json_object(RATIFICATION_PATH),
    )


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Validate the ratified D0 project state")


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    try:
        report = validate_all()
    except (
        ContractValidationError,
        KeyError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        sys.stderr.write(f"D0 PROJECT STATE INVALID: {error}\n")
        return 1
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
