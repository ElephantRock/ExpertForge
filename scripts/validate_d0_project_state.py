"""Validate the synchronized D0.0 project-state index."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.validate_d0_config_binding import FINGERPRINT_PATHS, ROOT
from scripts.validate_d0_final_review_report import MANIFEST_PATH as FINAL_REVIEW_MANIFEST_PATH
from scripts.validate_d0_generation_prompts import PROMPT_MANIFEST_PATH, load_prompt_manifest
from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    ContractValidationError,
    load_json_object,
)

MANIFEST_PATH = ROOT / "experiments/d0/project-state-v1.json"
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


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    _require(set(value) == expected, f"{field} keys changed")


def _fingerprint_digest(path: Path) -> str:
    record = load_json_object(path)
    digest = record.get("digest_str")
    if not isinstance(digest, str):
        raise ContractValidationError(f"{path.name}: fingerprint digest missing")
    return digest


def validate_project_state(
    manifest: Mapping[str, Any],
    project_state_bytes: bytes,
    contract: Mapping[str, Any],
    final_review_manifest: Mapping[str, Any],
) -> dict[str, object]:
    """Validate project-state identity, frozen evidence, and authorization boundary."""

    expected_keys = {
        "schema_version",
        "status",
        "issue",
        "parent_issue",
        "pull_request",
        "project_state_path",
        "project_state_sha256",
        "current_milestone",
        "d0_0_state",
        "remaining_ratification_blockers",
        "material_execution_authorized",
        "d0_1_authorized",
        "d0_2_authorized",
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
        manifest["schema_version"] == "expertforge-d0-project-state/1",
        "project-state schema changed",
    )
    _require(
        manifest["status"] == "synchronized_ratification_ready",
        "project-state status changed",
    )
    _require(manifest["issue"] == 42, "project-state issue changed")
    _require(manifest["parent_issue"] == 41, "project-state parent issue changed")
    _require(manifest["pull_request"] == 43, "project-state pull request changed")
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
    _require(
        manifest["d0_0_state"] == "ratification_ready_pending_acceptance_merge_and_issue_closure",
        "D0.0 project state changed",
    )
    _require(manifest["remaining_ratification_blockers"] == [], "project-state blocker drift")
    for field in (
        "material_execution_authorized",
        "d0_1_authorized",
        "d0_2_authorized",
        "qualification_attempt_authorized",
        "canonical_attempt_authorized",
        "actual_corpus_scan_completed",
    ):
        _require(manifest[field] is False, f"{field} must remain false")

    _require(contract["status"] == "proposed_not_ratified", "contract status changed prematurely")
    _require(contract["ratification_blockers"] == [], "contract blocker state is not terminal")
    dataset = _require_mapping(contract["dataset"], "dataset")
    tokenizer = _require_mapping(contract["tokenizer"], "tokenizer")
    prompts = _require_mapping(contract["generation_prompt_contract"], "generation_prompt_contract")
    _require(
        manifest["dataset_manifest_sha256"] == dataset["source_manifest_sha256"],
        "project-state dataset identity drift",
    )
    _require(
        manifest["tokenizer_manifest_sha256"] == tokenizer["source_manifest_sha256"],
        "project-state tokenizer identity drift",
    )
    _require(
        manifest["qualification_specification_fingerprint"]
        == _fingerprint_digest(FINGERPRINT_PATHS["qualification"]),
        "project-state qualification fingerprint drift",
    )
    _require(
        manifest["canonical_specification_fingerprint"]
        == _fingerprint_digest(FINGERPRINT_PATHS["canonical"]),
        "project-state canonical fingerprint drift",
    )
    prompt_manifest, prompt_raw = load_prompt_manifest(PROMPT_MANIFEST_PATH)
    _require(
        manifest["generation_prompt_manifest_sha256"]
        == hashlib.sha256(prompt_raw).hexdigest()
        == prompts["manifest_sha256"],
        "project-state prompt manifest identity drift",
    )
    _require(
        manifest["generation_prompt_payload_sha256"]
        == prompt_manifest["prompt_payload_sha256"]
        == prompts["prompt_payload_sha256"],
        "project-state prompt payload identity drift",
    )
    _require(
        manifest["final_review_report_sha256"] == final_review_manifest["report_sha256"],
        "project-state final review identity drift",
    )

    project_state = project_state_bytes.decode("utf-8")
    positions = [project_state.find(heading) for heading in _REQUIRED_HEADINGS]
    _require(all(position >= 0 for position in positions), "project-state required heading missing")
    _require(positions == sorted(positions), "project-state heading order changed")
    for claim in _FORBIDDEN_CLAIMS:
        _require(
            claim not in project_state.casefold(),
            f"PROJECT_STATE.md contains forbidden claim: {claim}",
        )
    required_phrases: Sequence[str] = (
        "ratification-ready; not yet ratified",
        "zero preparation blockers",
        "Issue #42 remains open",
        "D0.1 and D0.2 may proceed in parallel only after Issue #42 closes",
        "The corpus-wide contamination scan has not run",
        "No production dense model has been instantiated",
        "No material D0 execution is authorized",
        "uv run --locked python -m scripts.validate_d0_ratification",
    )
    lowered = project_state.casefold()
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
        "d0_1_authorized": False,
        "d0_2_authorized": False,
        "material_execution_authorized": False,
    }


def validate_all() -> dict[str, object]:
    return validate_project_state(
        load_json_object(MANIFEST_PATH),
        PROJECT_STATE_PATH.read_bytes(),
        load_json_object(ROOT / CONTRACT_PATH),
        load_json_object(FINAL_REVIEW_MANIFEST_PATH),
    )


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Validate the synchronized D0.0 project state")


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
        sys.stderr.write(f"D0 PROJECT STATE INVALID: {error}\n")
        return 1
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
