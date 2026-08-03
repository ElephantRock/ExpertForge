"""Validate the content-addressed D0.0 ratification record."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from typing import Any

from scripts.validate_d0_config_binding import FINGERPRINT_PATHS, ROOT
from scripts.validate_d0_final_review_report import MANIFEST_PATH as FINAL_REVIEW_MANIFEST_PATH
from scripts.validate_d0_generation_prompts import PROMPT_MANIFEST_PATH, load_prompt_manifest
from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    ContractValidationError,
    load_json_object,
)

RATIFICATION_PATH = ROOT / "experiments/d0/ratification-v1.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint_digest(profile: str) -> str:
    record = load_json_object(FINGERPRINT_PATHS[profile])
    digest = record.get("digest_str")
    if not isinstance(digest, str):
        raise ContractValidationError(f"{profile} fingerprint digest missing")
    return digest


def validate_ratification_record(
    record: Mapping[str, Any],
    contract: Mapping[str, Any],
    final_review: Mapping[str, Any],
) -> dict[str, object]:
    """Validate ratification identity, authorization, and claim boundaries."""

    expected_keys = {
        "schema_version",
        "status",
        "issue",
        "parent_issue",
        "pull_request",
        "post_ratification_issue",
        "accepted_source_head",
        "merge_commit",
        "accepted_ci",
        "accepted_at_utc",
        "blocking_review_findings",
        "all_blocking_review_threads_resolved",
        "remaining_ratification_blockers",
        "d0_1_authorized",
        "d0_2_authorized",
        "d0_3_authorized",
        "material_execution_authorized",
        "actual_corpus_scan_completed",
        "qualification_attempt_authorized",
        "canonical_attempt_authorized",
        "dataset_manifest_sha256",
        "tokenizer_manifest_sha256",
        "qualification_specification_fingerprint",
        "canonical_specification_fingerprint",
        "generation_prompt_manifest_sha256",
        "generation_prompt_payload_sha256",
        "final_review_report_sha256",
        "record_sha256",
    }
    _require(set(record) == expected_keys, "ratification record keys changed")
    _require(
        record["schema_version"] == "expertforge-d0-ratification-record/1",
        "ratification schema changed",
    )
    _require(record["status"] == "ratified", "ratification status changed")
    _require(record["issue"] == 42, "ratified issue changed")
    _require(record["parent_issue"] == 41, "parent issue changed")
    _require(record["pull_request"] == 43, "accepted pull request changed")
    _require(record["post_ratification_issue"] == 44, "state-transition issue changed")
    _require(
        record["accepted_source_head"] == "c1495d414d19f46d71241b9110bf9ce5e08efe17",
        "accepted source head changed",
    )
    _require(
        record["merge_commit"] == "9983ec62748b30e982e5879df1426fa80c900a7f",
        "merge commit changed",
    )
    _require(record["accepted_at_utc"] == "2026-08-03T14:07:58Z", "acceptance time changed")
    accepted_ci = _mapping(record["accepted_ci"], "accepted_ci")
    _require(
        accepted_ci == {"run_id": 30820314136, "run_number": 349, "conclusion": "success"},
        "accepted CI changed",
    )
    _require(record["blocking_review_findings"] == 5, "review finding count changed")
    _require(record["all_blocking_review_threads_resolved"] is True, "review remains unresolved")
    _require(record["remaining_ratification_blockers"] == [], "ratification blockers remain")
    _require(record["d0_1_authorized"] is True, "D0.1 must be authorized")
    _require(record["d0_2_authorized"] is True, "D0.2 must be authorized")
    for field in (
        "d0_3_authorized",
        "material_execution_authorized",
        "actual_corpus_scan_completed",
        "qualification_attempt_authorized",
        "canonical_attempt_authorized",
    ):
        _require(record[field] is False, f"{field} must remain false")

    payload = dict(record)
    recorded_digest = payload.pop("record_sha256")
    actual_digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    _require(
        recorded_digest == actual_digest,
        f"ratification record SHA-256 mismatch: actual {actual_digest}",
    )

    dataset = _mapping(contract["dataset"], "dataset")
    tokenizer = _mapping(contract["tokenizer"], "tokenizer")
    prompts = _mapping(contract["generation_prompt_contract"], "generation_prompt_contract")
    _require(contract["status"] == "proposed_not_ratified", "historical proposal status changed")
    _require(contract["ratification_blockers"] == [], "historical blocker state changed")
    _require(
        record["dataset_manifest_sha256"] == dataset["source_manifest_sha256"],
        "dataset identity drift",
    )
    _require(
        record["tokenizer_manifest_sha256"] == tokenizer["source_manifest_sha256"],
        "tokenizer identity drift",
    )
    _require(
        record["qualification_specification_fingerprint"]
        == _fingerprint_digest("qualification"),
        "qualification fingerprint drift",
    )
    _require(
        record["canonical_specification_fingerprint"] == _fingerprint_digest("canonical"),
        "canonical fingerprint drift",
    )
    prompt_manifest, prompt_raw = load_prompt_manifest(PROMPT_MANIFEST_PATH)
    _require(
        record["generation_prompt_manifest_sha256"]
        == hashlib.sha256(prompt_raw).hexdigest()
        == prompts["manifest_sha256"],
        "prompt manifest identity drift",
    )
    _require(
        record["generation_prompt_payload_sha256"]
        == prompt_manifest["prompt_payload_sha256"]
        == prompts["prompt_payload_sha256"],
        "prompt payload identity drift",
    )
    _require(
        record["final_review_report_sha256"] == final_review["report_sha256"],
        "final review identity drift",
    )

    return {
        "status": "valid_d0_ratification_record",
        "schema_version": record["schema_version"],
        "record_sha256": actual_digest,
        "accepted_source_head": record["accepted_source_head"],
        "merge_commit": record["merge_commit"],
        "accepted_ci_run_number": accepted_ci["run_number"],
        "remaining_ratification_blockers": [],
        "d0_1_authorized": True,
        "d0_2_authorized": True,
        "d0_3_authorized": False,
        "material_execution_authorized": False,
    }


def validate_all() -> dict[str, object]:
    return validate_ratification_record(
        load_json_object(RATIFICATION_PATH),
        load_json_object(ROOT / CONTRACT_PATH),
        load_json_object(FINAL_REVIEW_MANIFEST_PATH),
    )


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Validate the D0.0 ratification record")


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    try:
        report = validate_all()
    except (
        ContractValidationError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        sys.stderr.write(f"D0 RATIFICATION RECORD INVALID: {error}\n")
        return 1
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
