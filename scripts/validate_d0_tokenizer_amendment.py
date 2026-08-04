"""Validate the D0 tokenizer-vocabulary amendment (Decision 0012, #55).

Checks that:
- the amendment proposal is internally consistent and its self-digest verifies;
- the materialized v3 configs/fingerprints/formal-definition are present and
  match the canonical resolver output exactly (no hand-edits, no drift);
- the v3 fingerprints bind the corrected tokenizer manifest digest;
- model dimensions and parameter counts are recorded as unchanged;
- the amendment's ratification blockers are still present (the corpus scan has
  not yet been executed, so the amendment is NOT ratified).

Mirrors the structure of ``validate_d0_primitive_semantics_amendment`` /
``validate_d0_primitive_semantics_evidence``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from expertforge.config.resolve import canonical_bytes, resolve_config
from expertforge.identity.fingerprint import (
    ImmutableInput,
    SpecificationFingerprintRecord,
    verify_fingerprint,
)
from scripts.materialize_d0_tokenizer_amendment import (
    NEW_TOKENIZER_MANIFEST_PATH,
    NEW_TOKENIZER_MANIFEST_SHA256,
    materialize,
)
from scripts.validate_d0_source_manifests import ContractValidationError, load_json_object

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_PATH = ROOT / "experiments/d0/tokenizer-vocabulary-amendment-v1.proposed.json"
FORMAL_V3 = ROOT / "experiments/d0/formal-experiment-definition-v3.json"

# The amendment is NOT ratified until the dual-tokenizer corpus scan is executed.
REQUIRED_RATIFICATION_BLOCKERS = frozenset(
    {
        "full_dual_tokenizer_corpus_scan_not_executed",
        "amendment_review_not_yet_accepted",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_amendment_digest(amendment: Mapping[str, Any]) -> str:
    declared = amendment.get("manifest_sha256")
    _require(
        isinstance(declared, str) and len(declared) == 64,
        "tokenizer amendment manifest_sha256 is invalid",
    )
    payload = dict(amendment)
    payload.pop("manifest_sha256", None)
    actual = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    _require(declared == actual, "tokenizer amendment manifest_sha256 mismatch")
    return str(declared)


def _validate_fingerprint_binds_corrected_manifest(profile: str) -> None:
    """Each v3 fingerprint's tokenizer.manifest immutable input must be the
    corrected digest, and the fingerprint record must verify against the
    resolved v3 config envelope."""

    config_path = ROOT / f"configs/d0/{profile}-tokenizer-v3.yaml"
    fingerprint_path = ROOT / f"experiments/d0/{profile}-fingerprint-v3.json"
    _require(config_path.is_file(), f"{profile}-tokenizer-v3.yaml is missing")
    _require(fingerprint_path.is_file(), f"{profile}-fingerprint-v3.json is missing")

    envelope = resolve_config(config_path)
    record = SpecificationFingerprintRecord.model_validate_json(
        fingerprint_path.read_text(encoding="utf-8")
    )
    inputs = (
        ImmutableInput(
            name="dataset.manifest",
            algorithm="sha256",
            digest="85fab524b13c49da78264eb03124ce113395657d6403627e53008aade7eea7d7",
        ),
        ImmutableInput(
            name="primitive.semantics.amendment",
            algorithm="sha256",
            digest="5bc196942697833c4dfe9f227b16e2b69c6cafd07b05396a70650ecb34cebab0",
        ),
        ImmutableInput(
            name="tokenizer.manifest",
            algorithm="sha256",
            digest=NEW_TOKENIZER_MANIFEST_SHA256,
        ),
    )
    # verify_fingerprint raises on mismatch; if it passes, the record is
    # well-formed and binds the corrected manifest.
    verify_fingerprint(record, canonical_bytes(envelope), inputs)

    tokenizer_input = next(
        (i for i in record.immutable_inputs if i.name == "tokenizer.manifest"), None
    )
    if tokenizer_input is None:
        raise ContractValidationError(f"{profile} fingerprint missing tokenizer.manifest input")
    _require(
        tokenizer_input.digest == NEW_TOKENIZER_MANIFEST_SHA256,
        f"{profile} fingerprint binds the wrong tokenizer manifest digest",
    )


def validate_amendment(amendment: Mapping[str, Any]) -> dict[str, object]:
    """Validate the immutable proposal and its current authorization boundary."""

    _require(
        amendment.get("schema_version") == "expertforge-d0-tokenizer-vocabulary-amendment/1",
        "tokenizer amendment schema changed",
    )
    _require(amendment.get("status") == "proposed_not_ratified", "amendment status changed")
    _require(amendment.get("issue") == 55, "amendment issue changed")
    _require(amendment.get("parent_issue") == 41, "amendment parent issue changed")
    _require(amendment.get("ratified_issue") == 42, "amendment ratified issue changed")
    _require(
        amendment.get("resolution") == 1,
        "amendment must record Resolution 1 (tokenizer-side correction)",
    )
    _require(
        amendment.get("decision")
        == "doctrine/decisions/0012-d0-tokenizer-vocabulary-divergence.md",
        "amendment decision record path changed",
    )
    _require(
        amendment.get("scope") == "tokenizer_artifact_only",
        "amendment scope must be tokenizer_artifact_only",
    )

    # Model dimensions and parameter counts are unchanged.
    _require(amendment.get("model_vocabulary_rows") == 50257, "model vocabulary rows changed")
    _require(amendment.get("qualification_parameters") == 19685888, "qualification params changed")
    _require(amendment.get("canonical_parameters") == 76738176, "canonical params changed")
    _require(
        amendment.get("corrected_tokenizer_reachable_ids") == "0..50256",
        "corrected reachable id range changed",
    )

    old_tok = amendment.get("old_tokenizer", {})
    new_tok = amendment.get("new_tokenizer", {})
    _require(
        old_tok.get("manifest_path") == "tokenizers/manifests/d0-gpt-neox-v1.json",
        "old tokenizer manifest path changed",
    )
    _require(
        old_tok.get("status") == "rejected_for_d0_execution",
        "old tokenizer must be marked rejected_for_d0_execution",
    )
    _require(
        new_tok.get("manifest_path") == NEW_TOKENIZER_MANIFEST_PATH,
        "new tokenizer manifest path changed",
    )
    _require(
        new_tok.get("manifest_sha256") == NEW_TOKENIZER_MANIFEST_SHA256,
        "new tokenizer manifest sha256 changed",
    )
    _require(
        new_tok.get("status") == "active_for_d0_execution",
        "new tokenizer must be marked active_for_d0_execution",
    )

    # Ratification blockers: the amendment is NOT ratified until the corpus scan
    # executes. Both blockers must still be present.
    blockers = set(amendment.get("ratification_blockers", []))
    missing = REQUIRED_RATIFICATION_BLOCKERS - blockers
    _require(
        not missing,
        f"tokenizer amendment is missing required ratification blockers: {sorted(missing)}",
    )

    digest = _validate_amendment_digest(amendment)
    return {
        "schema": "expertforge.d0.tokenizer_amendment_validation/1",
        "amendment_sha256": digest,
        "ratification_blockers_present": sorted(blockers),
        "ratified": False,
    }


def validate_all() -> dict[str, object]:
    """Validate the full amendment bundle: proposal + materialized artifacts."""

    amendment = load_json_object(AMENDMENT_PATH)
    report = validate_amendment(amendment)

    # The materialized v3 artifacts must match the canonical resolver output.
    materialize(check=True)

    # Each v3 fingerprint must bind the corrected tokenizer manifest digest.
    for profile in ("qualification", "canonical"):
        _validate_fingerprint_binds_corrected_manifest(profile)

    # The formal experiment definition v3 must reference the v3 artifacts.
    formal = load_json_object(FORMAL_V3)
    _require(
        formal.get("schema_version") == "expertforge-d0-formal-experiment-definition/3",
        "formal definition v3 schema changed",
    )
    _require(formal.get("issue") == 55, "formal definition v3 issue changed")
    _require(
        formal.get("supersedes") == "experiments/d0/formal-experiment-definition-v2.json",
        "formal definition v3 must supersede v2",
    )
    _require(formal.get("model_dimensions_unchanged") is True, "model dims must be unchanged")
    _require(formal.get("parameter_counts_unchanged") is True, "parameter counts must be unchanged")
    _require(
        formal.get("corrected_tokenizer_reachable_ids") == "0..50256",
        "formal definition reachable id range changed",
    )

    report["v3_fingerprints_bind_corrected_manifest"] = True
    report["materialized_artifacts_match_resolver"] = True
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    try:
        report = validate_all()
    except ContractValidationError as exc:
        sys.stderr.write(f"D0 TOKENIZER AMENDMENT VALIDATION FAILED: {exc}\n")
        return 1
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
