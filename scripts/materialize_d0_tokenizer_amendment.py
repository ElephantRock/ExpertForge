"""Materialize the D0 tokenizer-vocabulary amendment (Decision 0012, #55).

Produces the versioned amendment siblings that bind the corrected D0 tokenizer
(``tokenizers/manifests/d0-gpt-neox-corrected-v1.json``) to the active D0
contract surface, while preserving the prior v1/v2 artifacts as superseded
evidence.

The amendment is compositional: it builds on the current head of the
amendment chain (the primitive-semantics v2 configs) and swaps only the
tokenizer manifest path/digest. Everything else — model dimensions, parameter
counts, primitive semantics, dataset identity — is unchanged.

Artifacts produced (all ``-v3`` siblings; originals are NOT overwritten):

- ``configs/d0/qualification-tokenizer-v3.yaml``
- ``configs/d0/canonical-tokenizer-v3.yaml``
- ``experiments/d0/qualification-fingerprint-v3.json``
- ``experiments/d0/canonical-fingerprint-v3.json``
- ``experiments/d0/formal-experiment-definition-v3.json``
- ``experiments/d0/tokenizer-vocabulary-amendment-v1.proposed.json``

Frozen identities::

    OLD (rejected, superseded for D0 execution):
      manifest_path   = tokenizers/manifests/d0-gpt-neox-v1.json
      manifest_sha256 = eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c

    NEW (corrected, active for D0 execution):
      manifest_path   = tokenizers/manifests/d0-gpt-neox-corrected-v1.json
      manifest_sha256 = 1142baf61c330a318be6e1b153f6769892a7f5af3b7241363a1d0a1f2ca453e1

Model dimensions and parameter counts are unchanged:
  qualification parameters: 19,685,888
  canonical parameters:     76,738,176
  model vocabulary rows:    50,257
  corrected reachable IDs:  exactly 0..50256
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from expertforge.config.resolve import canonical_bytes, resolve_config
from expertforge.identity.fingerprint import ImmutableInput, specification_fingerprint

ROOT = Path(__file__).resolve().parents[1]

# Frozen amendment identities (Decision 0012 Resolution 1).
OLD_TOKENIZER_MANIFEST_PATH = "tokenizers/manifests/d0-gpt-neox-v1.json"
OLD_TOKENIZER_MANIFEST_SHA256 = "eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c"
NEW_TOKENIZER_MANIFEST_PATH = "tokenizers/manifests/d0-gpt-neox-corrected-v1.json"
NEW_TOKENIZER_MANIFEST_SHA256 = "1142baf61c330a318be6e1b153f6769892a7f5af3b7241363a1d0a1f2ca453e1"
DATASET_SHA256 = "85fab524b13c49da78264eb03124ce113395657d6403627e53008aade7eea7d7"
PRIMITIVE_SEMANTICS_SHA256 = "5bc196942697833c4dfe9f227b16e2b69c6cafd07b05396a70650ecb34cebab0"

# The amendment chain: v2 (primitive-semantics) is the current head; v3 builds
# on v2 and swaps the tokenizer identity.
_PROFILE_PATHS = {
    "qualification": (
        ROOT / "configs/d0/qualification-primitive-v2.yaml",
        ROOT / "configs/d0/qualification-tokenizer-v3.yaml",
        ROOT / "experiments/d0/qualification-fingerprint-v3.json",
    ),
    "canonical": (
        ROOT / "configs/d0/canonical-primitive-v2.yaml",
        ROOT / "configs/d0/canonical-tokenizer-v3.yaml",
        ROOT / "experiments/d0/canonical-fingerprint-v3.json",
    ),
}
_FORMAL_V2 = ROOT / "experiments/d0/formal-experiment-definition-v2.json"
_FORMAL_V3 = ROOT / "experiments/d0/formal-experiment-definition-v3.json"
_AMENDMENT_PROPOSAL = ROOT / "experiments/d0/tokenizer-vocabulary-amendment-v1.proposed.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _swap_tokenizer_identity(config_text: str) -> str:
    """Replace the old tokenizer manifest path/sha with the corrected ones."""

    old_path_line = f"tokenizer_manifest_path: {OLD_TOKENIZER_MANIFEST_PATH}"
    new_path_line = f"tokenizer_manifest_path: {NEW_TOKENIZER_MANIFEST_PATH}"
    old_sha_line = f"tokenizer_manifest_sha256: {OLD_TOKENIZER_MANIFEST_SHA256}"
    new_sha_line = f"tokenizer_manifest_sha256: {NEW_TOKENIZER_MANIFEST_SHA256}"
    if old_path_line not in config_text:
        raise ValueError("source config does not reference the old tokenizer manifest path")
    if old_sha_line not in config_text:
        raise ValueError("source config does not reference the old tokenizer manifest sha256")
    return config_text.replace(old_path_line, new_path_line).replace(old_sha_line, new_sha_line)


def _config_bytes(source: Path) -> bytes:
    original = source.read_text(encoding="utf-8")
    swapped = _swap_tokenizer_identity(original)
    if not swapped.endswith("\n"):
        swapped += "\n"
    return swapped.encode("utf-8")


def _fingerprint_bytes(config_payload: bytes) -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory(prefix="expertforge-d0-tokenizer-amendment-") as directory:
        config_path = Path(directory) / "config.yaml"
        config_path.write_bytes(config_payload)
        envelope = resolve_config(config_path)
        fingerprint = specification_fingerprint(
            canonical_bytes(envelope),
            (
                ImmutableInput(
                    name="dataset.manifest",
                    algorithm="sha256",
                    digest=DATASET_SHA256,
                ),
                ImmutableInput(
                    name="primitive.semantics.amendment",
                    algorithm="sha256",
                    digest=PRIMITIVE_SEMANTICS_SHA256,
                ),
                ImmutableInput(
                    name="tokenizer.manifest",
                    algorithm="sha256",
                    digest=NEW_TOKENIZER_MANIFEST_SHA256,
                ),
            ),
        )
    value = fingerprint.model_dump(mode="json", by_alias=True)
    return _json_bytes(value), fingerprint.digest_str


def _formal_definition_bytes(fingerprints: dict[str, str]) -> bytes:
    value = json.loads(_FORMAL_V2.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("historical formal experiment definition must be an object")
    value["schema_version"] = "expertforge-d0-formal-experiment-definition/3"
    value["issue"] = 55
    value["pull_request"] = None
    value["ratified_issue"] = 42
    value["supersedes"] = "experiments/d0/formal-experiment-definition-v2.json"
    value["tokenizer_vocabulary_amendment_path"] = str(
        _AMENDMENT_PROPOSAL.relative_to(ROOT).as_posix()
    )
    value["tokenizer_vocabulary_amendment_sha256"] = _amendment_proposal_sha256()
    value["model_dimensions_unchanged"] = True
    value["parameter_counts_unchanged"] = True
    value["corrected_tokenizer_reachable_ids"] = "0..50256"
    profiles = value.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("historical formal definition profiles must be an object")
    for profile, fingerprint in fingerprints.items():
        entry = profiles.get(profile)
        if not isinstance(entry, dict):
            raise ValueError(f"missing formal profile {profile}")
        entry["config_path"] = f"configs/d0/{profile}-tokenizer-v3.yaml"
        entry["fingerprint_path"] = f"experiments/d0/{profile}-fingerprint-v3.json"
        entry["specification_fingerprint"] = fingerprint
        constraints = entry.get("fixed_constraints")
        if not isinstance(constraints, list) or not all(
            isinstance(item, str) for item in constraints
        ):
            raise ValueError(f"{profile} fixed constraints must be strings")
        updated = [
            f"configuration.specification_fingerprint={fingerprint}"
            if item.startswith("configuration.specification_fingerprint=")
            else item
            for item in constraints
        ]
        updated.extend(
            [
                f"tokenizer.manifest.sha256={NEW_TOKENIZER_MANIFEST_SHA256}",
                "tokenizer.reachable_ids=0..50256",
                "model.dimensions.unchanged=true",
                "model.parameter_counts.unchanged=true",
            ]
        )
        entry["fixed_constraints"] = sorted(updated)
    return _json_bytes(value)


_AMENDMENT_PROPOSAL_BODY: dict[str, object] = {
    "schema_version": "expertforge-d0-tokenizer-vocabulary-amendment/1",
    "status": "proposed_not_ratified",
    "issue": 55,
    "parent_issue": 41,
    "ratified_issue": 42,
    "decision": "doctrine/decisions/0012-d0-tokenizer-vocabulary-divergence.md",
    "resolution": 1,
    "historical_contract_path": "experiments/d0/baseline-contract-v1.proposed.json",
    "scope": "tokenizer_artifact_only",
    "unchanged_contract_domains": [
        "dataset",
        "model_dimensions",
        "parameter_counts",
        "batch",
        "optimizer",
        "initialization_distributions",
        "schedules",
        "precision_environment",
        "seeds",
        "evaluation",
        "thresholds",
        "authorization_boundary",
        "primitive_semantics",
    ],
    "changed_contract_domains": ["tokenizer"],
    "model_vocabulary_rows": 50257,
    "qualification_parameters": 19685888,
    "canonical_parameters": 76738176,
    "corrected_tokenizer_reachable_ids": "0..50256",
    "old_tokenizer": {
        "manifest_path": OLD_TOKENIZER_MANIFEST_PATH,
        "manifest_sha256": OLD_TOKENIZER_MANIFEST_SHA256,
        "status": "rejected_for_d0_execution",
        "preserved_as": "superseded_source_evidence",
    },
    "new_tokenizer": {
        "manifest_path": NEW_TOKENIZER_MANIFEST_PATH,
        "manifest_sha256": NEW_TOKENIZER_MANIFEST_SHA256,
        "derivation_record": "experiments/d0/tokenizer-amendment-evidence/derivation-record.json",
        "status": "active_for_d0_execution",
    },
    "derivation_rule": (
        "remove every added_tokens entry whose id > 50256; preserve all other "
        "fields unchanged (BPE model, normalizer, ByteLevel pre-tokenizer with "
        "add_prefix_space=false, post_processor, decoder); padding remains null"
    ),
    "ratification_blockers": [
        "full_dual_tokenizer_corpus_scan_not_executed",
        "amendment_review_not_yet_accepted",
    ],
    "affected_evidence_regenerated": {
        "profile_specification_fingerprints": True,
        "formal_experiment_definition": True,
    },
    "unaffected_prior_evidence": {
        "d0_2_model_tensors_and_parameter_identity": True,
        "d0_1_contamination_scan_normalized_source_text": True,
        "primitive_semantics_amendment": True,
        "rejected_original_tokenizer_manifest": True,
    },
}


def _amendment_proposal_sha256() -> str:
    import hashlib

    canonical = json.dumps(
        _AMENDMENT_PROPOSAL_BODY,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _amendment_proposal_bytes() -> bytes:
    body = dict(_AMENDMENT_PROPOSAL_BODY)
    body["manifest_sha256"] = _amendment_proposal_sha256()
    return _json_bytes(body)


def _expected_outputs() -> dict[Path, bytes]:
    outputs: dict[Path, bytes] = {}
    fingerprints: dict[str, str] = {}
    for profile, (source, amended, fingerprint_path) in _PROFILE_PATHS.items():
        config_payload = _config_bytes(source)
        outputs[amended] = config_payload
        fingerprint_bytes, fingerprint = _fingerprint_bytes(config_payload)
        outputs[fingerprint_path] = fingerprint_bytes
        fingerprints[profile] = fingerprint
    outputs[_FORMAL_V3] = _formal_definition_bytes(fingerprints)
    outputs[_AMENDMENT_PROPOSAL] = _amendment_proposal_bytes()
    return outputs


def materialize(*, check: bool) -> None:
    outputs = _expected_outputs()
    mismatches: list[str] = []
    for path, expected in outputs.items():
        if check:
            actual = path.read_bytes() if path.exists() else None
            if actual != expected:
                mismatches.append(path.relative_to(ROOT).as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(expected)
    if mismatches:
        raise ValueError(f"tokenizer amendment materialization is stale: {mismatches}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        materialize(check=args.check)
    except (OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"D0 TOKENIZER AMENDMENT MATERIALIZATION FAILED: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
