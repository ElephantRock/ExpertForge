"""Validate the fixed D0 generation prompts and contamination-check contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    ContractValidationError,
    load_json_object,
)

ROOT = Path(__file__).resolve().parents[1]
PROMPT_MANIFEST_PATH = ROOT / "experiments/d0/generation-prompts-v1.json"

_EXPECTED_PROMPT_IDS = tuple(f"d0-gen-{index:02d}" for index in range(1, 9))
_EXPECTED_CATEGORIES = (
    "narrative_continuation",
    "mechanistic_explanation",
    "risk_aware_planning",
    "systems_comparison",
    "causal_hypothesis",
    "diagnostic_dialogue",
    "technical_inventory",
    "constrained_prioritization",
)
_EXPECTED_CANONICALIZATION = {
    "encoding": "utf-8",
    "nul_policy": "reject",
    "newline_mapping": "CRLF_and_CR_to_LF",
    "other_codepoints_and_whitespace": "preserve_exactly",
    "terminal_newline_in_prompt_text": False,
}
_EXPECTED_CHECKS = (
    "exact_normalized_prompt_substring",
    "exact_normalized_probe_substring",
    "any_exact_contiguous_64_codepoint_prompt_window",
)
_FORBIDDEN_PLACEHOLDERS = ("tbd", "todo", "approximately", "approximate", "default")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _require_sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        raise ContractValidationError(f"{field} must be an array")
    return value


def _require_exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ContractValidationError(f"{field} must be an exact integer")
    if value < minimum:
        raise ContractValidationError(f"{field} must be >= {minimum}")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_d0_text(text: str) -> str:
    """Apply the frozen D0 newline mapping and reject NUL."""

    if "\x00" in text:
        raise ContractValidationError("text contains a forbidden NUL code point")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _find_first(haystack: str, needle: str) -> tuple[int, int] | None:
    start = haystack.find(needle)
    if start < 0:
        return None
    return start, start + len(needle)


def _partial_window_hit(
    document: str,
    prompt: str,
    window_codepoints: int,
) -> tuple[int, int] | None:
    best: tuple[int, int, int] | None = None
    for prompt_start in range(0, len(prompt) - window_codepoints + 1):
        window = prompt[prompt_start : prompt_start + window_codepoints]
        document_start = document.find(window)
        if document_start < 0:
            continue
        candidate = (document_start, prompt_start, document_start + window_codepoints)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    return best[0], best[2]


def scan_document_for_prompt_contamination(
    manifest: Mapping[str, Any],
    *,
    document_id: str,
    source_file_path: str,
    physical_row_index: int,
    text: str,
) -> list[dict[str, object]]:
    """Return at most one precedence-selected hit per prompt for one document."""

    _require(bool(document_id), "document_id must not be blank")
    _require(bool(source_file_path), "source_file_path must not be blank")
    _require_exact_int(physical_row_index, "physical_row_index")
    document = normalize_d0_text(text)

    policy = _require_mapping(manifest["contamination_policy"], "contamination_policy")
    window_codepoints = _require_exact_int(
        policy["partial_window_codepoints"],
        "contamination_policy.partial_window_codepoints",
        minimum=1,
    )
    prompts = _require_sequence(manifest["prompts"], "prompts")
    hits: list[dict[str, object]] = []
    for index, raw_prompt in enumerate(prompts):
        prompt = _require_mapping(raw_prompt, f"prompts[{index}]")
        prompt_id = str(prompt["id"])
        normalized_prompt = normalize_d0_text(str(prompt["text"]))
        normalized_probe = normalize_d0_text(str(prompt["contamination_probe_text"]))

        selected_check: str | None = None
        selected_span = _find_first(document, normalized_prompt)
        if selected_span is not None:
            selected_check = _EXPECTED_CHECKS[0]
        else:
            selected_span = _find_first(document, normalized_probe)
            if selected_span is not None:
                selected_check = _EXPECTED_CHECKS[1]
            else:
                selected_span = _partial_window_hit(
                    document,
                    normalized_prompt,
                    window_codepoints,
                )
                if selected_span is not None:
                    selected_check = _EXPECTED_CHECKS[2]

        if selected_check is not None and selected_span is not None:
            hits.append(
                {
                    "prompt_id": prompt_id,
                    "check": selected_check,
                    "document_id": document_id,
                    "source_file_path": source_file_path,
                    "physical_row_index": physical_row_index,
                    "match_start_codepoint": selected_span[0],
                    "match_end_codepoint": selected_span[1],
                }
            )
    return hits


def _validate_envelope(
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    raw_manifest_bytes: bytes,
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "issue",
        "parent_issue",
        "pull_request",
        "contract_path",
        "dataset_manifest_path",
        "dataset_manifest_sha256",
        "tokenizer_manifest_path",
        "tokenizer_manifest_sha256",
        "prompt_count",
        "prompt_payload_sha256",
        "prompt_set_id",
        "canonicalization",
        "decoding",
        "contamination_policy",
        "prompts",
    }
    _require(set(manifest) == expected_keys, "generation prompt manifest fields changed")
    _require(
        manifest["schema_version"] == "expertforge-d0-generation-prompt-set/1",
        "generation prompt schema version changed",
    )
    _require(
        manifest["status"] == "prospective_not_scanned_or_executed",
        "generation prompt manifest status changed",
    )
    _require(manifest["issue"] == contract["issue"] == 42, "generation prompts issue changed")
    _require(
        manifest["parent_issue"] == contract["parent_issue"] == 41,
        "generation prompts parent issue changed",
    )
    _require(manifest["pull_request"] == 43, "generation prompts must bind draft PR #43")
    _require(
        manifest["contract_path"] == CONTRACT_PATH.as_posix(),
        "generation prompt contract path changed",
    )

    evaluation = _require_mapping(contract["evaluation"], "contract.evaluation")
    _require(
        evaluation["generation_prompt_manifest_path"]
        == PROMPT_MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "contract generation prompt manifest path changed",
    )
    _require(
        _sha256_bytes(raw_manifest_bytes) == evaluation["generation_prompt_manifest_sha256"],
        "generation prompt manifest SHA-256 mismatch",
    )
    _require(
        manifest["prompt_payload_sha256"] == evaluation["generation_prompt_payload_sha256"],
        "generation prompt payload SHA-256 disagrees with contract",
    )
    _require(
        manifest["dataset_manifest_path"] == contract["dataset"]["source_manifest_path"],
        "generation prompt dataset manifest path changed",
    )
    _require(
        manifest["dataset_manifest_sha256"] == contract["dataset"]["source_manifest_sha256"],
        "generation prompt dataset manifest SHA-256 changed",
    )
    _require(
        manifest["tokenizer_manifest_path"] == contract["tokenizer"]["source_manifest_path"],
        "generation prompt tokenizer manifest path changed",
    )
    _require(
        manifest["tokenizer_manifest_sha256"] == contract["tokenizer"]["source_manifest_sha256"],
        "generation prompt tokenizer manifest SHA-256 changed",
    )


def _validate_protocol(
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    _require(
        dict(_require_mapping(manifest["canonicalization"], "canonicalization"))
        == _EXPECTED_CANONICALIZATION,
        "generation prompt canonicalization changed",
    )
    evaluation = _require_mapping(contract["evaluation"], "contract.evaluation")
    seeds = _require_mapping(contract["seeds"], "contract.seeds")
    expected_decoding = {
        "generation_max_new_tokens": evaluation["generation_max_new_tokens"],
        "generation_temperature": evaluation["generation_temperature"],
        "generation_top_p": evaluation["generation_top_p"],
        "generation_top_k": evaluation["generation_top_k"],
        "generation_repetition_penalty": evaluation["generation_repetition_penalty"],
        "generation_stop_token_id": evaluation["generation_stop_token_id"],
        "generation_seeds": seeds["canonical_generation_seeds"],
    }
    _require(
        dict(_require_mapping(manifest["decoding"], "decoding")) == expected_decoding,
        "generation decoding protocol changed",
    )

    policy = _require_mapping(manifest["contamination_policy"], "contamination_policy")
    expected_policy_keys = {
        "corpus_scope",
        "document_normalization",
        "checks_in_precedence_order",
        "partial_window_codepoints",
        "maximum_allowed_hits",
        "hit_record_fields",
        "report_order",
        "on_any_hit",
        "actual_corpus_scan_stage",
    }
    _require(set(policy) == expected_policy_keys, "contamination policy fields changed")
    _require(
        policy["corpus_scope"]
        == evaluation["generation_contamination_corpus_scope"]
        == "all_accepted_normalized_deduplicated_documents_before_split",
        "contamination corpus scope changed",
    )
    _require(
        policy["document_normalization"] == "d0_dataset_normalization_contract",
        "contamination normalization changed",
    )
    _require(
        tuple(_require_sequence(policy["checks_in_precedence_order"], "checks"))
        == _EXPECTED_CHECKS,
        "contamination checks changed",
    )
    _require(
        _require_exact_int(
            policy["partial_window_codepoints"],
            "contamination_policy.partial_window_codepoints",
            minimum=1,
        )
        == evaluation["generation_contamination_partial_window_codepoints"]
        == 64,
        "contamination partial-window threshold changed",
    )
    _require(
        _require_exact_int(
            policy["maximum_allowed_hits"],
            "contamination_policy.maximum_allowed_hits",
        )
        == evaluation["generation_contamination_maximum_allowed_hits"]
        == 0,
        "contamination hit threshold changed",
    )
    _require(
        tuple(_require_sequence(policy["hit_record_fields"], "hit_record_fields"))
        == (
            "prompt_id",
            "check",
            "document_id",
            "source_file_path",
            "physical_row_index",
            "match_start_codepoint",
            "match_end_codepoint",
        ),
        "contamination hit record changed",
    )
    _require(
        tuple(_require_sequence(policy["report_order"], "report_order"))
        == (
            "prompt_id_ascending",
            "check_precedence",
            "document_id_ascending",
            "match_start_codepoint_ascending",
        ),
        "contamination report order changed",
    )
    _require(
        policy["on_any_hit"] == "fail_closed_and_require_prompt_set_amendment",
        "contamination hit action changed",
    )
    _require(
        policy["actual_corpus_scan_stage"] == "D0.1_preflight_before_packing_or_training",
        "contamination scan stage changed",
    )


def _validate_prompts(
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    prompts = _require_sequence(manifest["prompts"], "prompts")
    prompt_count = _require_exact_int(manifest["prompt_count"], "prompt_count", minimum=1)
    _require(prompt_count == len(prompts) == 8, "generation prompt count changed")
    _require(
        prompt_count == contract["evaluation"]["generation_prompt_count"],
        "generation prompt count disagrees with contract",
    )

    seen_texts: set[str] = set()
    seen_probes: set[str] = set()
    payload_prompts: list[Mapping[str, Any]] = []
    for index, raw_prompt in enumerate(prompts):
        prompt = _require_mapping(raw_prompt, f"prompts[{index}]")
        expected_keys = {
            "id",
            "category",
            "text",
            "contamination_probe_text",
            "text_sha256",
            "contamination_probe_sha256",
        }
        _require(set(prompt) == expected_keys, f"prompt {index} fields changed")
        _require(prompt["id"] == _EXPECTED_PROMPT_IDS[index], f"prompt {index} id changed")
        _require(
            prompt["category"] == _EXPECTED_CATEGORIES[index],
            f"{prompt['id']} category changed",
        )

        text = prompt["text"]
        probe = prompt["contamination_probe_text"]
        _require(isinstance(text, str) and bool(text), f"{prompt['id']} text is blank")
        _require(isinstance(probe, str) and bool(probe), f"{prompt['id']} probe is blank")
        _require(text == normalize_d0_text(text), f"{prompt['id']} text is not canonical")
        _require(probe == normalize_d0_text(probe), f"{prompt['id']} probe is not canonical")
        _require(text == text.strip(), f"{prompt['id']} text has edge whitespace")
        _require(probe == probe.strip(), f"{prompt['id']} probe has edge whitespace")
        _require(not text.endswith("\n"), f"{prompt['id']} text has terminal newline")
        _require(len(probe) >= 64, f"{prompt['id']} probe is shorter than 64 code points")
        _require(probe in text, f"{prompt['id']} probe is not an exact prompt substring")
        _require(
            prompt["text_sha256"] == _sha256_bytes(text.encode("utf-8")),
            f"{prompt['id']} text SHA-256 mismatch",
        )
        _require(
            prompt["contamination_probe_sha256"] == _sha256_bytes(probe.encode("utf-8")),
            f"{prompt['id']} probe SHA-256 mismatch",
        )
        _require(text not in seen_texts, f"{prompt['id']} duplicates prompt text")
        _require(probe not in seen_probes, f"{prompt['id']} duplicates contamination probe")
        seen_texts.add(text)
        seen_probes.add(probe)
        payload_prompts.append(prompt)

        lowered = f"{text}\n{probe}".casefold()
        for placeholder in _FORBIDDEN_PLACEHOLDERS:
            _require(
                placeholder not in lowered,
                f"{prompt['id']} contains forbidden placeholder language {placeholder!r}",
            )

    payload = {
        "prompt_set_id": manifest["prompt_set_id"],
        "canonicalization": manifest["canonicalization"],
        "decoding": manifest["decoding"],
        "contamination_policy": manifest["contamination_policy"],
        "prompts": payload_prompts,
    }
    _require(
        manifest["prompt_set_id"] == "d0-generation-prompts-v1",
        "generation prompt set identity changed",
    )
    _require(
        manifest["prompt_payload_sha256"] == _sha256_bytes(_canonical_json_bytes(payload)),
        "generation prompt canonical payload SHA-256 mismatch",
    )


def validate_generation_prompts(
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    raw_manifest_bytes: bytes,
) -> dict[str, object]:
    """Validate prompt identity, decoding, and contamination mechanics."""

    _validate_envelope(contract, manifest, raw_manifest_bytes)
    _validate_protocol(contract, manifest)
    _validate_prompts(contract, manifest)
    return {
        "status": "valid_generation_prompt_contract",
        "prompt_set_id": manifest["prompt_set_id"],
        "prompt_count": manifest["prompt_count"],
        "prompt_payload_sha256": manifest["prompt_payload_sha256"],
        "manifest_sha256": _sha256_bytes(raw_manifest_bytes),
        "contamination_checks": list(_EXPECTED_CHECKS),
        "maximum_allowed_hits": 0,
        "actual_corpus_scan_completed": False,
        "actual_corpus_scan_stage": "D0.1_preflight_before_packing_or_training",
    }


def load_prompt_manifest(path: Path) -> tuple[Mapping[str, Any], bytes]:
    raw = path.read_bytes()
    parsed = json.loads(raw.decode("utf-8"))
    return _require_mapping(parsed, "generation prompt manifest"), raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the fixed D0 generation prompts and contamination contract"
    )
    parser.add_argument("--contract", type=Path, default=ROOT / CONTRACT_PATH)
    parser.add_argument("--prompts", type=Path, default=PROMPT_MANIFEST_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        contract = load_json_object(args.contract)
        manifest, raw = load_prompt_manifest(args.prompts)
        report = validate_generation_prompts(contract, manifest, raw)
    except (
        ContractValidationError,
        KeyError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        sys.stderr.write(f"D0 GENERATION PROMPTS INVALID: {error}\n")
        return 1
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
