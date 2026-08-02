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

_PROMPT_IDS = tuple(f"d0-gen-{index:02d}" for index in range(1, 9))
_CATEGORIES = (
    "narrative_continuation",
    "mechanistic_explanation",
    "risk_aware_planning",
    "systems_comparison",
    "causal_hypothesis",
    "diagnostic_dialogue",
    "technical_inventory",
    "constrained_prioritization",
)
_CHECKS = (
    "exact_normalized_prompt_substring",
    "exact_normalized_probe_substring",
    "any_exact_contiguous_64_codepoint_prompt_window",
)
_CANONICALIZATION = {
    "encoding": "utf-8",
    "nul_policy": "reject",
    "newline_mapping": "CRLF_and_CR_to_LF",
    "other_codepoints_and_whitespace": "preserve_exactly",
    "terminal_newline_in_prompt_text": False,
}
_PLACEHOLDERS = ("tbd", "todo", "approximately", "approximate", "default")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        raise ContractValidationError(f"{field} must be an array")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ContractValidationError(f"{field} must be an exact integer")
    if value < minimum:
        raise ContractValidationError(f"{field} must be >= {minimum}")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def normalize_d0_text(text: str) -> str:
    """Apply the frozen D0 newline mapping and reject NUL."""

    if "\x00" in text:
        raise ContractValidationError("text contains a forbidden NUL code point")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _first_span(document: str, needle: str) -> tuple[int, int] | None:
    start = document.find(needle)
    return None if start < 0 else (start, start + len(needle))


def _window_span(document: str, prompt: str, width: int) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for prompt_start in range(len(prompt) - width + 1):
        document_start = document.find(prompt[prompt_start : prompt_start + width])
        if document_start >= 0:
            candidates.append((document_start, prompt_start))
    if not candidates:
        return None
    document_start, _ = min(candidates)
    return document_start, document_start + width


def scan_document_for_prompt_contamination(
    manifest: Mapping[str, Any],
    *,
    document_id: str,
    source_file_path: str,
    physical_row_index: int,
    text: str,
) -> list[dict[str, object]]:
    """Return one precedence-selected hit per prompt for one normalized document."""

    _require(bool(document_id), "document_id must not be blank")
    _require(bool(source_file_path), "source_file_path must not be blank")
    _integer(physical_row_index, "physical_row_index")
    document = normalize_d0_text(text)
    policy = _mapping(manifest["contamination_policy"], "contamination_policy")
    width = _integer(
        policy["partial_window_codepoints"],
        "partial_window_codepoints",
        minimum=1,
    )

    hits: list[dict[str, object]] = []
    for index, raw_prompt in enumerate(_sequence(manifest["prompts"], "prompts")):
        prompt = _mapping(raw_prompt, f"prompts[{index}]")
        full = normalize_d0_text(str(prompt["text"]))
        probe = normalize_d0_text(str(prompt["contamination_probe_text"]))
        check = _CHECKS[0]
        span = _first_span(document, full)
        if span is None:
            check = _CHECKS[1]
            span = _first_span(document, probe)
        if span is None:
            check = _CHECKS[2]
            span = _window_span(document, full, width)
        if span is None:
            continue
        hits.append(
            {
                "prompt_id": prompt["id"],
                "check": check,
                "document_id": document_id,
                "source_file_path": source_file_path,
                "physical_row_index": physical_row_index,
                "match_start_codepoint": span[0],
                "match_end_codepoint": span[1],
            }
        )
    return hits


def _validate_envelope(
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    raw: bytes,
) -> Mapping[str, Any]:
    expected = {
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
    _require(set(manifest) == expected, "generation prompt manifest fields changed")
    _require(
        manifest["schema_version"] == "expertforge-d0-generation-prompt-set/1",
        "generation prompt schema version changed",
    )
    _require(
        manifest["status"] == "prospective_not_scanned_or_executed",
        "generation prompt manifest status changed",
    )
    _require(
        manifest["issue"] == contract["issue"] == 42,
        "generation prompts issue changed",
    )
    _require(
        manifest["parent_issue"] == contract["parent_issue"] == 41,
        "generation prompts parent issue changed",
    )
    _require(
        manifest["pull_request"] == 43,
        "generation prompts must bind draft PR #43",
    )
    _require(
        manifest["contract_path"] == CONTRACT_PATH.as_posix(),
        "contract path changed",
    )

    prompt_contract = _mapping(
        contract["generation_prompt_contract"],
        "contract.generation_prompt_contract",
    )
    _require(
        prompt_contract["manifest_path"]
        == PROMPT_MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "contract generation prompt manifest path changed",
    )
    _require(
        _sha256(raw) == prompt_contract["manifest_sha256"],
        "generation prompt manifest SHA-256 mismatch",
    )
    _require(
        manifest["prompt_payload_sha256"] == prompt_contract["prompt_payload_sha256"],
        "generation prompt payload SHA-256 disagrees with contract",
    )
    for source in ("dataset", "tokenizer"):
        _require(
            manifest[f"{source}_manifest_path"]
            == contract[source]["source_manifest_path"],
            f"generation prompt {source} manifest path changed",
        )
        _require(
            manifest[f"{source}_manifest_sha256"]
            == contract[source]["source_manifest_sha256"],
            f"generation prompt {source} manifest SHA-256 changed",
        )
    return prompt_contract


def _validate_protocol(
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
) -> None:
    _require(
        dict(_mapping(manifest["canonicalization"], "canonicalization"))
        == _CANONICALIZATION,
        "generation prompt canonicalization changed",
    )
    evaluation = _mapping(contract["evaluation"], "contract.evaluation")
    seeds = _mapping(contract["seeds"], "contract.seeds")
    expected_decoding = {
        "generation_max_new_tokens": evaluation["generation_max_new_tokens"],
        "generation_temperature": evaluation["generation_temperature"],
        "generation_top_p": evaluation["generation_top_p"],
        "generation_top_k": evaluation["generation_top_k"],
        "generation_repetition_penalty": evaluation[
            "generation_repetition_penalty"
        ],
        "generation_stop_token_id": evaluation["generation_stop_token_id"],
        "generation_seeds": seeds["canonical_generation_seeds"],
    }
    _require(
        dict(_mapping(manifest["decoding"], "decoding")) == expected_decoding,
        "generation decoding protocol changed",
    )

    policy = _mapping(manifest["contamination_policy"], "contamination_policy")
    _require(
        tuple(_sequence(policy["checks_in_precedence_order"], "checks"))
        == _CHECKS,
        "contamination checks changed",
    )
    _require(
        policy["corpus_scope"]
        == prompt_contract["contamination_corpus_scope"]
        == "all_accepted_normalized_deduplicated_documents_before_split",
        "contamination corpus scope changed",
    )
    _require(
        policy["document_normalization"] == "d0_dataset_normalization_contract",
        "contamination normalization changed",
    )
    _require(
        _integer(
            policy["partial_window_codepoints"],
            "partial_window_codepoints",
            minimum=1,
        )
        == prompt_contract["contamination_partial_window_codepoints"]
        == 64,
        "contamination partial-window threshold changed",
    )
    _require(
        _integer(policy["maximum_allowed_hits"], "maximum_allowed_hits")
        == prompt_contract["contamination_maximum_allowed_hits"]
        == 0,
        "contamination hit threshold changed",
    )
    _require(
        tuple(_sequence(policy["hit_record_fields"], "hit_record_fields"))
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
        tuple(_sequence(policy["report_order"], "report_order"))
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
        policy["actual_corpus_scan_stage"]
        == "D0.1_preflight_before_packing_or_training",
        "contamination scan stage changed",
    )


def _validate_prompts(
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    prompts = _sequence(manifest["prompts"], "prompts")
    count = _integer(manifest["prompt_count"], "prompt_count", minimum=1)
    _require(
        count
        == len(prompts)
        == contract["evaluation"]["generation_prompt_count"]
        == 8,
        "generation prompt count changed",
    )
    seen_texts: set[str] = set()
    seen_probes: set[str] = set()
    payload_prompts: list[Mapping[str, Any]] = []
    expected_fields = {
        "id",
        "category",
        "text",
        "contamination_probe_text",
        "text_sha256",
        "contamination_probe_sha256",
    }
    for index, raw_prompt in enumerate(prompts):
        prompt = _mapping(raw_prompt, f"prompts[{index}]")
        _require(set(prompt) == expected_fields, f"prompt {index} fields changed")
        _require(prompt["id"] == _PROMPT_IDS[index], f"prompt {index} id changed")
        _require(
            prompt["category"] == _CATEGORIES[index],
            f"{prompt['id']} category changed",
        )
        text = prompt["text"]
        probe = prompt["contamination_probe_text"]
        _require(isinstance(text, str) and bool(text), f"{prompt['id']} text is blank")
        _require(
            isinstance(probe, str) and bool(probe),
            f"{prompt['id']} probe is blank",
        )
        _require(
            text == normalize_d0_text(text) == text.strip(),
            f"{prompt['id']} text is not canonical",
        )
        _require(
            probe == normalize_d0_text(probe) == probe.strip(),
            f"{prompt['id']} probe is not canonical",
        )
        _require(not text.endswith("\n"), f"{prompt['id']} text has terminal newline")
        _require(
            len(probe) >= 64,
            f"{prompt['id']} probe is shorter than 64 code points",
        )
        _require(
            probe in text,
            f"{prompt['id']} probe is not an exact prompt substring",
        )
        _require(
            prompt["text_sha256"] == _sha256(text.encode("utf-8")),
            f"{prompt['id']} text SHA-256 mismatch",
        )
        _require(
            prompt["contamination_probe_sha256"]
            == _sha256(probe.encode("utf-8")),
            f"{prompt['id']} probe SHA-256 mismatch",
        )
        _require(text not in seen_texts, f"{prompt['id']} duplicates prompt text")
        _require(
            probe not in seen_probes,
            f"{prompt['id']} duplicates contamination probe",
        )
        seen_texts.add(text)
        seen_probes.add(probe)
        payload_prompts.append(prompt)
        lowered = f"{text}\n{probe}".casefold()
        for placeholder in _PLACEHOLDERS:
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
        manifest["prompt_payload_sha256"] == _sha256(_canonical_bytes(payload)),
        "generation prompt canonical payload SHA-256 mismatch",
    )


def validate_generation_prompts(
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    raw_manifest_bytes: bytes,
) -> dict[str, object]:
    """Validate prompt identity, decoding, and contamination mechanics."""

    prompt_contract = _validate_envelope(contract, manifest, raw_manifest_bytes)
    _validate_protocol(contract, manifest, prompt_contract)
    _validate_prompts(contract, manifest)
    return {
        "status": "valid_generation_prompt_contract",
        "prompt_set_id": manifest["prompt_set_id"],
        "prompt_count": manifest["prompt_count"],
        "prompt_payload_sha256": manifest["prompt_payload_sha256"],
        "manifest_sha256": _sha256(raw_manifest_bytes),
        "contamination_checks": list(_CHECKS),
        "maximum_allowed_hits": 0,
        "actual_corpus_scan_completed": False,
        "actual_corpus_scan_stage": "D0.1_preflight_before_packing_or_training",
    }


def load_prompt_manifest(path: Path) -> tuple[Mapping[str, Any], bytes]:
    raw = path.read_bytes()
    return _mapping(
        json.loads(raw.decode("utf-8")),
        "generation prompt manifest",
    ), raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate D0 generation prompts")
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
