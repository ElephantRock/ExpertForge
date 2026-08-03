"""Validate the proposed D0 baseline contract without importing model code."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    ContractValidationError,
    load_json_object,
    validate_source_manifests,
)

_FORBIDDEN_PLACEHOLDERS = ("tbd", "approximately", "approximate", "default", "todo")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _require_exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ContractValidationError(f"{field} must be an exact integer")
    if value < minimum:
        raise ContractValidationError(f"{field} must be >= {minimum}")
    return value


def _walk_strings(value: object) -> Sequence[str]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, Mapping):
        for key, member in value.items():
            strings.extend(_walk_strings(key))
            strings.extend(_walk_strings(member))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for member in value:
            strings.extend(_walk_strings(member))
    return strings


def swiglu_width(model_width: int) -> int:
    """Return ceil((8/3)*d) rounded upward to a multiple of 256."""

    numerator = 8 * model_width
    unrounded = (numerator + 2) // 3
    return ((unrounded + 255) // 256) * 256


def parameter_count(*, vocabulary_size: int, layers: int, width: int, ffn_width: int) -> int:
    """Count tied-embedding, bias-free pre-norm RoPE/SwiGLU parameters."""

    embedding = vocabulary_size * width
    one_block = 4 * width * width + 3 * width * ffn_width + 2 * width
    final_norm = width
    return embedding + layers * one_block + final_norm


def _validate_architecture(contract: Mapping[str, Any]) -> None:
    architecture = contract["architecture"]
    expected = {
        "family": "decoder_only_transformer",
        "attention": "exact_causal_multi_head_self_attention",
        "normalization": "pre_norm_rmsnorm",
        "position_encoding": "rope",
        "feed_forward": "swiglu",
        "objective": "next_token_prediction",
        "embedding_output_tied": True,
        "biases": False,
        "dropout": 0.0,
    }
    for field, value in expected.items():
        _require(architecture.get(field) == value, f"architecture.{field} changed")


def _validate_batch(contract: Mapping[str, Any]) -> tuple[int, int]:
    sequence = contract["sequence_semantics"]
    target_tokens_per_sequence = _require_exact_int(
        sequence["target_tokens_per_sequence"],
        "sequence.target_tokens_per_sequence",
        minimum=1,
    )
    _require(
        sequence["packed_source_window_tokens"] == target_tokens_per_sequence + 1,
        "packed source window must contain one lookback token",
    )
    _require(
        sequence["cursor_advance_tokens_per_sequence"] == target_tokens_per_sequence,
        "cursor advancement must equal target-token contribution",
    )

    batch = contract["batch"]
    devices = _require_exact_int(batch["devices"], "batch.devices", minimum=1)
    microbatch = _require_exact_int(
        batch["microbatch_sequences_per_device"],
        "batch.microbatch_sequences_per_device",
        minimum=1,
    )
    accumulation = _require_exact_int(
        batch["gradient_accumulation_steps"],
        "batch.gradient_accumulation_steps",
        minimum=1,
    )
    derived_sequences = devices * microbatch * accumulation
    derived_tokens = derived_sequences * target_tokens_per_sequence
    _require(
        batch["global_sequences_per_update"] == derived_sequences,
        "global sequence batch mismatch",
    )
    _require(
        batch["target_tokens_per_update"] == derived_tokens,
        "global target-token batch mismatch",
    )
    return derived_sequences, derived_tokens


def _validate_models(contract: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    tokenizer = contract["tokenizer"]
    vocabulary_size = _require_exact_int(
        tokenizer["vocabulary_size"], "tokenizer.vocabulary_size", minimum=1
    )
    _require(tokenizer["padding_token"] is None, "training tokenizer must not define padding")
    _require(tokenizer["eos_token_id"] == 0, "document terminator must remain token 0")

    reports: dict[str, dict[str, int]] = {}
    specifications = (
        ("qualification", (15_000_000, 30_000_000)),
        ("canonical", (50_000_000, 100_000_000)),
    )
    for model_name, size_bounds in specifications:
        model = contract["models"][model_name]
        layers = _require_exact_int(model["layers"], f"models.{model_name}.layers", minimum=1)
        width = _require_exact_int(
            model["model_width"], f"models.{model_name}.model_width", minimum=1
        )
        heads = _require_exact_int(
            model["attention_heads"], f"models.{model_name}.attention_heads", minimum=1
        )
        head_dimension = _require_exact_int(
            model["head_dimension"], f"models.{model_name}.head_dimension", minimum=1
        )
        ffn_width = _require_exact_int(
            model["swiglu_intermediate_width"],
            f"models.{model_name}.swiglu_intermediate_width",
            minimum=1,
        )
        _require(width == heads * head_dimension, f"{model_name} width/head product mismatch")
        _require(
            ffn_width == swiglu_width(width),
            f"{model_name} SwiGLU width violates rounding rule",
        )
        derived_parameters = parameter_count(
            vocabulary_size=vocabulary_size,
            layers=layers,
            width=width,
            ffn_width=ffn_width,
        )
        declared_parameters = _require_exact_int(
            model["trainable_parameters"],
            f"models.{model_name}.trainable_parameters",
            minimum=1,
        )
        _require(
            derived_parameters == declared_parameters,
            f"{model_name} parameter count mismatch",
        )
        _require(
            size_bounds[0] <= declared_parameters <= size_bounds[1],
            f"{model_name} outside size gate",
        )
        _require(
            model["non_trainable_parameters"] == 0,
            f"{model_name} non-trainable parameters changed",
        )
        reports[model_name] = {
            "declared_parameters": declared_parameters,
            "derived_parameters": derived_parameters,
        }
    return reports


def _validate_schedules(
    contract: Mapping[str, Any], target_tokens_per_update: int
) -> dict[str, dict[str, int]]:
    reports: dict[str, dict[str, int]] = {}
    for schedule_name in ("qualification", "canonical"):
        schedule = contract["schedules"][schedule_name]
        budget = _require_exact_int(
            schedule["training_target_tokens"],
            f"schedules.{schedule_name}.training_target_tokens",
            minimum=1,
        )
        updates = _require_exact_int(
            schedule["optimizer_updates"],
            f"schedules.{schedule_name}.optimizer_updates",
            minimum=1,
        )
        _require(
            budget % target_tokens_per_update == 0,
            f"{schedule_name} token budget is not update aligned",
        )
        _require(
            budget // target_tokens_per_update == updates,
            f"{schedule_name} optimizer-update count mismatch",
        )
        _require(
            schedule["warmup_updates"] < updates,
            f"{schedule_name} warmup must end before training",
        )
        _require(
            math.isfinite(schedule["peak_learning_rate"]),
            f"{schedule_name} peak LR must be finite",
        )
        _require(
            math.isfinite(schedule["final_learning_rate"]),
            f"{schedule_name} final LR must be finite",
        )
        _require(
            0 < schedule["final_learning_rate"] < schedule["peak_learning_rate"],
            f"{schedule_name} final LR must be positive and below peak",
        )
        reports[schedule_name] = {
            "declared_updates": updates,
            "derived_updates": budget // target_tokens_per_update,
            "training_target_tokens": budget,
        }
    return reports


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ``contract`` and return a deterministic summary report."""

    _require(contract.get("status") == "proposed_not_ratified", "unexpected proposal status")
    _require(contract.get("issue") == 42, "contract must bind Issue #42")
    _require(contract.get("parent_issue") == 41, "contract must bind parent Issue #41")
    _validate_architecture(contract)
    source_reports = validate_source_manifests(
        contract,
        load_json_object(Path(contract["tokenizer"]["source_manifest_path"])),
        load_json_object(Path(contract["dataset"]["source_manifest_path"])),
    )
    _, target_tokens_per_update = _validate_batch(contract)
    model_reports = _validate_models(contract)
    schedule_reports = _validate_schedules(contract, target_tokens_per_update)

    for text in _walk_strings(contract):
        lowered = text.casefold()
        for placeholder in _FORBIDDEN_PLACEHOLDERS:
            _require(
                placeholder not in lowered,
                f"forbidden placeholder language found: {placeholder!r}",
            )

    blockers = contract.get("ratification_blockers")
    if not isinstance(blockers, list):
        raise ContractValidationError("ratification_blockers must be a list")
    _require(len(blockers) == 0, "proposal must have zero remaining blockers")

    return {
        "status": "valid_proposal",
        "schema_version": contract["schema_version"],
        "target_tokens_per_update": target_tokens_per_update,
        "models": model_reports,
        "schedules": schedule_reports,
        "source_manifests": source_reports,
        "ratification_blocker_count": len(blockers),
    }


def load_contract(path: Path) -> Mapping[str, Any]:
    return load_json_object(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the proposed D0 baseline contract")
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_contract(load_contract(args.contract))
    except (ContractValidationError, KeyError, OSError, json.JSONDecodeError) as error:
        sys.stderr.write(f"D0 CONTRACT INVALID: {error}\n")
        return 1
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
