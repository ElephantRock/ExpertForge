"""Validate the proposed D0 baseline contract without importing model code."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from expertforge.rng.derivation import SeedContext, derive_seed
from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    ContractValidationError,
    load_json_object,
    validate_source_manifests,
)

_FORBIDDEN_PLACEHOLDERS = ("tbd", "approximately", "approximate", "default", "todo")
_INDEXING = (
    "optimizer_update_index_u_is_one_based_for_applied_updates; u_in_1_through_optimizer_updates"
)
_WARMUP = "lr(u)=peak_learning_rate*u/warmup_updates for 1<=u<=warmup_updates"
_COSINE = (
    "lr(u)=final_learning_rate+0.5*(peak_learning_rate-final_learning_rate)*"
    "(1+cos(pi*(u-warmup_updates)/(optimizer_updates-warmup_updates))) "
    "for warmup_updates<u<=optimizer_updates"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


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
    numerator = 8 * model_width
    unrounded = (numerator + 2) // 3
    return ((unrounded + 255) // 256) * 256


def parameter_count(*, vocabulary_size: int, layers: int, width: int, ffn_width: int) -> int:
    return (
        vocabulary_size * width
        + layers * (4 * width * width + 3 * width * ffn_width + 2 * width)
        + width
    )


def _validate_architecture(contract: Mapping[str, Any]) -> None:
    architecture = _mapping(contract["architecture"], "architecture")
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


def _validate_batch(contract: Mapping[str, Any]) -> int:
    sequence = _mapping(contract["sequence_semantics"], "sequence_semantics")
    target = _require_exact_int(
        sequence["target_tokens_per_sequence"],
        "sequence.target_tokens_per_sequence",
        minimum=1,
    )
    _require(
        sequence["packed_source_window_tokens"] == target + 1,
        "packed source window must contain one lookback token",
    )
    _require(
        sequence["cursor_advance_tokens_per_sequence"] == target,
        "cursor advancement must equal target-token contribution",
    )
    batch = _mapping(contract["batch"], "batch")
    sequences = (
        _require_exact_int(batch["devices"], "batch.devices", minimum=1)
        * _require_exact_int(
            batch["microbatch_sequences_per_device"],
            "batch.microbatch_sequences_per_device",
            minimum=1,
        )
        * _require_exact_int(
            batch["gradient_accumulation_steps"],
            "batch.gradient_accumulation_steps",
            minimum=1,
        )
    )
    tokens = sequences * target
    _require(
        batch["global_sequences_per_update"] == sequences,
        "global sequence batch mismatch",
    )
    _require(
        batch["target_tokens_per_update"] == tokens,
        "global target-token batch mismatch",
    )
    return tokens


def _validate_models(contract: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    vocabulary = _require_exact_int(
        _mapping(contract["tokenizer"], "tokenizer")["vocabulary_size"],
        "tokenizer.vocabulary_size",
        minimum=1,
    )
    reports: dict[str, dict[str, int]] = {}
    for name, bounds in (
        ("qualification", (15_000_000, 30_000_000)),
        ("canonical", (50_000_000, 100_000_000)),
    ):
        model = _mapping(_mapping(contract["models"], "models")[name], f"models.{name}")
        layers = _require_exact_int(
            model["layers"], f"models.{name}.layers", minimum=1
        )
        width = _require_exact_int(
            model["model_width"], f"models.{name}.model_width", minimum=1
        )
        heads = _require_exact_int(
            model["attention_heads"], f"models.{name}.attention_heads", minimum=1
        )
        head_dim = _require_exact_int(
            model["head_dimension"], f"models.{name}.head_dimension", minimum=1
        )
        ffn = _require_exact_int(
            model["swiglu_intermediate_width"],
            f"models.{name}.swiglu_intermediate_width",
            minimum=1,
        )
        _require(width == heads * head_dim, f"{name} width/head product mismatch")
        _require(
            ffn == swiglu_width(width),
            f"{name} SwiGLU width violates rounding rule",
        )
        derived = parameter_count(
            vocabulary_size=vocabulary,
            layers=layers,
            width=width,
            ffn_width=ffn,
        )
        declared = _require_exact_int(
            model["trainable_parameters"],
            f"models.{name}.trainable_parameters",
            minimum=1,
        )
        _require(derived == declared, f"{name} parameter count mismatch")
        _require(bounds[0] <= declared <= bounds[1], f"{name} outside size gate")
        _require(
            model["non_trainable_parameters"] == 0,
            f"{name} non-trainable parameters changed",
        )
        reports[name] = {
            "declared_parameters": declared,
            "derived_parameters": derived,
        }
    return reports


def _validate_schedules(
    contract: Mapping[str, Any], target_tokens_per_update: int
) -> dict[str, dict[str, int | float]]:
    reports: dict[str, dict[str, int | float]] = {}
    schedules = _mapping(contract["schedules"], "schedules")
    for name in ("qualification", "canonical"):
        schedule = _mapping(schedules[name], f"schedules.{name}")
        _require(
            schedule.get("semantic_update_indexing") == _INDEXING,
            f"{name} update indexing changed",
        )
        _require(
            schedule.get("learning_rate_at_update_zero") == 0.0,
            f"{name} update-zero LR changed",
        )
        _require(
            schedule.get("warmup_formula") == _WARMUP,
            f"{name} warmup formula changed",
        )
        _require(schedule.get("decay") == "cosine", f"{name} decay changed")
        _require(
            schedule.get("cosine_formula") == _COSINE,
            f"{name} cosine formula changed",
        )
        budget = _require_exact_int(
            schedule["training_target_tokens"],
            f"schedules.{name}.training_target_tokens",
            minimum=1,
        )
        updates = _require_exact_int(
            schedule["optimizer_updates"],
            f"schedules.{name}.optimizer_updates",
            minimum=1,
        )
        warmup = _require_exact_int(
            schedule["warmup_updates"],
            f"schedules.{name}.warmup_updates",
            minimum=1,
        )
        peak = float(schedule["peak_learning_rate"])
        final = float(schedule["final_learning_rate"])
        _require(
            budget == updates * target_tokens_per_update,
            f"{name} optimizer-update count mismatch",
        )
        _require(warmup < updates, f"{name} warmup must end before training")
        _require(
            math.isfinite(peak) and math.isfinite(final),
            f"{name} LR must be finite",
        )
        _require(
            0 < final < peak,
            f"{name} final LR must be positive and below peak",
        )
        first_lr = peak / warmup
        warmup_lr = peak * warmup / warmup
        final_lr = final + 0.5 * (peak - final) * (
            1 + math.cos(math.pi * (updates - warmup) / (updates - warmup))
        )
        _require(
            first_lr > 0 and warmup_lr == peak and final_lr == final,
            f"{name} LR boundary mismatch",
        )
        reports[name] = {
            "declared_updates": updates,
            "derived_updates": budget // target_tokens_per_update,
            "training_target_tokens": budget,
            "first_applied_learning_rate": first_lr,
        }
    return reports


def _validate_data_order_seed(contract: Mapping[str, Any]) -> int:
    seeds = _mapping(contract["seeds"], "seeds")
    record = _mapping(seeds.get("data_order_seed"), "seeds.data_order_seed")
    _require(
        record.get("derivation_schema") == "expertforge.seed-derivation",
        "data seed schema changed",
    )
    _require(record.get("derivation_version") == 1, "data seed version changed")
    _require(
        record.get("root_seed_field") == "seeds.master_seed",
        "data seed root binding changed",
    )
    _require(
        record.get("projection")
        == "seed_u64_first_8_sha256_bytes_big_endian_unsigned",
        "data seed projection changed",
    )
    context = _mapping(record.get("context"), "seeds.data_order_seed.context")
    _require(
        context
        == {
            "component": "data.order",
            "worker": 0,
            "rank": 0,
            "device": 0,
            "stream": 0,
        },
        "data seed context changed",
    )
    derived = derive_seed(
        _require_exact_int(seeds["master_seed"], "seeds.master_seed"),
        SeedContext(component="data.order", worker=0, rank=0, device=0, stream=0),
    ).seed_u64
    _require(
        record.get("data_seed_u64") == derived,
        "data seed derivation mismatch",
    )
    dataset = _mapping(contract["dataset"], "dataset")
    _require(
        "seeds.data_order_seed.data_seed_u64" in str(dataset.get("training_order")),
        "training order is not bound to the derived data seed",
    )
    return derived


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("status") == "proposed_not_ratified",
        "unexpected proposal status",
    )
    _require(contract.get("issue") == 42, "contract must bind Issue #42")
    _require(
        contract.get("parent_issue") == 41,
        "contract must bind parent Issue #41",
    )
    _validate_architecture(contract)
    source_reports = validate_source_manifests(
        contract,
        load_json_object(
            Path(
                _mapping(contract["tokenizer"], "tokenizer")[
                    "source_manifest_path"
                ]
            )
        ),
        load_json_object(
            Path(_mapping(contract["dataset"], "dataset")["source_manifest_path"])
        ),
    )
    target_tokens_per_update = _validate_batch(contract)
    models = _validate_models(contract)
    schedules = _validate_schedules(contract, target_tokens_per_update)
    data_seed_u64 = _validate_data_order_seed(contract)
    for text in _walk_strings(contract):
        lowered = text.casefold()
        for placeholder in _FORBIDDEN_PLACEHOLDERS:
            _require(
                placeholder not in lowered,
                f"forbidden placeholder language found: {placeholder!r}",
            )
    blockers = contract.get("ratification_blockers")
    _require(
        isinstance(blockers, list),
        "ratification_blockers must be a list",
    )
    _require(blockers == [], "proposal must have zero preparation blockers")
    return {
        "status": "valid_proposal",
        "schema_version": contract["schema_version"],
        "target_tokens_per_update": target_tokens_per_update,
        "models": models,
        "schedules": schedules,
        "data_seed_u64": data_seed_u64,
        "source_manifests": source_reports,
        "ratification_blocker_count": 0,
    }


def load_contract(path: Path) -> Mapping[str, Any]:
    return load_json_object(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the proposed D0 baseline contract"
    )
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
