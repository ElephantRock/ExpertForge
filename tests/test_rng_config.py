"""Configuration-boundary tests for Issue #8 reproducibility policy."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from expertforge.config.models import ConfigRoot
from expertforge.config.resolve import canonical_bytes, resolve_config


def _base_mapping() -> dict[str, object]:
    return {
        "format_version": 1,
        "run": {"name": "rng-test"},
        "model": {"dim": 16, "n_layers": 1, "n_heads": 2, "ffn_dim": 32},
        "training": {
            "seed": 7,
            "tokens": 64,
            "batch_size": 1,
            "lr": 0.001,
        },
        "checkpointing": {"interval_tokens": 64},
    }


def test_reproducibility_policy_defaults_are_backward_compatible() -> None:
    config = ConfigRoot.model_validate(_base_mapping())

    assert config.training.seed == 7
    assert config.training.determinism_mode == "reproducible"
    assert config.training.unsupported_determinism == "error"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("determinism_mode", "sometimes"),
        ("unsupported_determinism", "ignore"),
    ],
)
def test_reproducibility_policy_rejects_unknown_values(field: str, value: str) -> None:
    mapping = _base_mapping()
    training = dict(mapping["training"])  # type: ignore[arg-type]
    training[field] = value
    mapping["training"] = training

    with pytest.raises(ValidationError):
        ConfigRoot.model_validate(mapping)


def test_policy_fields_participate_in_canonical_configuration() -> None:
    base = resolve_config("configs/smoke.yaml")
    performance = resolve_config(
        "configs/smoke.yaml",
        ["training.determinism_mode=performance"],
    )

    assert canonical_bytes(base) != canonical_bytes(performance)
    payload = json.loads(canonical_bytes(base))
    assert payload["training"]["determinism_mode"] == "reproducible"
    assert payload["training"]["unsupported_determinism"] == "error"


def test_root_seed_remains_the_single_declared_seed_field() -> None:
    config = resolve_config("configs/smoke.yaml").config.model_dump(mode="json")

    seed_paths: list[str] = []

    def walk(value: object, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else key
                if key == "seed":
                    seed_paths.append(path)
                walk(item, path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{prefix}[{index}]")

    walk(config)

    assert seed_paths == ["training.seed"]
