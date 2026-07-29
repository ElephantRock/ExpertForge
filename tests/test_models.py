"""Tests for the frozen Pydantic configuration models (Issue #5 decision §2).

Covers: sections exist, frozen/immutability, extra=forbid (unknown fields
rejected), strict scalar typing, impossible values, cross-field validation,
defaults are explicit and inspectable.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from expertforge.config.models import (
    CONFIG_FORMAT_VERSION,
    ArtifactConfig,
    CheckpointConfig,
    ConfigRoot,
    DataConfig,
    EvaluationConfig,
    HardwareConfig,
    LoggingConfig,
    RunConfig,
    TokenizerConfig,
)

# --- helpers ---------------------------------------------------------------


def _minimal_kwargs() -> dict[str, object]:
    """Return only the fields with no default — everything else defaults."""
    return {
        "run": {"name": "smoke-a"},
        "model": {"dim": 64, "n_layers": 2, "n_heads": 2, "ffn_dim": 128},
        "training": {"seed": 7, "tokens": 1024, "batch_size": 4, "lr": 1e-4},
    }


# --- structure: required sections present ----------------------------------


class TestConfigRootStructure:
    def test_minimal_valid_config_loads(self) -> None:
        cfg = ConfigRoot.model_validate(_minimal_kwargs())
        assert cfg.run.name == "smoke-a"
        # format_version has an explicit default.
        assert cfg.format_version == CONFIG_FORMAT_VERSION

    def test_all_sections_have_explicit_defaults(self) -> None:
        # Only run/model/training have hard-required fields; the remaining
        # sections (data, tokenizer, evaluation, checkpointing, logging,
        # artifacts, hardware) must construct from explicit defaults.
        cfg = ConfigRoot.model_validate(_minimal_kwargs())
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.tokenizer, TokenizerConfig)
        assert isinstance(cfg.evaluation, EvaluationConfig)
        assert isinstance(cfg.checkpointing, CheckpointConfig)
        assert isinstance(cfg.logging, LoggingConfig)
        assert isinstance(cfg.artifacts, ArtifactConfig)
        assert isinstance(cfg.hardware, HardwareConfig)


# --- extra="forbid": unknown fields rejected everywhere --------------------


class TestExtraFieldsForbidden:
    def test_unknown_root_field_rejected(self) -> None:
        kw = _minimal_kwargs() | {"mystery_section": {}}
        with pytest.raises(ValidationError) as exc:
            ConfigRoot.model_validate(kw)
        assert "extra" in str(exc.value).lower()

    def test_unknown_section_field_rejected(self) -> None:
        kw = _minimal_kwargs()
        kw["run"] = {"name": "smoke-a", "rogue_key": 1}
        with pytest.raises(ValidationError) as exc:
            ConfigRoot.model_validate(kw)
        assert "extra" in str(exc.value).lower()


# --- strict typing: wrong types rejected -----------------------------------


class TestStrictTyping:
    def test_string_where_int_expected_rejected(self) -> None:
        kw = _minimal_kwargs()
        kw["model"] = {"dim": "sixty-four", "n_layers": 2, "n_heads": 2, "ffn_dim": 128}
        with pytest.raises(ValidationError):
            ConfigRoot.model_validate(kw)

    def test_float_seed_rejected(self) -> None:
        kw = _minimal_kwargs()
        kw["training"] = {"seed": 7.5, "tokens": 1024, "batch_size": 4, "lr": 1e-4}
        with pytest.raises(ValidationError):
            ConfigRoot.model_validate(kw)

    def test_lr_is_float(self) -> None:
        kw = _minimal_kwargs()
        cfg = ConfigRoot.model_validate(kw)
        assert cfg.training.lr == 1e-4


# --- impossible values rejected via constraints ----------------------------


class TestImpossibleValues:
    def test_negative_dim_rejected(self) -> None:
        kw = _minimal_kwargs()
        kw["model"] = {"dim": -64, "n_layers": 2, "n_heads": 2, "ffn_dim": 128}
        with pytest.raises(ValidationError):
            ConfigRoot.model_validate(kw)

    def test_zero_tokens_rejected(self) -> None:
        kw = _minimal_kwargs()
        kw["training"] = {"seed": 7, "tokens": 0, "batch_size": 4, "lr": 1e-4}
        with pytest.raises(ValidationError):
            ConfigRoot.model_validate(kw)

    def test_zero_batch_size_rejected(self) -> None:
        kw = _minimal_kwargs()
        kw["training"] = {"seed": 7, "tokens": 1024, "batch_size": 0, "lr": 1e-4}
        with pytest.raises(ValidationError):
            ConfigRoot.model_validate(kw)

    def test_non_positive_lr_rejected(self) -> None:
        kw = _minimal_kwargs()
        kw["training"] = {"seed": 7, "tokens": 1024, "batch_size": 4, "lr": 0.0}
        with pytest.raises(ValidationError):
            ConfigRoot.model_validate(kw)


# --- cross-field validation ------------------------------------------------


class TestCrossFieldValidation:
    def test_dim_not_divisible_by_heads_rejected(self) -> None:
        # dim must be divisible by n_heads (attention head dimension integer).
        kw = _minimal_kwargs()
        kw["model"] = {"dim": 66, "n_layers": 2, "n_heads": 4, "ffn_dim": 128}
        with pytest.raises(ValidationError):
            ConfigRoot.model_validate(kw)

    def test_dim_divisible_by_heads_accepted(self) -> None:
        kw = _minimal_kwargs()
        kw["model"] = {"dim": 64, "n_layers": 2, "n_heads": 4, "ffn_dim": 128}
        cfg = ConfigRoot.model_validate(kw)
        assert cfg.model.n_heads == 4

    def test_checkpoint_interval_must_divide_or_be_at_least_tokens(self) -> None:
        # checkpoint_interval must be > 0 and <= tokens.
        kw = _minimal_kwargs()
        kw["checkpointing"] = {"interval_tokens": 10_000}
        # tokens here is 1024; interval > tokens is implausible.
        with pytest.raises(ValidationError):
            ConfigRoot.model_validate(kw)


# --- immutability: frozen models -------------------------------------------


class TestImmutability:
    def test_root_is_frozen(self) -> None:
        cfg = ConfigRoot.model_validate(_minimal_kwargs())
        with pytest.raises(ValidationError):
            cfg.run = RunConfig(name="other")  # type: ignore[misc]

    def test_nested_section_is_frozen(self) -> None:
        cfg = ConfigRoot.model_validate(_minimal_kwargs())
        with pytest.raises(ValidationError):
            cfg.run.name = "other"  # type: ignore[misc]

    def test_config_root_config_is_frozen(self) -> None:
        cfg = ConfigRoot.model_validate(_minimal_kwargs())
        assert cfg.model_config.get("frozen") is True
