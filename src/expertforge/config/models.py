"""Frozen Pydantic configuration models (Issue #5 decision §2).

The optional D0 section is an additive format-version-1 extension. Legacy
configuration files do not acquire behavioral fingerprint changes: canonical
serialization omits the section when it is absent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from expertforge.config.d0_models import D0Config

__all__ = [
    "CONFIG_FORMAT_VERSION",
    "ArtifactConfig",
    "CheckpointConfig",
    "ConfigRoot",
    "D0Config",
    "DataConfig",
    "EvaluationConfig",
    "HardwareConfig",
    "LoggingConfig",
    "ModelConfig",
    "RunConfig",
    "TokenizerConfig",
    "TrainingConfig",
]

CONFIG_FORMAT_VERSION: int = 1


class _Section(BaseModel):
    """Frozen, strict configuration section with unknown fields forbidden."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
    )


class RunConfig(_Section):
    """Identity and lifecycle of a single run."""

    name: str = Field(..., min_length=1, description="Stable run identifier.")
    description: str = Field(default="", description="Free-form run description.")


class DataConfig(_Section):
    """Versioned dataset identity."""

    dataset_id: str = Field(default="fixture-tiny", description="Immutable dataset identifier.")
    split: str = Field(default="train", description="Which split to consume.")
    seq_len: int = Field(default=128, gt=0, description="Packed sequence length in tokens.")


class TokenizerConfig(_Section):
    """Tokenizer identity."""

    tokenizer_id: str = Field(default="fixture-byte", description="Immutable tokenizer identifier.")
    vocab_size: int = Field(default=256, gt=0, description="Vocabulary size.")


class ModelConfig(_Section):
    """Decoder-only dense model dimensions."""

    dim: int = Field(..., gt=0, description="Model hidden dimension.")
    n_layers: int = Field(..., gt=0, description="Number of transformer blocks.")
    n_heads: int = Field(..., gt=0, description="Number of attention heads.")
    ffn_dim: int = Field(..., gt=0, description="Feed-forward inner dimension.")

    @model_validator(mode="after")
    def _dim_divisible_by_heads(self) -> ModelConfig:
        if self.dim % self.n_heads != 0:
            raise ValueError(
                f"model.dim ({self.dim}) must be divisible by model.n_heads ({self.n_heads})."
            )
        return self


class TrainingConfig(_Section):
    """Training-loop parameters and root reproducibility policy."""

    seed: int = Field(..., description="Root RNG seed for deterministic derived streams.")
    determinism_mode: Literal["reproducible", "performance"] = Field(default="reproducible")
    unsupported_determinism: Literal["error", "warn"] = Field(default="error")
    tokens: int = Field(..., gt=0, description="Training-token budget for this run.")
    batch_size: int = Field(..., gt=0, description="Effective batch size in sequences.")
    seq_len: int | None = Field(default=None, gt=0)
    lr: float = Field(..., gt=0.0, allow_inf_nan=False)


class EvaluationConfig(_Section):
    """Evaluation cadence and optional smoke-gate threshold."""

    eval_interval_tokens: int = Field(default=1024, gt=0)
    loss_improvement_threshold: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
    )


class CheckpointConfig(_Section):
    """Checkpoint cadence."""

    interval_tokens: int = Field(default=512, gt=0)


class LoggingConfig(_Section):
    """Structured logging controls."""

    log_interval_steps: int = Field(default=10, gt=0)
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    console_enabled: bool = Field(default=True)
    fsync_interval_records: int = Field(default=100, ge=1)


class ArtifactConfig(_Section):
    """Run-artifact location policy."""

    dir: str = Field(default="runs")


class HardwareConfig(_Section):
    """Hardware/runtime target."""

    device: Literal["auto", "cpu", "cuda"] = Field(default="auto")
    dtype: Literal["bf16", "fp16", "fp32"] = Field(default="fp32")


class ConfigRoot(_Section):
    """The fully resolved run configuration. Format-versioned."""

    format_version: int = Field(
        default=CONFIG_FORMAT_VERSION,
        description="On-disk configuration format version.",
    )

    run: RunConfig
    data: DataConfig = Field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    d0: D0Config | None = Field(default=None, exclude_if=lambda value: value is None)

    @field_validator("format_version")
    @classmethod
    def _check_format_version(cls, value: int) -> int:
        if value != CONFIG_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported configuration format_version {value}; this version of "
                f"ExpertForge understands format_version={CONFIG_FORMAT_VERSION}."
            )
        return value

    @model_validator(mode="after")
    def _checkpoint_interval_within_budget(self) -> ConfigRoot:
        if self.checkpointing.interval_tokens > self.training.tokens:
            raise ValueError(
                f"checkpointing.interval_tokens "
                f"({self.checkpointing.interval_tokens}) must not exceed "
                f"training.tokens ({self.training.tokens})."
            )
        return self
