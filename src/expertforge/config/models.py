"""Frozen Pydantic configuration models (Issue #5 decision §2).

All models use::

    ConfigDict(frozen=True, extra="forbid", validate_default=True)

with strict/constrained scalars and explicit cross-field validators. These are
the canonical typed configuration sections. YAML is the authoring format only;
these models are the validated in-memory representation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

__all__ = [
    "CONFIG_FORMAT_VERSION",
    "ArtifactConfig",
    "CheckpointConfig",
    "ConfigRoot",
    "DataConfig",
    "EvaluationConfig",
    "HardwareConfig",
    "LoggingConfig",
    "ModelConfig",
    "RunConfig",
    "TokenizerConfig",
    "TrainingConfig",
]


# The configuration format version. Bumped when the on-disk schema changes in a
# way that invalidates existing source files. Issue #5 decision §1/§2 require a
# format/version field and compatibility policy.
CONFIG_FORMAT_VERSION: int = 1


class _Section(BaseModel):
    """Base for every configuration section: frozen, forbid extras, validate
    defaults."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
    )


# --- project / run ---------------------------------------------------------


class RunConfig(_Section):
    """Identity and lifecycle of a single run."""

    name: str = Field(..., min_length=1, description="Stable run identifier.")
    # Optional human description; no machine semantics.
    description: str = Field(default="", description="Free-form run description.")


# --- data ------------------------------------------------------------------


class DataConfig(_Section):
    """Versioned dataset identity. The dataset itself is out of scope for #5;
    these fields establish the immutable-identity contract early."""

    dataset_id: str = Field(default="fixture-tiny", description="Immutable dataset identifier.")
    split: str = Field(default="train", description="Which split to consume.")
    seq_len: int = Field(default=128, gt=0, description="Packed sequence length in tokens.")


# --- tokenizer -------------------------------------------------------------


class TokenizerConfig(_Section):
    """Tokenizer identity. The tokenizer is a versioned model component
    (doctrine/data-and-training.md §2)."""

    tokenizer_id: str = Field(default="fixture-byte", description="Immutable tokenizer identifier.")
    vocab_size: int = Field(default=256, gt=0, description="Vocabulary size.")


# --- model -----------------------------------------------------------------


class ModelConfig(_Section):
    """Decoder-only dense model dimensions (D0 policy,
    doctrine/model-lineage.md §1). dim must be divisible by n_heads."""

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


# --- training --------------------------------------------------------------


class TrainingConfig(_Section):
    """Training-loop parameters. Progress is measured primarily in tokens
    (doctrine/data-and-training.md §3.1)."""

    seed: int = Field(..., description="Base RNG seed; derived sub-seeds are deterministic.")
    tokens: int = Field(..., gt=0, description="Training-token budget for this run.")
    batch_size: int = Field(..., gt=0, description="Effective batch size in sequences.")
    seq_len: int | None = Field(default=None, gt=0, description="Override for data.seq_len if set.")
    lr: float = Field(..., gt=0.0, description="Peak learning rate.")


# --- evaluation ------------------------------------------------------------


class EvaluationConfig(_Section):
    """Evaluation cadence. Concrete metrics arrive with later issues."""

    eval_interval_tokens: int = Field(
        default=1024, gt=0, description="Evaluate every N processed tokens."
    )


# --- checkpointing ---------------------------------------------------------


class CheckpointConfig(_Section):
    """Checkpoint cadence. interval_tokens must not exceed the token budget."""

    interval_tokens: int = Field(
        default=512, gt=0, description="Write a checkpoint every N processed tokens."
    )


# --- logging ---------------------------------------------------------------


class LoggingConfig(_Section):
    """Structured logging cadence."""

    log_interval_steps: int = Field(
        default=10, gt=0, description="Emit a log line every N optimizer steps."
    )
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")


# --- artifacts -------------------------------------------------------------


class ArtifactConfig(_Section):
    """Where run artifacts are written. Real artifacts live outside the repo
    (doctrine/collaboration.md §14); this only records the location policy."""

    dir: str = Field(default="runs", description="Run output directory (gitignored).")


# --- hardware / runtime ----------------------------------------------------


class HardwareConfig(_Section):
    """Hardware/runtime target. No GPU requirement in Milestone 0 (#4 non-goal)."""

    device: Literal["auto", "cpu", "cuda"] = Field(default="auto")
    dtype: Literal["bf16", "fp16", "fp32"] = Field(default="fp32")


# --- root ------------------------------------------------------------------


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

    @field_validator("format_version")
    @classmethod
    def _check_format_version(cls, v: int) -> int:
        if v != CONFIG_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported configuration format_version {v}; this version of "
                f"ExpertForge understands format_version={CONFIG_FORMAT_VERSION}."
            )
        return v

    @model_validator(mode="after")
    def _checkpoint_interval_within_budget(self) -> ConfigRoot:
        if self.checkpointing.interval_tokens > self.training.tokens:
            raise ValueError(
                f"checkpointing.interval_tokens "
                f"({self.checkpointing.interval_tokens}) must not exceed "
                f"training.tokens ({self.training.tokens})."
            )
        return self
