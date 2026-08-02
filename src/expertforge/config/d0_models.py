"""Typed additive D0 execution-contract configuration sections."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "D0ArchitectureConfig",
    "D0BatchConfig",
    "D0Config",
    "D0EvaluationProtocolConfig",
    "D0InitializationConfig",
    "D0ModelContractConfig",
    "D0OptimizerConfig",
    "D0PrecisionConfig",
    "D0ScheduleConfig",
    "D0SeedConfig",
    "D0SequenceConfig",
    "D0SourceConfig",
    "D0ThresholdConfig",
]


class _Section(BaseModel):
    """Frozen, strict D0 configuration section."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
    )


class D0SourceConfig(_Section):
    """Immutable contract and source-manifest bindings."""

    contract_path: str = Field(..., min_length=1)
    dataset_repository: str = Field(..., min_length=1)
    dataset_revision: str = Field(..., pattern=r"^[0-9a-f]{40}$")
    dataset_configuration: str = Field(..., min_length=1)
    dataset_manifest_path: str = Field(..., min_length=1)
    dataset_manifest_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    tokenizer_repository: str = Field(..., min_length=1)
    tokenizer_revision: str = Field(..., pattern=r"^[0-9a-f]{40}$")
    tokenizer_manifest_path: str = Field(..., min_length=1)
    tokenizer_manifest_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")


class D0ArchitectureConfig(_Section):
    """Frozen dense-baseline architecture semantics."""

    family: Literal["decoder_only_transformer"]
    attention: Literal["exact_causal_multi_head_self_attention"]
    normalization: Literal["pre_norm_rmsnorm"]
    position_encoding: Literal["rope"]
    feed_forward: Literal["swiglu"]
    objective: Literal["next_token_prediction"]
    embedding_output_tied: Literal[True]
    biases: Literal[False]
    dropout: float = Field(..., ge=0.0, le=0.0, allow_inf_nan=False)
    rmsnorm_epsilon: float = Field(..., gt=0.0, allow_inf_nan=False)
    rope_base: int = Field(..., gt=0)
    rope_dimension_coverage: Literal["full_head_dimension"]
    rope_scaling: Literal["none"]


class D0SequenceConfig(_Section):
    """Exact input/target packing and token-accounting semantics."""

    model_context_tokens: int = Field(..., gt=0)
    packed_source_window_tokens: int = Field(..., gt=0)
    input_tokens_per_sequence: int = Field(..., gt=0)
    target_tokens_per_sequence: int = Field(..., gt=0)
    cursor_advance_tokens_per_sequence: int = Field(..., gt=0)
    input_lookback_overlap_tokens: int = Field(..., ge=0)
    cross_document_attention: bool
    padding_used: bool
    all_target_positions_in_loss: bool
    processed_token_definition: Literal["non_padding_target_tokens_participating_in_training_loss"]

    @model_validator(mode="after")
    def _packing_arithmetic(self) -> D0SequenceConfig:
        if self.packed_source_window_tokens != self.target_tokens_per_sequence + 1:
            raise ValueError("packed source window must contain exactly one lookback token.")
        if self.input_tokens_per_sequence != self.model_context_tokens:
            raise ValueError("input token count must equal model context.")
        if self.target_tokens_per_sequence != self.model_context_tokens:
            raise ValueError("target token count must equal model context.")
        if self.cursor_advance_tokens_per_sequence != self.target_tokens_per_sequence:
            raise ValueError("cursor advance must equal target-token contribution.")
        if self.input_lookback_overlap_tokens != 1:
            raise ValueError("D0 requires exactly one lookback-overlap token.")
        if self.padding_used:
            raise ValueError("D0 training must not use padding.")
        if not self.all_target_positions_in_loss:
            raise ValueError("D0 requires every target position in the loss.")
        return self


class D0ModelContractConfig(_Section):
    """Model identity and independently declared parameter accounting."""

    model_id: str = Field(..., min_length=1)
    head_dimension: int = Field(..., gt=0)
    trainable_parameters: int = Field(..., gt=0)
    non_trainable_parameters: int = Field(..., ge=0)
    parameter_formula: Literal["V*d + L*(4*d^2 + 3*d*f + 2*d) + d"]
    swiglu_rounding_rule: Literal["ceil_to_multiple((8/3)*model_width,256)"]
    rope_trainable_parameters: Literal[0]
    separate_output_head_parameters: Literal[0]


class D0BatchConfig(_Section):
    """Microbatch, accumulation, and global-token arithmetic."""

    devices: int = Field(..., gt=0)
    microbatch_sequences_per_device: int = Field(..., gt=0)
    gradient_accumulation_steps: int = Field(..., gt=0)
    global_sequences_per_update: int = Field(..., gt=0)
    target_tokens_per_update: int = Field(..., gt=0)


class D0OptimizerConfig(_Section):
    """Optimizer and gradient semantics."""

    name: Literal["AdamW"]
    beta1: float = Field(..., ge=0.0, lt=1.0, allow_inf_nan=False)
    beta2: float = Field(..., ge=0.0, lt=1.0, allow_inf_nan=False)
    epsilon: float = Field(..., gt=0.0, allow_inf_nan=False)
    weight_decay: float = Field(..., ge=0.0, allow_inf_nan=False)
    weight_decay_includes: tuple[str, ...] = Field(..., min_length=1, strict=False)
    weight_decay_excludes: tuple[str, ...] = Field(..., min_length=1, strict=False)
    gradient_clip_global_l2_norm: float = Field(..., gt=0.0, allow_inf_nan=False)
    loss_reduction: Literal["mean_over_all_target_tokens_in_optimizer_update"]


class D0InitializationConfig(_Section):
    """Initialization contract."""

    embedding_qkv_swiglu_gate_up_std: float = Field(..., gt=0.0, allow_inf_nan=False)
    attention_output_and_swiglu_down_std: Literal["0.02/sqrt(2*layers)"]
    rmsnorm_weight: float = Field(..., gt=0.0, allow_inf_nan=False)
    all_biases_absent: Literal[True]


class D0ScheduleConfig(_Section):
    """Exact learning-rate schedule, budget, and evidence cadences."""

    peak_learning_rate: float = Field(..., gt=0.0, allow_inf_nan=False)
    warmup_updates: int = Field(..., ge=0)
    final_learning_rate: float = Field(..., gt=0.0, allow_inf_nan=False)
    decay: Literal["cosine"]
    training_target_tokens: int = Field(..., gt=0)
    optimizer_updates: int = Field(..., gt=0)
    validation_interval_updates: int = Field(..., gt=0)
    checkpoint_interval_updates: int = Field(..., gt=0)
    generation_interval_updates: int = Field(..., gt=0)
    profiling_interval_updates: int = Field(..., gt=0)

    @model_validator(mode="after")
    def _schedule_order(self) -> D0ScheduleConfig:
        if self.warmup_updates >= self.optimizer_updates:
            raise ValueError("warmup must end before the final optimizer update.")
        if self.final_learning_rate >= self.peak_learning_rate:
            raise ValueError("final learning rate must be below peak learning rate.")
        return self


class D0PrecisionConfig(_Section):
    """Hardware, precision, and fallback policy."""

    primary_accelerator: Literal["single_CUDA_accelerator_with_native_BF16"]
    minimum_device_memory_bytes: int = Field(..., gt=0)
    minimum_host_memory_bytes: int = Field(..., gt=0)
    minimum_free_local_storage_bytes: int = Field(..., gt=0)
    parameter_and_compute_dtype: Literal["bfloat16"]
    gradient_accumulation_dtype: Literal["float32"]
    loss_reduction_dtype: Literal["float32"]
    optimizer_state_dtype: Literal["float32"]
    master_parameter_dtype: Literal["float32"]
    canonical_fallback: Literal["none_fail_closed"]
    qualification_fallback: Literal["float32_as_distinct_non_equivalent_attempt"]
    fp16_authorized: Literal[False]


class D0SeedConfig(_Section):
    """Deterministic seed and recovery semantics."""

    master_seed: int
    substream_derivation: Literal["sha256_domain_separation"]
    canonical_generation_seeds: tuple[int, ...] = Field(..., min_length=1, strict=False)
    qualification_requires_uninterrupted_and_interrupted_resumed_match: bool
    canonical_requires_genuine_interruption_into_distinct_attempt: bool
    training_seed_variance_claimed: bool


class D0EvaluationProtocolConfig(_Section):
    """Fixed validation, generation, throughput, and inference protocol."""

    validation_target_tokens_per_boundary: int = Field(..., gt=0)
    validation_loss_units: Literal["natural_log_nats"]
    perplexity: Literal["exp(validation_loss)"]
    generation_prompt_count: int = Field(..., gt=0)
    generation_max_new_tokens: int = Field(..., gt=0)
    generation_temperature: float = Field(..., gt=0.0, allow_inf_nan=False)
    generation_top_p: float = Field(..., gt=0.0, le=1.0, allow_inf_nan=False)
    generation_top_k: int | None = Field(default=None, gt=0)
    generation_repetition_penalty: float = Field(..., gt=0.0, allow_inf_nan=False)
    generation_stop_token_id: int = Field(..., ge=0)
    throughput_measurement_updates: int = Field(..., gt=0)
    throughput_warmup_updates: int = Field(..., ge=0)
    inference_batch_size: int = Field(..., gt=0)
    inference_input_tokens: int = Field(..., gt=0)
    inference_generated_tokens: int = Field(..., gt=0)
    inference_warmup_runs: int = Field(..., ge=0)
    inference_measured_runs: int = Field(..., gt=0)


class D0ThresholdConfig(_Section):
    """Profile acceptance plus common failure and kill thresholds."""

    minimum_final_validation_loss_improvement_nats: float = Field(..., gt=0.0, allow_inf_nan=False)
    maximum_final_loss_above_best_prior_nats: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    maximum_consecutive_regressing_validation_boundaries: int | None = Field(default=None, ge=0)
    regression_boundary_delta_nats: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    minimum_final_to_initial_throughput_ratio: float | None = Field(
        default=None, gt=0.0, le=1.0, allow_inf_nan=False
    )
    maximum_skipped_updates: int = Field(..., ge=0)
    maximum_peak_device_memory_fraction: float = Field(..., gt=0.0, le=1.0, allow_inf_nan=False)
    requires_exact_checkpoint_round_trip: bool | None = None
    requires_locked_environment_resume_equality: bool | None = None
    maximum_checkpoint_write_seconds: int = Field(..., gt=0)
    maximum_checkpoint_read_seconds: int = Field(..., gt=0)
    maximum_peak_host_memory_fraction: float = Field(..., gt=0.0, le=1.0, allow_inf_nan=False)
    minimum_loss_improvement_at_quarter_budget_nats: float = Field(..., gt=0.0, allow_inf_nan=False)
    maximum_rejected_recovery_attempts_before_kill: int = Field(..., ge=0)
    kill_on_any_non_finite_value: bool
    kill_on_any_skipped_optimizer_update: bool
    kill_on_any_checkpoint_or_resume_state_mismatch: bool
    maximum_out_of_memory_failures_after_remediation: int = Field(..., ge=0)
    maximum_failed_recovery_attempts: int = Field(..., ge=0)
    minimum_loss_improvement_at_half_budget_nats: float = Field(..., gt=0.0, allow_inf_nan=False)


class D0Config(_Section):
    """Complete additive D0 execution-contract binding."""

    profile: Literal["qualification", "canonical"]
    sources: D0SourceConfig
    architecture: D0ArchitectureConfig
    sequence: D0SequenceConfig
    model: D0ModelContractConfig
    batch: D0BatchConfig
    optimizer: D0OptimizerConfig
    initialization: D0InitializationConfig
    schedule: D0ScheduleConfig
    precision: D0PrecisionConfig
    seeds: D0SeedConfig
    evaluation: D0EvaluationProtocolConfig
    thresholds: D0ThresholdConfig

    @model_validator(mode="after")
    def _execution_arithmetic(self) -> D0Config:
        expected_sequences = (
            self.batch.devices
            * self.batch.microbatch_sequences_per_device
            * self.batch.gradient_accumulation_steps
        )
        if self.batch.global_sequences_per_update != expected_sequences:
            raise ValueError("D0 global sequence batch arithmetic mismatch.")
        expected_tokens = (
            self.batch.global_sequences_per_update * self.sequence.target_tokens_per_sequence
        )
        if self.batch.target_tokens_per_update != expected_tokens:
            raise ValueError("D0 target-token batch arithmetic mismatch.")
        if self.schedule.training_target_tokens != (
            self.schedule.optimizer_updates * self.batch.target_tokens_per_update
        ):
            raise ValueError("D0 token budget/update arithmetic mismatch.")
        return self
