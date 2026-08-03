"""Typed D0 primitive-semantics amendment configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "D0AttentionSemanticsConfig",
    "D0PrimitiveSemanticsConfig",
    "D0RMSNormSemanticsConfig",
    "D0RoPESemanticsConfig",
    "D0SwiGLUSemanticsConfig",
]


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)


class D0RoPESemanticsConfig(_Section):
    pairing: Literal["adjacent_pairs"]
    pair_indices: Literal["(0,1),(2,3),..."]
    head_dimension_requirement: Literal["positive_even_integer"]
    position_indexing: Literal["zero_based"]
    inverse_frequency_formula: Literal["inv_freq[i]=rope_base**(-2*i/head_dimension)"]
    inverse_frequency_index_range: Literal["i_in_0_through_head_dimension_over_2_minus_1"]
    angle_formula: Literal["theta=position*inv_freq[i]"]
    rotation_formula: tuple[
        Literal["rotated_even=x_even*cos(theta)-x_odd*sin(theta)"],
        Literal["rotated_odd=x_even*sin(theta)+x_odd*cos(theta)"],
    ] = Field(..., strict=False)
    application: tuple[Literal["query"], Literal["key"]] = Field(..., strict=False)
    value_rotated: Literal[False]
    tensor_order_before_rotation: Literal["batch_heads_sequence_head_dimension"]
    operation_order: Literal[
        "project_then_reshape_heads_then_apply_rope_then_compute_attention_scores"
    ]
    scaling: Literal["none"]
    cache_trainable: Literal[False]
    cache_persistent: Literal[False]
    state_dict_keys: tuple[str, ...] = Field(..., max_length=0, strict=False)


class D0AttentionSemanticsConfig(_Section):
    score_formula: Literal["(Q@K_transpose)/sqrt(head_dimension)"]
    causal_permission: Literal["key_position<=query_position"]
    masked_probability_mass: float = Field(..., ge=0.0, le=0.0, allow_inf_nan=False)
    softmax_accumulation_dtype_under_reduced_precision: Literal["float32"]
    softmax_output_conversion: Literal[
        "convert_to_declared_compute_dtype_before_value_aggregation_when_required"
    ]


class D0RMSNormSemanticsConfig(_Section):
    formula: Literal["x*rsqrt(mean(x^2,axis=-1,keepdims=true)+epsilon)*weight"]
    statistics_accumulation_dtype_under_reduced_precision: Literal["float32"]
    centering: Literal[False]
    bias: Literal[False]
    trainable_parameters: tuple[Literal["weight"]] = Field(..., strict=False)
    weight_initial_value: float = Field(..., ge=1.0, le=1.0, allow_inf_nan=False)


class D0SwiGLUSemanticsConfig(_Section):
    formula: Literal["down_proj(silu(gate_proj(x))*up_proj(x))"]
    gate_projection_bias: Literal[False]
    up_projection_bias: Literal[False]
    down_projection_bias: Literal[False]


class D0PrimitiveSemanticsConfig(_Section):
    amendment_path: Literal["experiments/d0/primitive-semantics-amendment-v1.proposed.json"]
    amendment_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    rope: D0RoPESemanticsConfig
    attention: D0AttentionSemanticsConfig
    rmsnorm: D0RMSNormSemanticsConfig
    swiglu: D0SwiGLUSemanticsConfig

    @model_validator(mode="after")
    def _amendment_identity(self) -> D0PrimitiveSemanticsConfig:
        if self.amendment_sha256 != (
            "5bc196942697833c4dfe9f227b16e2b69c6cafd07b05396a70650ecb34cebab0"
        ):
            raise ValueError("D0 primitive-semantics amendment identity changed.")
        return self
