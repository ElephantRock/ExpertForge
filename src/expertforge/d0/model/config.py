"""Frozen D0 dense-decoder model configuration.

The two profiles (qualification and canonical) are the only supported
instantiations of the D0 contract. Their dimensions are pinned by the ratified
baseline contract and the parameter inventory
(``experiments/d0/parameter-inventory-v1.json``); changing them would alter the
parameter count and break the lineage contract.

The :class:`D0ModelConfig` dataclass is the single source of truth for the
architectural hyperparameters consumed by the model primitives. It is frozen so
that downstream code cannot mutate the contract in place.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "D0ModelConfig",
    "canonical_config",
    "qualification_config",
]


@dataclass(frozen=True, slots=True)
class D0ModelConfig:
    """Architectural configuration for the D0 dense decoder.

    Attributes mirror the ratified D0 primitive semantics
    (``configs/d0/qualification-primitive-v2.yaml``). Defaults encode the
    tokenizer vocabulary (GPT-NeoX-20B, 50257 tokens) and the primitive
    constants (RMSNorm epsilon, RoPE base, sequence length, tied embeddings,
    bias-free projections).
    """

    n_layers: int
    dim: int
    n_heads: int
    head_dim: int
    ffn_dim: int
    vocab_size: int = 50257
    rmsnorm_epsilon: float = 1e-5
    rope_base: float = 10000.0
    max_seq_len: int = 1024
    tie_embeddings: bool = True
    bias: bool = False

    def __post_init__(self) -> None:
        if self.bias:
            raise ValueError("D0 contracts are bias-free; bias must be False.")
        if not self.tie_embeddings:
            raise ValueError(
                "D0 contracts tie the token embedding and output projection; "
                "tie_embeddings must be True."
            )
        if self.head_dim != self.dim // self.n_heads:
            raise ValueError(
                "D0 requires head_dim == dim // n_heads; got "
                f"head_dim={self.head_dim}, dim={self.dim}, n_heads={self.n_heads}."
            )
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads}).")
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer (adjacent-pair RoPE).")


def qualification_config() -> D0ModelConfig:
    """Return the qualification profile (19,685,888 trainable parameters)."""
    return D0ModelConfig(
        n_layers=8,
        dim=256,
        n_heads=4,
        head_dim=64,
        ffn_dim=768,
    )


def canonical_config() -> D0ModelConfig:
    """Return the canonical profile (76,738,176 trainable parameters)."""
    return D0ModelConfig(
        n_layers=12,
        dim=576,
        n_heads=9,
        head_dim=64,
        ffn_dim=1536,
    )
