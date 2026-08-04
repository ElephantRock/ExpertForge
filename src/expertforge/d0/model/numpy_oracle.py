"""Independent NumPy reference implementations of the D0 primitives.

These SMALL, dependency-light functions exist solely to back the numerical
reference tests. They are deliberately written from the ratified primitive
specification rather than by importing the PyTorch modules, so a discrepancy
between the oracle and the PyTorch implementation signals a genuine bug rather
than a shared misunderstanding.

Each function operates on plain :class:`numpy.ndarray` inputs and returns plain
arrays. They use only :mod:`numpy`, which is a base dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from expertforge.d0.model.config import D0ModelConfig

__all__ = [
    "numpy_attention",
    "numpy_full_model_forward",
    "numpy_rmsnorm",
    "numpy_rope",
    "numpy_swiglu",
]


def numpy_rmsnorm(x: np.ndarray, weight: np.ndarray, epsilon: float) -> np.ndarray:
    """RMSNorm: ``x * rsqrt(mean(x**2, -1, keepdims) + eps) * weight``."""
    x32 = x.astype(np.float32)
    variance = np.mean(x32**2, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(variance + np.float32(epsilon))
    normed = x32 * inv_rms
    scaled: np.ndarray = normed * weight.astype(normed.dtype)
    return scaled


def numpy_rope(x: np.ndarray, positions: np.ndarray, rope_base: float, head_dim: int) -> np.ndarray:
    """Adjacent-pair RoPE applied along the final ``head_dim`` axis.

    ``x`` has shape ``(..., head_dim)``; ``positions`` has shape ``(...)``.
    """
    half = head_dim // 2
    i = np.arange(half, dtype=np.float32)
    inv_freq = rope_base ** (-2.0 * i / float(head_dim))  # (half,)
    pos = positions.astype(np.float32)
    theta = pos[..., None] * inv_freq[None, :]  # (..., half)
    cos = np.cos(theta)
    sin = np.sin(theta)

    x32 = x.astype(np.float32)
    x_even = x32[..., 0::2]
    x_odd = x32[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    out = np.empty_like(x32)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


def numpy_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, head_dim: int, mask: np.ndarray
) -> np.ndarray:
    """Single-head (per-head) causal attention.

    ``q``, ``k``, ``v`` have shape ``(seq, head_dim)``; ``mask`` is a boolean
    ``(seq, seq)`` array where ``True`` means the key may attend (allowed).
    Returns ``(seq, head_dim)``.
    """
    scores = (q.astype(np.float32) @ k.astype(np.float32).T) / np.sqrt(np.float32(head_dim))
    scores = np.where(mask, scores, np.float32("-inf"))
    # Numerically stable softmax in float32.
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(scores)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    out: np.ndarray = probs @ v.astype(np.float32)
    return out


def numpy_swiglu(
    x: np.ndarray, gate_weight: np.ndarray, up_weight: np.ndarray, down_weight: np.ndarray
) -> np.ndarray:
    """SwiGLU: ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Inputs follow the PyTorch ``nn.Linear`` convention where weights have shape
    ``[out_features, in_features]`` and the projection is ``x @ weight.T``.
    """
    gate = x.astype(np.float32) @ gate_weight.astype(np.float32).T
    up = x.astype(np.float32) @ up_weight.astype(np.float32).T
    silu_gate = gate / (1.0 + np.exp(-gate))
    out: np.ndarray = (silu_gate * up) @ down_weight.astype(np.float32).T
    return out


def _numpy_linear(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Bias-free ``nn.Linear`` forward: ``x @ weight.T`` in float32."""
    return x.astype(np.float32) @ weight.astype(np.float32).T


def _numpy_softmax_masked(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Numerically stable float32 softmax with a boolean allow-mask.

    ``mask`` is ``True`` where the key is allowed to attend; disallowed cells
    receive zero probability.
    """
    masked = np.where(mask, scores, np.float32("-inf"))
    masked = masked - np.max(masked, axis=-1, keepdims=True)
    exp = np.exp(masked)
    probs: np.ndarray = exp / np.sum(exp, axis=-1, keepdims=True)
    return probs


def numpy_full_model_forward(
    input_ids: np.ndarray,
    weights_dict: Mapping[str, np.ndarray],
    config: D0ModelConfig,
) -> np.ndarray:
    """Full D0 decoder forward pass in pure NumPy.

    Implements the ratified architecture end-to-end so that agreement with the
    PyTorch :class:`~expertforge.d0.model.transformer.D0Model` is strong evidence
    of correctness for every primitive **and** their composition:

    - Token embedding lookup (``token_embedding.weight``).
    - For each block ``i``: pre-attention ``RMSNorm`` -> causal multi-head
      attention with full-head RoPE -> residual; pre-FFN ``RMSNorm`` -> SwiGLU ->
      residual.
    - Final ``RMSNorm`` followed by the tied output projection
      (``x @ token_embedding.weight.T``).

    Parameters
    ----------
    input_ids:
        Integer array of shape ``(batch, seq_len)``.
    weights_dict:
        Mapping from state-dict names to float NumPy arrays, exactly as produced
        by ``D0Model.state_dict()`` (after ``.numpy()`` conversion). Weights are
        consumed by reference and not mutated.
    config:
        Frozen :class:`D0ModelConfig` providing ``n_layers``, ``n_heads``,
        ``head_dim``, ``rmsnorm_epsilon`` and ``rope_base``.

    Returns
    -------
    np.ndarray
        Logits of shape ``(batch, seq_len, vocab_size)`` in float32.
    """

    batch, seq_len = input_ids.shape
    n_heads = config.n_heads
    head_dim = config.head_dim
    epsilon = config.rmsnorm_epsilon
    rope_base = config.rope_base

    # Token embedding lookup. ``input_ids`` may be a python-int dtype; coerce.
    embedding = weights_dict["token_embedding.weight"].astype(np.float32)
    x = embedding[input_ids.astype(np.int64)].astype(np.float32)

    # Precompute RoPE positions broadcastable to (batch, heads, seq).
    positions = np.arange(seq_len, dtype=np.float32).reshape(1, 1, seq_len)
    causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))

    for i in range(config.n_layers):
        prefix = f"blocks.{i}."
        attn_norm_w = weights_dict[prefix + "attention_norm.weight"]
        x_normed = numpy_rmsnorm(x, attn_norm_w, epsilon)

        q = _numpy_linear(x_normed, weights_dict[prefix + "attention.q_proj.weight"])
        k = _numpy_linear(x_normed, weights_dict[prefix + "attention.k_proj.weight"])
        v = _numpy_linear(x_normed, weights_dict[prefix + "attention.v_proj.weight"])

        # Reshape to (batch, heads, seq, head_dim).
        q = q.reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE to Q and K (never V). numpy_rope takes positions with shape
        # broadcastable to the leading axes excluding the final head_dim axis.
        q = numpy_rope(q, positions, rope_base, head_dim)
        k = numpy_rope(k, positions, rope_base, head_dim)

        # Scores = (Q @ K^T) / sqrt(head_dim) with shape (batch, heads, seq, seq).
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(np.float32(head_dim))
        probs = _numpy_softmax_masked(scores, causal)

        context = probs @ v  # (batch, heads, seq, head_dim)
        # Merge heads back to (batch, seq, dim).
        context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, n_heads * head_dim)
        attn_out = _numpy_linear(context, weights_dict[prefix + "attention.out_proj.weight"])

        x = x + attn_out

        ffn_norm_w = weights_dict[prefix + "ffn_norm.weight"]
        x_normed_ffn = numpy_rmsnorm(x, ffn_norm_w, epsilon)
        ffn_out = numpy_swiglu(
            x_normed_ffn,
            weights_dict[prefix + "ffn.gate_proj.weight"],
            weights_dict[prefix + "ffn.up_proj.weight"],
            weights_dict[prefix + "ffn.down_proj.weight"],
        )
        x = x + ffn_out

    x = numpy_rmsnorm(x, weights_dict["final_norm.weight"], epsilon)
    # Tied output projection: x @ token_embedding.weight.T (no new parameter).
    logits: np.ndarray = x @ embedding.T
    return logits
