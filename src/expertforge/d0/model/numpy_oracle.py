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

import numpy as np

__all__ = [
    "numpy_attention",
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
