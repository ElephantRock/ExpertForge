"""Full-head RoPE primitive (adjacent-pair rotation).

Implements the ratified D0 RoPE semantics. This is a **pure function** with no
trainable parameters and no persistent cache: the inverse frequencies are
derived from ``rope_base`` and ``head_dim`` on every call, so the module
contributes zero keys to the ``state_dict`` and zero trainable parameters.

Semantics (``configs/d0/qualification-primitive-v2.yaml`` §``rope``):

- Pairing: adjacent pairs ``(0, 1), (2, 3), ...`` of head-dimension channels.
- Position indexing: zero-based.
- ``inv_freq[i] = rope_base ** (-2*i / head_dim)`` for
  ``i in 0 .. head_dim//2 - 1``.
- ``theta = position * inv_freq[i]``.
- ``rotated_even = x_even * cos(theta) - x_odd * sin(theta)``
- ``rotated_odd  = x_even * sin(theta) + x_odd * cos(theta)``
- Applied to queries and keys only (never values).

The tensor order before rotation is ``(batch, heads, sequence, head_dim)``; the
operation order is project → reshape heads → apply RoPE → compute attention
scores.
"""

from __future__ import annotations

from expertforge.d0.model._torch import require_torch as _require_torch

_require_torch()

import torch  # noqa: E402

__all__ = ["apply_rope"]


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_base: float,
    head_dim: int,
) -> torch.Tensor:
    """Apply adjacent-pair RoPE to ``x`` along its final ``head_dim`` axis.

    Parameters
    ----------
    x:
        Tensor of shape ``(..., head_dim)`` — typically
        ``(batch, heads, sequence, head_dim)`` for Q or K.
    positions:
        Integer tensor broadcastable to the leading axes of ``x`` *excluding*
        the final ``head_dim`` axis, i.e. shape ``(batch, heads, sequence)`` or
        a broadcast-compatible prefix. Zero-based token positions.
    rope_base:
        RoPE base frequency (``10000`` for D0).
    head_dim:
        Per-head channel count; must be a positive even integer and equal to the
        final size of ``x``.

    Returns
    -------
    Tensor
        Same shape and dtype as ``x``.
    """
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer for adjacent-pair RoPE.")
    if x.shape[-1] != head_dim:
        raise ValueError(f"x final dimension ({x.shape[-1]}) must equal head_dim ({head_dim}).")

    orig_dtype = x.dtype
    half = head_dim // 2

    i = torch.arange(half, dtype=torch.float32, device=x.device)
    inv_freq = rope_base ** (-2.0 * i / float(head_dim))  # (half,)

    # positions: (...,) broadcastable to the leading axes. Reshape to multiply
    # against inv_freq of shape (half,).
    positions = positions.to(torch.float32)
    # theta shape: positions[..., None] * inv_freq[None] -> (..., half)
    theta = positions.unsqueeze(-1) * inv_freq.reshape(*[1] * positions.dim(), half)
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    x32 = x.to(torch.float32)
    x_even = x32[..., 0::2]  # channels 0, 2, 4, ... (pair even members)
    x_odd = x32[..., 1::2]  # channels 1, 3, 5, ... (pair odd members)

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # Interleave back into adjacent-pair order.
    out = torch.empty_like(x32)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd

    return out.to(orig_dtype)
