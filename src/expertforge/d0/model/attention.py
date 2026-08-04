"""Exact causal multi-head self-attention primitive.

Implements the ratified D0 attention semantics
(``configs/d0/qualification-primitive-v2.yaml`` §``attention``):

- Separate bias-free Q, K, V projections and an output projection.
- Scores ``= (Q @ K^T) / sqrt(head_dim)``.
- Causal mask: a key position ``k`` may attend to query position ``q`` iff
  ``k <= q`` (future keys are masked to probability mass 0).
- Softmax accumulates in float32 under reduced precision.
- Operation order: project → reshape heads → apply RoPE to Q and K → compute
  scores.

All four projections are ``nn.Linear(dim, dim, bias=False)`` so the stored
weight has shape ``[dim, dim]`` matching the parameter inventory.
"""

from __future__ import annotations

from expertforge.d0.model._torch import require_torch as _require_torch

_require_torch()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from expertforge.d0.model.rope import apply_rope  # noqa: E402

__all__ = ["CausalSelfAttention"]


class CausalSelfAttention(nn.Module):
    """Bias-free causal multi-head self-attention with full-head RoPE.

    Parameters
    ----------
    dim:
        Model width (input and output of every projection).
    n_heads:
        Number of attention heads.
    head_dim:
        Per-head channel count.
    rope_base:
        RoPE base frequency.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, heads, sequence, head_dim).
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (never V). Zero-based positions per sequence.
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        # Broadcast positions to (batch, heads, sequence) — RoPE operates on
        # the final head_dim axis, leading axes are batch/heads.
        positions = positions.view(1, 1, seq_len)
        q = apply_rope(q, positions, self.rope_base, self.head_dim)
        k = apply_rope(k, positions, self.rope_base, self.head_dim)

        # Scores = (Q @ K^T) / sqrt(head_dim).
        scores = torch.matmul(q, k.transpose(-2, -1)) / (float(self.head_dim) ** 0.5)

        # Causal mask: key_position <= query_position. Masked mass is -inf so
        # the softmax assigns exactly 0 probability to future keys.
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~causal, float("-inf"))

        # Softmax in float32 for deterministic accumulation.
        probs = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)

        # Aggregate values and project out.
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out: torch.Tensor = self.out_proj(context)
        return out
