"""SwiGLU feed-forward primitive.

Implements the ratified D0 feed-forward semantics
(``configs/d0/qualification-primitive-v2.yaml`` §``swiglu``)::

    y = down_proj(silu(gate_proj(x)) * up_proj(x))

All three projections are bias-free. ``gate_proj`` and ``up_proj`` map
``dim -> ffn_dim`` (stored weight shape ``[ffn_dim, dim]``); ``down_proj`` maps
``ffn_dim -> dim`` (stored weight shape ``[dim, ffn_dim]``), matching the
parameter inventory.
"""

from __future__ import annotations

from expertforge.d0.model._torch import require_torch as _require_torch

_require_torch()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

__all__ = ["SwiGLU"]


class SwiGLU(nn.Module):
    """Bias-free SwiGLU feed-forward block.

    Parameters
    ----------
    dim:
        Model width (input to gate/up projections, output of down projection).
    ffn_dim:
        Hidden width of the gate and up projections.
    """

    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = nn.functional.silu(self.gate_proj(x))
        out: torch.Tensor = self.down_proj(gated * self.up_proj(x))
        return out
