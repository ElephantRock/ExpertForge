"""Pre-norm RMSNorm primitive.

Implements the ratified formula
(D0 primitive semantics, ``rmsnorm`` section)::

    y = x * rsqrt(mean(x**2, axis=-1, keepdims=True) + epsilon) * weight

The squared-mean statistics accumulate in float32 under reduced precision. The
single trainable parameter is the per-feature scale ``weight``, initialised to
1.0. There is no bias.
"""

from __future__ import annotations

# Resolve :mod:`torch` through the lazy helper first so that a missing optional
# dependency surfaces as a typed, actionable error pointing at the ``d0-model``
# extra. The subsequent bare imports then satisfy static typing.
from expertforge.d0.model._torch import require_torch as _require_torch

_require_torch()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

__all__ = ["RMSNorm"]


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (pre-norm variant).

    Parameters
    ----------
    dim:
        Feature dimension normalised along the final axis.
    epsilon:
        Numerical-stability constant added to the squared mean before the
        reciprocal square root.
    """

    def __init__(self, dim: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulate the squared-mean statistics in float32 for numerical
        # determinism under reduced precision (ratified semantics).
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        variance = x32.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.epsilon)
        normed = (x32 * inv_rms).to(orig_dtype)
        return normed * self.weight
