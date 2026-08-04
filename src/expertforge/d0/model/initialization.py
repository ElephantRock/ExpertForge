"""Frozen deterministic initialisation for the D0 model.

Applies the ratified D0 initialisation scheme
(``configs/d0/qualification-primitive-v2.yaml`` §``initialization``):

- Token embeddings, Q/K/V projections, SwiGLU gate/up projections:
  ``normal(0, 0.02)``.
- Attention output projection and SwiGLU down projection:
  ``normal(0, 0.02 / sqrt(2 * n_layers))`` (residual-scaling init).
- RMSNorm weights: ``1.0`` exactly.
- All biases absent (the projections are bias-free by construction).

Initialisation is fully determined by a caller-supplied :class:`torch.Generator`
so that two runs sharing the same seed and initialisation order reproduce the
identical parameter tensors. The sampling order below is part of the contract:
the helper walks the model in the order PyTorch materialises parameters and
fills each declared tensor in place.
"""

from __future__ import annotations

from expertforge.d0.model._torch import require_torch as _require_torch

_require_torch()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

__all__ = ["initialize_model"]

_EMBEDDING_QKV_SWIGLU_GATE_UP_STD = 0.02


def initialize_model(model: nn.Module, n_layers: int, generator: torch.Generator) -> None:
    """Initialise ``model`` in place following the D0 scheme.

    Parameters
    ----------
    model:
        A :class:`~expertforge.d0.model.transformer.D0Model` (or compatible)
        whose parameters match the D0 state-dict structure.
    n_layers:
        Layer count ``L`` used for the residual-scaling std
        ``0.02 / sqrt(2 * L)``.
    generator:
        Deterministic :class:`torch.Generator` driving every random draw. The
        generator controls device and RNG state.
    """
    residual_std = _EMBEDDING_QKV_SWIGLU_GATE_UP_STD / (2.0 * n_layers) ** 0.5

    for name, param in model.named_parameters():
        # RMSNorm weights are constant 1.0 and must not be perturbed.
        if name.endswith("norm.weight"):
            with torch.no_grad():
                param.fill_(1.0)
            continue

        if (
            name == "token_embedding.weight"
            or name.endswith("attention.q_proj.weight")
            or name.endswith("attention.k_proj.weight")
            or name.endswith("attention.v_proj.weight")
            or name.endswith("ffn.gate_proj.weight")
            or name.endswith("ffn.up_proj.weight")
        ):
            std = _EMBEDDING_QKV_SWIGLU_GATE_UP_STD
        elif name.endswith("attention.out_proj.weight") or name.endswith("ffn.down_proj.weight"):
            std = residual_std
        else:  # pragma: no cover - defensive; inventory is closed.
            raise ValueError(f"D0 initialisation has no rule for parameter {name!r}.")

        with torch.no_grad():
            param.normal_(mean=0.0, std=std, generator=generator)
