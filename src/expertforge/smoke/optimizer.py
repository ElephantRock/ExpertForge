"""Minimal AdamW optimizer for the smoke gate (Issue #14).

Holds per-parameter first-moment (``exp_avg``) and second-moment (``exp_avg_sq``)
slots plus a per-parameter scalar ``step``. This exercises the **full multi-slot
optimizer state** path the checkpoint encoder supports (amendment E): every slot
round-trips through ``optimizer_slots`` and the scalar steps through
``optimizer_scalar_state``.

After :meth:`update`, the optimizer is quiescent (``accumulation_position == 0``),
satisfying the V1 save boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["AdamWState", "SmokeAdamW", "OPTIMIZER_TYPE"]

OPTIMIZER_TYPE = "adamw"


@dataclass
class AdamWState:
    """The complete, serializable AdamW state.

    ``exp_avg`` / ``exp_avg_sq`` map parameter name -> float32 array (same shape
    as the parameter). ``steps`` maps parameter name -> int step count. The scalar
    step state is serialized separately so the checkpoint round-trips both the
    tensor slots and the structured scalar state.
    """

    exp_avg: dict[str, np.ndarray] = field(default_factory=dict)
    exp_avg_sq: dict[str, np.ndarray] = field(default_factory=dict)
    steps: dict[str, int] = field(default_factory=dict)

    def clone(self) -> AdamWState:
        return AdamWState(
            exp_avg={k: v.copy() for k, v in self.exp_avg.items()},
            exp_avg_sq={k: v.copy() for k, v in self.exp_avg_sq.items()},
            steps=dict(self.steps),
        )


class SmokeAdamW:
    """AdamW with bias correction, decoupled weight decay, and float32 state."""

    def __init__(
        self,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.state = AdamWState()

    def initialize(self, parameters: dict[str, np.ndarray]) -> None:
        """Allocate zeroed first/second-moment slots for every parameter."""
        self.state = AdamWState(
            exp_avg={k: np.zeros_like(v, dtype=np.float32) for k, v in parameters.items()},
            exp_avg_sq={k: np.zeros_like(v, dtype=np.float32) for k, v in parameters.items()},
            steps={k: 0 for k in parameters},
        )

    def update(self, parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray]) -> None:
        """Apply one AdamW update in place to ``parameters``.

        Exactly one update == one optimizer step == ``global_update`` +1 and
        ``accumulation_position`` returns to 0. State arrays stay float32 and
        C-contiguous.
        """
        for name in parameters:
            g = gradients[name]
            m = self.state.exp_avg[name]
            v = self.state.exp_avg_sq[name]
            step = self.state.steps[name] + 1
            m[...] = self.beta1 * m + (1.0 - self.beta1) * g
            v[...] = self.beta2 * v + (1.0 - self.beta2) * (g * g)
            m_hat = m / (1.0 - self.beta1**step)
            v_hat = v / (1.0 - self.beta2**step)
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay != 0.0:
                update = update + self.weight_decay * parameters[name]
            parameters[name][...] = parameters[name] - self.lr * update
            self.state.steps[name] = step
            # Keep state C-contiguous float32 after in-place assignment.
            self.state.exp_avg[name] = np.ascontiguousarray(m, dtype=np.float32)
            self.state.exp_avg_sq[name] = np.ascontiguousarray(v, dtype=np.float32)

    # -- state introspection for checkpoint capture ----------------------

    def slot_arrays(self) -> dict[str, np.ndarray]:
        """All per-parameter slot arrays keyed by ``optimizer.<slot>.<param>``.

        The logical name MUST be ``optimizer.<slot_name>.<param>`` so the
        checkpoint encoder binds each slot tensor to the optimizer descriptor's
        ``(group, param, slot)`` tuple.
        """
        out: dict[str, np.ndarray] = {}
        for name in self.state.exp_avg:
            out[f"optimizer.exp_avg.{name}"] = self.state.exp_avg[name]
            out[f"optimizer.exp_avg_sq.{name}"] = self.state.exp_avg_sq[name]
        return out

    def scalar_state(self) -> dict[str, Any]:
        """Structured scalar optimizer state: per-parameter step counts."""
        return {"steps": dict(self.state.steps), "optimizer_type": OPTIMIZER_TYPE}

    def apply_state(
        self,
        slot_tensors: dict[str, np.ndarray],
        scalar_state: dict[str, Any],
    ) -> None:
        """Restore optimizer state from decoded slot tensors + scalar state."""
        steps = dict(scalar_state.get("steps", {}))
        exp_avg: dict[str, np.ndarray] = {}
        exp_avg_sq: dict[str, np.ndarray] = {}
        for key, arr in slot_tensors.items():
            # key is optimizer.<slot>.<param>
            _opt, slot, param = key.split(".", 2)
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            if slot == "exp_avg":
                exp_avg[param] = arr
            elif slot == "exp_avg_sq":
                exp_avg_sq[param] = arr
            else:
                raise ValueError(f"unknown optimizer slot {slot!r}")
        self.state = AdamWState(exp_avg=exp_avg, exp_avg_sq=exp_avg_sq, steps=steps)
