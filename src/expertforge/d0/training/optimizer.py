"""Frozen D0 optimizer: AdamW over weight-class parameter groups.

Implements the ratified optimizer contract on the D0.2 torch model:

- **AdamW, decoupled**, bias correction on (PyTorch ``torch.optim.AdamW``
  semantics — there is no bias-correction toggle), ``foreach=False`` so the
  elementwise path is the only code path;
- **weight decay by weight class**: attention and SwiGLU matrix weights
  decay; the token embedding (which is also the tied output head by object
  identity) and all RMSNorm weights do not;
- **gradient clipping**: global L2 norm, applied to the accumulated
  gradients after the last microstep of an update and before the optimizer
  step;
- **state surface** matching the checkpoint contract: per-parameter
  ``exp_avg`` / ``exp_avg_sq`` slots keyed ``optimizer.<slot>.<param>``,
  per-parameter integer steps plus the optimizer identity in the scalar
  state, and an :class:`~expertforge.checkpoints.models.OptimizerDescriptor`
  describing groups/options/slots for exact-compatibility restore checks.

Master parameters are float32 (the frozen ``master_parameter_dtype``); the
optimizer consumes and stores float32 gradients and float32 moment slots
(``optimizer_state_dtype``). The bf16 compute path is the caller's concern
(see :mod:`expertforge.d0.training.precision`).

Deferred to the runner tranche (documented contract flags): stream-edge /
drop-last semantics and the non-data RNG substreams.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from expertforge.config.d0_models import D0OptimizerConfig
from expertforge.d0.errors import NonFiniteGradientError, OptimizerStateError
from expertforge.d0.training._torch import require_torch as _require_torch

# Resolve :mod:`torch` through the lazy helper first so a missing optional
# dependency surfaces as a typed, actionable error (d0-model extra).
_require_torch()

import torch  # noqa: E402
from torch import nn  # noqa: E402

if TYPE_CHECKING:
    from expertforge.checkpoints.models import OptimizerDescriptor

__all__ = [
    "OPTIMIZER_TYPE",
    "SLOT_NAMES",
    "DECAY_GROUP_INDEX",
    "NO_DECAY_GROUP_INDEX",
    "build_param_groups",
    "D0Optimizer",
]

OPTIMIZER_TYPE = "AdamW"
SLOT_NAMES = ("exp_avg", "exp_avg_sq")
DECAY_GROUP_INDEX = 0
NO_DECAY_GROUP_INDEX = 1

# The frozen weight classes and the parameter-name predicates they select.
# The D0.2 parameter inventory is closed: every parameter name must match
# exactly one class, and every class named in the config must be one of
# these four. Unknown config classes or unknown parameter names fail closed.

_DECAY_CLASSES = ("attention_matrix_weights", "swiglu_matrix_weights")
_NO_DECAY_CLASSES = ("token_embedding_weight", "rmsnorm_weights")

_ATTENTION_PROJECTIONS = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight")
_SWIGLU_PROJECTIONS = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")


def _classify(name: str) -> int:
    """Return the group index for one parameter name, failing on unknowns."""

    if name == "token_embedding.weight":
        return NO_DECAY_GROUP_INDEX
    if name.endswith("norm.weight"):
        return NO_DECAY_GROUP_INDEX
    if name.endswith(_ATTENTION_PROJECTIONS) or name.endswith(_SWIGLU_PROJECTIONS):
        return DECAY_GROUP_INDEX
    raise OptimizerStateError(
        f"parameter {name!r} matches no frozen D0 weight class "
        f"(decay={_DECAY_CLASSES}, no-decay={_NO_DECAY_CLASSES})"
    )


def build_param_groups(
    model: nn.Module, config: D0OptimizerConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition model parameters into the two frozen weight-decay groups.

    Group 0 holds the decay classes (attention + SwiGLU matrix weights) with
    the configured weight decay; group 1 holds the no-decay classes (token
    embedding — which is the tied output head — and RMSNorm weights) with
    weight decay 0. Parameter names are sorted within each group so the
    partition is deterministic.

    Unknown weight-class names in the config, or model parameters outside
    the frozen D0 weight classes, fail closed.
    """

    for class_name in (*config.weight_decay_includes, *config.weight_decay_excludes):
        if class_name not in _DECAY_CLASSES and class_name not in _NO_DECAY_CLASSES:
            raise OptimizerStateError(f"unknown weight class {class_name!r} in optimizer config")
        if (
            class_name in config.weight_decay_includes
            and class_name in config.weight_decay_excludes
        ):
            raise OptimizerStateError(
                f"weight class {class_name!r} appears in both include and exclude lists"
            )
    for expected in (*_DECAY_CLASSES, *_NO_DECAY_CLASSES):
        in_config = (
            expected in config.weight_decay_includes or expected in config.weight_decay_excludes
        )
        if not in_config:
            raise OptimizerStateError(
                f"frozen weight class {expected!r} is missing from the optimizer config"
            )

    decay_names: list[str] = []
    no_decay_names: list[str] = []
    params_by_name: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        params_by_name[name] = parameter
        if _classify(name) == DECAY_GROUP_INDEX:
            decay_names.append(name)
        else:
            no_decay_names.append(name)
    if not decay_names and not no_decay_names:
        raise OptimizerStateError("model exposes no parameters")

    return (
        {
            "params": [params_by_name[n] for n in sorted(decay_names)],
            "weight_decay": float(config.weight_decay),
        },
        {
            "params": [params_by_name[n] for n in sorted(no_decay_names)],
            "weight_decay": 0.0,
        },
    )


class D0Optimizer:
    """The frozen AdamW binding over a D0 model.

    The learning rate is supplied per update by the frozen schedule (see
    :mod:`expertforge.d0.training.schedule`), never held here.
    """

    def __init__(self, model: nn.Module, config: D0OptimizerConfig) -> None:
        self._model = model
        self._config = config
        self._group_dicts = build_param_groups(model, config)
        # lr is a placeholder; the schedule assigns it per update before step.
        self._impl = torch.optim.AdamW(
            self._group_dicts,
            lr=1.0,
            betas=(float(config.beta1), float(config.beta2)),
            eps=float(config.epsilon),
            weight_decay=float(config.weight_decay),
            foreach=False,
        )
        self._param_by_name: dict[str, torch.Tensor] = {
            name: parameter for name, parameter in model.named_parameters()
        }
        self._last_clip_global_norm: float | None = None

    # -- stepping ---------------------------------------------------------

    def zero_grad(self) -> None:
        """Zero all parameter gradients (``set_to_none=True``)."""

        self._impl.zero_grad(set_to_none=True)

    def clip_and_step(self, learning_rate: float) -> float:
        """Clip accumulated gradients and apply one optimizer update.

        Returns the pre-clip global L2 gradient norm (for telemetry). The
        clip point is after gradient accumulation and before the step, per
        the frozen contract. Non-finite gradients fail closed before any
        parameter is touched.
        """

        if not learning_rate > 0.0:
            raise OptimizerStateError(
                f"learning rate must be positive for an applied update; got {learning_rate!r}"
            )
        gradients: list[torch.Tensor] = [
            p.grad
            for group in self._impl.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        expected = sum(len(group["params"]) for group in self._impl.param_groups)
        if len(gradients) != expected:
            raise OptimizerStateError(
                f"{expected - len(gradients)} parameter(s) have no gradient at step time"
            )
        for gradient in gradients:
            if not bool(torch.isfinite(gradient).all()):
                raise NonFiniteGradientError(
                    "non-finite gradient observed; refusing to step "
                    "(diagnostic: non_finite_gradient)"
                )
        total_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [p for group in self._impl.param_groups for p in group["params"]],
                max_norm=float(self._config.gradient_clip_global_l2_norm),
            )
        )
        for group in self._impl.param_groups:
            group["lr"] = float(learning_rate)
        self._impl.step()
        self._last_clip_global_norm = total_norm
        return total_norm

    @property
    def last_clip_global_norm(self) -> float | None:
        return self._last_clip_global_norm

    # -- checkpoint state surface -----------------------------------------

    def slot_tensors(self) -> dict[str, torch.Tensor]:
        """Moment slots keyed ``optimizer.<slot>.<param>`` (float32, CPU).

        The logical name convention is fixed by the checkpoint encoder so
        each tensor binds to the descriptor's ``(group, param, slot)``.
        """

        out: dict[str, torch.Tensor] = {}
        for name in sorted(self._param_by_name):
            state = self._impl.state[self._param_by_name[name]]
            for slot in SLOT_NAMES:
                tensor = state[slot].detach().to(device="cpu", dtype=torch.float32)
                out[f"optimizer.{slot}.{name}"] = tensor.contiguous()
        return out

    def scalar_state(self) -> dict[str, Any]:
        """Per-parameter step counts plus the optimizer identity."""

        steps = {
            name: int(self._impl.state[self._param_by_name[name]]["step"].item())
            for name in sorted(self._param_by_name)
        }
        return {
            "steps": steps,
            "optimizer_type": OPTIMIZER_TYPE,
            # Bias correction is inherent to torch AdamW (contract flag A1);
            # recorded so a restore verifies the same semantics.
            "bias_correction": True,
        }

    def apply_state(
        self,
        slot_tensors: dict[str, torch.Tensor],
        scalar_state: dict[str, Any],
    ) -> None:
        """Restore moment slots and step counts from captured state.

        The captured slot set must cover exactly the two slots of every
        parameter; extra or missing entries fail closed.
        """

        if scalar_state.get("optimizer_type") != OPTIMIZER_TYPE:
            raise OptimizerStateError(
                f"scalar state optimizer_type {scalar_state.get('optimizer_type')!r} "
                f"!= {OPTIMIZER_TYPE!r}"
            )
        if scalar_state.get("bias_correction") is not True:
            raise OptimizerStateError("restored optimizer state lacks bias-correction identity")
        steps: dict[str, int] = dict(scalar_state.get("steps", {}))
        expected_keys = {
            f"optimizer.{slot}.{name}" for name in self._param_by_name for slot in SLOT_NAMES
        }
        observed_keys = set(slot_tensors)
        if observed_keys != expected_keys:
            missing = sorted(expected_keys - observed_keys)
            extra = sorted(observed_keys - expected_keys)
            raise OptimizerStateError(
                f"captured optimizer slot set mismatch; missing={missing[:4]}... "
                f"extra={extra[:4]}..."
            )
        by_param: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in slot_tensors.items():
            _optimizer, slot, name = key.split(".", 2)
            by_param.setdefault(name, {})[slot] = tensor
        for name, parameter in self._param_by_name.items():
            slots = by_param[name]
            step_value = steps.get(name)
            if type(step_value) is not int or step_value < 0:
                raise OptimizerStateError(f"missing or invalid step count for {name!r}")
            self._impl.state[parameter] = {
                "step": torch.tensor(float(step_value), dtype=torch.float32),
                "exp_avg": slots["exp_avg"]
                .to(device=parameter.device, dtype=torch.float32)
                .clone(),
                "exp_avg_sq": slots["exp_avg_sq"]
                .to(device=parameter.device, dtype=torch.float32)
                .clone(),
            }

    def descriptor(self) -> OptimizerDescriptor:
        """Build the checkpoint compatibility descriptor."""

        from expertforge.checkpoints.models import (
            OptimizerDescriptor,
            OptimizerParamGroup,
            OptimizerStateSlot,
        )

        group_names: list[list[str]] = [[], []]
        for name in self._param_by_name:
            group_names[_classify(name)].append(name)
        from expertforge.checkpoints.encoder import safe_value_from_native

        options = (
            ("bias_correction", safe_value_from_native(True)),
            ("foreach", safe_value_from_native(False)),
        )
        groups = (
            OptimizerParamGroup(
                group_index=DECAY_GROUP_INDEX,
                param_names=tuple(sorted(group_names[DECAY_GROUP_INDEX])),
                options=options,
            ),
            OptimizerParamGroup(
                group_index=NO_DECAY_GROUP_INDEX,
                param_names=tuple(sorted(group_names[NO_DECAY_GROUP_INDEX])),
                options=options,
            ),
        )
        slots = []
        for name in sorted(self._param_by_name):
            shape = tuple(self._param_by_name[name].shape)
            for slot in SLOT_NAMES:
                slots.append(
                    OptimizerStateSlot(
                        group_index=_classify(name),
                        param_name=name,
                        slot_name=slot,
                        shape=shape,
                        dtype="float32",
                    )
                )
        slots.sort(key=lambda s: (s.group_index, s.param_name, s.slot_name))
        return OptimizerDescriptor(
            optimizer_type=OPTIMIZER_TYPE,
            param_groups=groups,
            state_slots=tuple(slots),
        )
