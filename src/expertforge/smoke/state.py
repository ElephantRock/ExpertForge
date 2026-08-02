"""Smoke checkpoint state provider and restore factory (Issue #14).

Implements the smoke-specific :class:`StateProvider` /
:class:`StateFactory` boundary (amendment E). This is deliberately NOT
:class:`~expertforge.checkpoints.reference_adapter.ReferenceStateProvider`,
which owns hard-coded fixture descriptors. The smoke provider builds truthful
descriptors from the actual parameter/optimizer/scheduler/data arrays.

Capture (:meth:`SmokeStateProvider.capture_checkpoint_snapshot`) returns ONE
quiescent :class:`CapturedCheckpointState` containing:

- all model parameters/buffers as canonical little-endian contiguous float32
  bytes;
- complete AdamW state (every first/second-moment slot) via ``optimizer_slots``
  plus per-parameter step counts via ``optimizer_scalar_state``;
- the scheduler tensor (current LR) plus ``scheduler_scalar_state`` (step count);
- the current :class:`RngStateBundle` bytes;
- :class:`DataCursor` and :class:`CounterSnapshot` with truthful descriptors.

Restore (:class:`SmokeStateFactory`) constructs isolated fresh objects, applies
all decoded state, and commits to live via the two-phase transaction with RNG
restored LAST (the supported holder pattern). Direct assignment from decoded
members is prohibited — restoration flows through
:class:`~expertforge.checkpoints.restore.RestoreTransaction`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from expertforge.checkpoints.models import (
    CapturedCheckpointState,
    CapturedTensor,
    CompatibilityDescriptor,
    CounterSnapshot,
    DataCursor,
    DataDescriptor,
    ModelDescriptor,
    ModelParameterDescriptor,
    OptimizerDescriptor,
    OptimizerParamGroup,
    OptimizerStateSlot,
    RngDescriptor,
    SchedulerDescriptor,
    TopologyDescriptor,
)
from expertforge.rng.state import RngStateBundle

__all__ = [
    "SmokeRuntime",
    "SmokeStateProvider",
    "SmokeStateFactory",
    "build_expected_descriptor",
    "safe_value_to_native",
]


@dataclass
class SmokeRuntime:
    """The live, mutable smoke training state owned by one attempt.

    Held by the runner; captured by :class:`SmokeStateProvider` and reconstructed
    by :class:`SmokeStateFactory`. The factory's fresh target is a copy of this
    structure (without the live RNG, which is restored separately).
    """

    model_params: dict[str, np.ndarray]
    model_buffers: dict[str, np.ndarray]
    optimizer_state: Any  # SmokeAdamW
    scheduler: Any  # SmokeScheduler
    cursor: Any  # SmokeDataCursor
    counters: CounterSnapshot
    rng_manager: Any  # RngManager


def _captured_tensor(name: str, array: np.ndarray) -> CapturedTensor:
    """Build a CapturedTensor from a float32 C-contiguous array."""
    arr = np.ascontiguousarray(array, dtype=np.float32)
    return CapturedTensor(
        logical_name=name,
        dtype="float32",
        shape=tuple(int(d) for d in arr.shape),
        raw_bytes=arr.tobytes(),
    )


class SmokeStateProvider:
    """Authoritative save boundary for the smoke runtime: one quiescent snapshot."""

    def __init__(self, runtime: SmokeRuntime) -> None:
        self._runtime = runtime

    def capture_checkpoint_snapshot(self) -> CapturedCheckpointState:
        rt = self._runtime
        if rt.counters.accumulation_position != 0:
            raise ValueError("smoke capture requires accumulation_position == 0")

        # Parameters (sorted by name for canonical order).
        param_names = sorted(rt.model_params)
        parameters = tuple(_captured_tensor(name, rt.model_params[name]) for name in param_names)
        buffers = tuple(
            _captured_tensor(name, rt.model_buffers[name]) for name in sorted(rt.model_buffers)
        )

        # Optimizer slots: exp_avg / exp_avg_sq per parameter.
        slot_arrays = rt.optimizer_state.slot_arrays()
        optimizer_slots = tuple(
            _captured_tensor(name, arr) for name, arr in sorted(slot_arrays.items())
        )
        optimizer_scalar_state = _native_to_safe_value(rt.optimizer_state.scalar_state())

        # Scheduler tensor + scalar state.
        scheduler_tensor = _captured_tensor("scheduler.state", rt.scheduler.state_tensor())
        scheduler_scalar_state = _native_to_safe_value(rt.scheduler.scalar_state())

        # RNG bundle bytes.
        rng_bundle = rt.rng_manager.capture_state()
        rng_bundle_bytes = rng_bundle.to_deterministic_json()

        # Descriptors derived from the actual arrays.
        model_descriptor = _build_model_descriptor(rt)
        optimizer_descriptor = _build_optimizer_descriptor(rt, param_names)
        scheduler_descriptor = _build_scheduler_descriptor(rt)
        rng_descriptor = _build_rng_descriptor(rng_bundle)
        data_cursor, data_identity = rt.cursor.snapshot()
        data_descriptor = DataDescriptor(
            identity=data_identity,
            sampler_type=data_cursor.sampler_type,
            sampler_version=data_cursor.sampler_version,
            batch_size=data_cursor.batch_size,
            sequence_length=data_cursor.sequence_length,
            drop_last=data_cursor.drop_last,
        )
        topology_descriptor = TopologyDescriptor(world_size=1, rank_assignment=(0,))

        return CapturedCheckpointState(
            parameters=parameters,
            buffers=buffers,
            alias_groups=(),
            optimizer=None,
            scheduler=scheduler_tensor,
            scaler=None,
            optimizer_slots=optimizer_slots,
            optimizer_scalar_state=optimizer_scalar_state,
            scheduler_scalar_state=scheduler_scalar_state,
            rng_bundle_bytes=rng_bundle_bytes,
            data_cursor=data_cursor,
            counters=rt.counters,
            model_descriptor=model_descriptor,
            optimizer_descriptor=optimizer_descriptor,
            scheduler_descriptor=scheduler_descriptor,
            scaler_descriptor=None,
            rng_descriptor=rng_descriptor,
            data_descriptor=data_descriptor,
            topology_descriptor=topology_descriptor,
            optimizer_update_complete=True,
            accumulation_position=0,
            async_prefetch_active=False,
        )


def build_expected_descriptor(
    runtime: SmokeRuntime, *, specification_fingerprint: str
) -> CompatibilityDescriptor:
    """Build the compatibility descriptor the restore transaction gates against.

    Must exactly match the captured snapshot's compatibility descriptor so
    :meth:`RestoreTransaction.prepare` accepts the archive as ``exact``. The
    ``specification_fingerprint`` is the resuming attempt's fingerprint (R1's),
    which equals R0's because RESUME preserves the specification fingerprint.
    """
    rt = runtime
    provider = SmokeStateProvider(rt)
    # Build the descriptor from a fresh capture without persisting it.
    captured = provider.capture_checkpoint_snapshot()
    model_descriptor = captured.model_descriptor
    optimizer_descriptor = captured.optimizer_descriptor
    scheduler_descriptor = captured.scheduler_descriptor
    rng_descriptor = captured.rng_descriptor
    data_descriptor = captured.data_descriptor
    topology_descriptor = captured.topology_descriptor
    return CompatibilityDescriptor(
        specification_fingerprint=specification_fingerprint,
        model_descriptor=model_descriptor,
        optimizer_descriptor=optimizer_descriptor,
        scheduler_descriptor=scheduler_descriptor,
        scaler_descriptor=None,
        rng_descriptor=rng_descriptor,
        data_descriptor=data_descriptor,
        topology_descriptor=topology_descriptor,
    )


class SmokeStateFactory:
    """Constructs isolated fresh smoke objects and applies restored state.

    Implements the full :class:`StateFactory` protocol including
    :meth:`commit_to_live` / :meth:`revert_commit_to_live` so the two-phase
    restore transaction (RNG last) swaps handles atomically.
    """

    def __init__(
        self,
        live_runtime: SmokeRuntime,
        rng_factory: Any,
    ) -> None:
        """``live_runtime`` is the pre-restore live runtime (authoritative before
        commit). ``rng_factory`` builds a fresh, uninitialized
        :class:`RngManager` configured identically to the live one (used to
        ``restore_state`` the bundle during commit)."""
        self._live = live_runtime
        self._rng_factory = rng_factory
        self._target: SmokeRuntime | None = None
        self._previous_handles: SmokeRuntime | None = None

    def create(self) -> SmokeRuntime:
        """Construct a fresh isolated runtime mirroring the live configuration."""
        rt = self._live
        fresh_params = {k: np.zeros_like(v) for k, v in rt.model_params.items()}
        fresh_buffers = {k: np.zeros_like(v) for k, v in rt.model_buffers.items()}
        # Fresh optimizer/scheduler initialized to the same shapes/types.
        fresh_optimizer = _clone_uninitialized_optimizer(rt.optimizer_state)
        fresh_scheduler = _clone_uninitialized_scheduler(rt.scheduler)
        fresh_cursor = _clone_uninitialized_cursor(rt.cursor)
        fresh_counters = CounterSnapshot(
            global_update=0,
            completed_microsteps=0,
            accumulation_position=0,
            accepted_samples=0,
            accepted_sequences=0,
            processed_tokens=0,
        )
        # The live RNG manager is restored separately by the transaction; the
        # fresh target carries a placeholder equal to the live manager until the
        # transaction swaps the restored one in.
        self._target = SmokeRuntime(
            model_params=fresh_params,
            model_buffers=fresh_buffers,
            optimizer_state=fresh_optimizer,
            scheduler=fresh_scheduler,
            cursor=fresh_cursor,
            counters=fresh_counters,
            rng_manager=rt.rng_manager,
        )
        return self._target

    # -- apply methods (operate on the fresh target) ---------------------

    def apply_parameters(self, target: SmokeRuntime, tensors: dict[str, CapturedTensor]) -> None:
        for name, t in tensors.items():
            arr = _tensor_to_f32_array(t)
            target.model_params[name] = np.ascontiguousarray(arr, dtype=np.float32)

    def apply_buffers(self, target: SmokeRuntime, tensors: dict[str, CapturedTensor]) -> None:
        for name, t in tensors.items():
            arr = _tensor_to_f32_array(t)
            target.model_buffers[name] = np.ascontiguousarray(arr, dtype=np.float32)

    def apply_optimizer(self, target: SmokeRuntime, tensor: CapturedTensor | None) -> None:
        # The smoke optimizer uses structured multi-slot state; this legacy
        # single-tensor path is not used. Provided for protocol completeness.
        if tensor is not None:
            raise ValueError("smoke optimizer does not use a legacy single tensor")

    def apply_optimizer_state(
        self,
        target: SmokeRuntime,
        state_tensor: CapturedTensor | None,
        slots: dict[tuple[int, str, str], CapturedTensor],
        scalar_state: Any,
    ) -> None:
        """Apply the full AdamW multi-slot state + scalar step counts.

        ``slots`` is keyed by ``(group_index, param_name, slot_name)`` per the
        checkpoint descriptor. Each captured tensor is materialized and routed to
        the optimizer under its ``optimizer.<slot>.<param>`` logical name.
        """
        slot_arrays: dict[str, np.ndarray] = {}
        for (group_index, param_name, slot_name), captured in slots.items():
            del group_index  # single group in the smoke gate
            arr = _tensor_to_f32_array(captured)
            slot_arrays[f"optimizer.{slot_name}.{param_name}"] = arr
        scalar_native = safe_value_to_native(scalar_state) if scalar_state is not None else {}
        target.optimizer_state.apply_state(slot_arrays, scalar_native)

    def apply_scheduler(self, target: SmokeRuntime, tensor: CapturedTensor | None) -> None:
        if tensor is not None:
            arr = _tensor_to_f32_array(tensor)
            target.scheduler.lr = float(arr.reshape(-1)[0])

    def apply_scheduler_state(self, target: SmokeRuntime, bundle: Any) -> None:
        """Apply the scheduler tensor + scalar state (full-state path).

        The scalar_state carries the authoritative full-precision ``lr`` and
        ``step_count``; the one-element tensor is a derived float32 representation.
        Prefer the scalar_state's exact float64 ``lr`` so a restored scheduler
        matches an uninterrupted one byte-for-byte (U0's LR comes straight from
        the float64 config value).
        """
        tensor = getattr(bundle, "tensor", None)
        scalar_state = getattr(bundle, "scalar_state", None)
        scalar_native = safe_value_to_native(scalar_state) if scalar_state is not None else {}
        if "lr" in scalar_native:
            target.scheduler.lr = float(scalar_native["lr"])
        elif tensor is not None:
            target.scheduler.lr = float(_tensor_to_f32_array(tensor).reshape(-1)[0])
        target.scheduler.step_count = int(
            scalar_native.get("step_count", target.scheduler.step_count)
        )

    def apply_scaler(self, target: SmokeRuntime, tensor: CapturedTensor | None) -> None:
        # No scaler in the smoke gate.
        if tensor is not None:
            raise ValueError("smoke gate has no scaler")

    def apply_data_cursor(self, target: SmokeRuntime, cursor: DataCursor) -> None:
        target.cursor = _clone_cursor_from_restored(target.cursor, cursor)

    def apply_counters(self, target: SmokeRuntime, counters: CounterSnapshot) -> None:
        target.counters = CounterSnapshot(
            global_update=int(counters.global_update),
            completed_microsteps=int(counters.completed_microsteps),
            accumulation_position=int(counters.accumulation_position),
            accepted_samples=int(counters.accepted_samples),
            accepted_sequences=int(counters.accepted_sequences),
            processed_tokens=int(counters.processed_tokens),
        )

    # -- commit / revert (two-phase, RNG last) ---------------------------

    def commit_to_live(self, target: SmokeRuntime) -> SmokeRuntime:
        """Swap the fresh target's non-RNG handles into the live runtime.

        Captures the previous handles first so :meth:`revert_commit_to_live` can
        restore them if the subsequent RNG restore fails.
        """
        live = self._live
        self._previous_handles = SmokeRuntime(
            model_params=dict(live.model_params),
            model_buffers=dict(live.model_buffers),
            optimizer_state=live.optimizer_state,
            scheduler=live.scheduler,
            cursor=live.cursor,
            counters=live.counters,
            rng_manager=live.rng_manager,
        )
        live.model_params = target.model_params
        live.model_buffers = target.model_buffers
        live.optimizer_state = target.optimizer_state
        live.scheduler = target.scheduler
        live.cursor = target.cursor
        live.counters = target.counters
        # The RNG manager handle is swapped by the transaction's RNG-last step
        # (R1 swaps in a freshly restore_state'd manager). Keep target's
        # placeholder manager for now.
        live.rng_manager = target.rng_manager
        return live

    def revert_commit_to_live(self, target: SmokeRuntime) -> None:
        """Restore the pre-commit handles when RNG restoration fails."""
        if self._previous_handles is None:
            return
        prev = self._previous_handles
        live = self._live
        live.model_params = prev.model_params
        live.model_buffers = prev.model_buffers
        live.optimizer_state = prev.optimizer_state
        live.scheduler = prev.scheduler
        live.cursor = prev.cursor
        live.counters = prev.counters
        live.rng_manager = prev.rng_manager
        self._previous_handles = None


# ---------------------------------------------------------------------------
# Descriptor builders (truthful, derived from actual arrays)
# ---------------------------------------------------------------------------


def _build_model_descriptor(rt: SmokeRuntime) -> ModelDescriptor:
    params = tuple(
        ModelParameterDescriptor(
            name=name, shape=tuple(rt.model_params[name].shape), dtype="float32"
        )
        for name in sorted(rt.model_params)
    )
    buffers = tuple(
        ModelParameterDescriptor(
            name=name, shape=tuple(rt.model_buffers[name].shape), dtype="float32"
        )
        for name in sorted(rt.model_buffers)
    )
    return ModelDescriptor(parameters=params, buffers=buffers, alias_groups=())


def _build_optimizer_descriptor(rt: SmokeRuntime, param_names: list[str]) -> OptimizerDescriptor:
    state_slots: list[OptimizerStateSlot] = []
    for name in param_names:
        shape = tuple(rt.model_params[name].shape)
        state_slots.append(
            OptimizerStateSlot(
                group_index=0,
                param_name=name,
                slot_name="exp_avg",
                shape=shape,
                dtype="float32",
            )
        )
        state_slots.append(
            OptimizerStateSlot(
                group_index=0,
                param_name=name,
                slot_name="exp_avg_sq",
                shape=shape,
                dtype="float32",
            )
        )
    return OptimizerDescriptor(
        optimizer_type="adamw",
        param_groups=(
            OptimizerParamGroup(group_index=0, param_names=tuple(param_names), options=()),
        ),
        state_slots=tuple(state_slots),
    )


def _build_scheduler_descriptor(rt: SmokeRuntime) -> SchedulerDescriptor:
    return SchedulerDescriptor(
        scheduler_type="constant_with_step",
        state_shape=(1,),
    )


def _build_rng_descriptor(rng_bundle: RngStateBundle) -> RngDescriptor:
    providers = tuple(sorted({s.provider for s in rng_bundle.framework_states}))
    return RngDescriptor(
        adapter_set=providers,
        framework_versions=(),
        rng_state_schema_version=rng_bundle.rng_state_schema_version,
    )


# ---------------------------------------------------------------------------
# SafeValue encode/decode helpers (optimizer/scheduler scalar state)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SafeValue encode/decode — delegated to the public checkpoint API (finding F5).
# The smoke package must NOT reach into private ``_Safe*`` models; it uses the
# exported ``safe_value_from_native`` (native -> SafeValue model) and
# ``safe_value_to_native`` (SafeValue model -> native) pair.
# ---------------------------------------------------------------------------


def _native_to_safe_value(value: Any) -> Any:
    """Encode native Python scalars/mappings to a SafeValue model (public API)."""
    from expertforge.checkpoints.encoder import safe_value_from_native

    return safe_value_from_native(value)


def safe_value_to_native(value: Any) -> Any:
    """Decode a SafeValue model back to native Python (public API).

    Used by the factory to consume optimizer/scheduler scalar state restored
    from a checkpoint.
    """
    from expertforge.checkpoints.encoder import safe_value_to_native as _decode

    return _decode(value)


# ---------------------------------------------------------------------------
# Cloning helpers
# ---------------------------------------------------------------------------


def _tensor_to_f32_array(tensor: CapturedTensor) -> np.ndarray:
    arr = np.frombuffer(tensor.raw_bytes, dtype=np.dtype(tensor.dtype)).reshape(tensor.shape)
    return np.ascontiguousarray(arr, dtype=np.float32).copy()


def _clone_uninitialized_optimizer(optimizer: Any) -> Any:
    fresh = type(optimizer)(
        lr=optimizer.lr,
        betas=(optimizer.beta1, optimizer.beta2),
        eps=optimizer.eps,
        weight_decay=optimizer.weight_decay,
    )
    fresh.initialize(dict(optimizer.state.exp_avg))  # shapes from live slots
    # Zero out freshly-initialized state so the target is truly isolated.
    fresh.state = type(optimizer.state)(
        exp_avg={k: np.zeros_like(v) for k, v in fresh.state.exp_avg.items()},
        exp_avg_sq={k: np.zeros_like(v) for k, v in fresh.state.exp_avg_sq.items()},
        steps={k: 0 for k in fresh.state.steps},
    )
    return fresh


def _clone_uninitialized_scheduler(scheduler: Any) -> Any:
    fresh = type(scheduler)(lr=scheduler.lr)
    fresh.initialize()
    return fresh


def _clone_uninitialized_cursor(cursor: Any) -> Any:
    from expertforge.smoke.data import SmokeDataCursor

    return SmokeDataCursor(
        tokens=cursor.tokens.copy(),
        batch_size=cursor.batch_size,
        sequence_length=cursor.sequence_length,
        drop_last=cursor.drop_last,
    )


def _clone_cursor_from_restored(template: Any, restored: DataCursor) -> Any:
    from expertforge.smoke.data import SmokeDataCursor

    return SmokeDataCursor.restore_from(
        restored,
        batch_size=template.batch_size,
        sequence_length=template.sequence_length,
        drop_last=template.drop_last,
    )
