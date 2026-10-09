"""Production state adapters for the D0 training path.

Formalizes into :mod:`src` what the tranche-1 round-trip test proved on a
tiny model: capturing a quiescent :class:`CapturedCheckpointState` from a
live D0 training state and restoring it through a
:class:`~expertforge.checkpoints.restore.RestoreTransaction` bit-exactly.

- :class:`D0TrainingState` — the live bundle (model, optimizer, schedule,
  last applied update).
- :class:`D0StateProvider` — capture: parameters as sorted float32
  little-endian tensors, ``optimizer.<slot>.<param>`` moment slots, the
  scheduler tensor (1-element float32 LR) plus its float64-authoritative
  scalar state, counters, and the compatibility descriptors (T1's
  optimizer descriptor; the data descriptor from the batch source).
- :class:`D0StateFactory` — the :class:`StateFactory` protocol
  implementation: builds a fresh state via a caller-supplied factory,
  applies restored tensors (parameters via byte copy, optimizer slots via
  :meth:`D0Optimizer.apply_state`, scheduler scalar via
  ``safe_value_to_native`` with the float64 LR authoritative — the smoke
  precedent), and swaps two-phase with rollback.

RNG is deliberately outside these adapters: the runner owns the RNG
bundle loader/consumer exactly as the smoke runner does, and the
transaction restores RNG last.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from expertforge.d0.training._torch import require_torch as _require_torch

# Resolve :mod:`torch` through the lazy helper first so a missing optional
# dependency surfaces as a typed, actionable error (d0-model extra).
_require_torch()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from expertforge.checkpoints.encoder import (  # noqa: E402
    safe_value_from_native,
    safe_value_to_native,
)
from expertforge.checkpoints.models import (  # noqa: E402
    CapturedCheckpointState,
    CapturedTensor,
    CompatibilityDescriptor,
    CounterSnapshot,
    DataCursor,
    DataDescriptor,
    DataIdentity,
    ModelDescriptor,
    ModelParameterDescriptor,
    RngDescriptor,
    SchedulerDescriptor,
    TopologyDescriptor,
)
from expertforge.d0.errors import D0TrainingError  # noqa: E402
from expertforge.d0.training.accumulation import UpdateAccumulator  # noqa: E402
from expertforge.d0.training.optimizer import D0Optimizer  # noqa: E402
from expertforge.d0.training.schedule import SCHEDULER_TYPE, D0Schedule  # noqa: E402

__all__ = [
    "D0TrainingState",
    "D0StateProvider",
    "D0StateFactory",
]

_TOPOLOGY = TopologyDescriptor(world_size=1, rank_assignment=(0,))


def _param_bytes(parameter: torch.Tensor) -> bytes:
    return parameter.detach().to("cpu").contiguous().numpy().astype("<f4", copy=False).tobytes()


def _captured_parameter(name: str, parameter: torch.Tensor) -> CapturedTensor:
    return CapturedTensor(
        logical_name=name,
        dtype="float32",
        shape=tuple(parameter.shape),
        raw_bytes=_param_bytes(parameter),
    )


@dataclass
class D0TrainingState:
    """The live D0 training bundle the runner advances."""

    model: nn.Module
    optimizer: D0Optimizer
    schedule: D0Schedule
    last_applied_update: int = 0
    # Sidecar values applied during restore (cursor + counters). The runner
    # adopts them from the factory target after commit.
    restored_cursor: DataCursor | None = None
    restored_counters: CounterSnapshot | None = None


class D0StateProvider:
    """Captures quiescent checkpoints from a live :class:`D0TrainingState`."""

    def __init__(
        self,
        *,
        state: D0TrainingState,
        accumulator: UpdateAccumulator,
        rng_bundle_bytes: Callable[[], bytes],
        data_identity: DataIdentity,
    ) -> None:
        self._state = state
        self._accumulator = accumulator
        self._rng_bundle_bytes = rng_bundle_bytes
        self._data_identity = data_identity

    # -- descriptors (shared by capture and restore compatibility) --------

    def _model_descriptor(self) -> ModelDescriptor:
        parameters = tuple(
            ModelParameterDescriptor(name=name, shape=tuple(parameter.shape), dtype="float32")
            for name, parameter in sorted(self._state.model.named_parameters())
        )
        return ModelDescriptor(parameters=parameters, buffers=())

    def _scheduler_descriptor(self) -> SchedulerDescriptor:
        return SchedulerDescriptor(scheduler_type=SCHEDULER_TYPE, state_shape=(1,))

    def rng_descriptor(self) -> RngDescriptor:
        return RngDescriptor(adapter_set=(), framework_versions=())

    def topology_descriptor(self) -> TopologyDescriptor:
        return _TOPOLOGY

    def expected_compatibility(
        self, *, specification_fingerprint: str, cursor: DataCursor
    ) -> CompatibilityDescriptor:
        """The compatibility descriptor a matching fresh state must satisfy.

        Descriptors depend only on configuration (parameter shapes, optimizer
        groups, scheduler identity, data identity) — identical between the
        live state and any fresh state built from the same config — so the
        runner uses the live provider to compute the restore expectation.
        """

        return CompatibilityDescriptor(
            specification_fingerprint=specification_fingerprint,
            model_descriptor=self._model_descriptor(),
            optimizer_descriptor=self._state.optimizer.descriptor(),
            scheduler_descriptor=self._scheduler_descriptor(),
            scaler_descriptor=None,
            rng_descriptor=self.rng_descriptor(),
            data_descriptor=self._data_descriptor(cursor),
            topology_descriptor=self.topology_descriptor(),
        )

    def _data_descriptor(self, cursor: DataCursor) -> DataDescriptor:
        return DataDescriptor(
            identity=self._data_identity,
            sampler_type=cursor.sampler_type,
            sampler_version=cursor.sampler_version,
            batch_size=cursor.batch_size,
            sequence_length=cursor.sequence_length,
            drop_last=cursor.drop_last,
        )

    def _expected_compatibility(
        self, *, specification_fingerprint: str, cursor: DataCursor
    ) -> CompatibilityDescriptor:
        return CompatibilityDescriptor(
            specification_fingerprint=specification_fingerprint,
            model_descriptor=self._model_descriptor(),
            optimizer_descriptor=self._state.optimizer.descriptor(),
            scheduler_descriptor=self._scheduler_descriptor(),
            scaler_descriptor=None,
            rng_descriptor=RngDescriptor(adapter_set=(), framework_versions=()),
            data_descriptor=self._data_descriptor(cursor),
            topology_descriptor=_TOPOLOGY,
        )

    # -- capture -----------------------------------------------------------

    def capture_checkpoint_snapshot(self, *, data_cursor: DataCursor) -> CapturedCheckpointState:
        """Capture one quiescent checkpoint snapshot.

        The accumulator must sit exactly on an update boundary
        (``accumulation_position == 0`` and ``global_update ==
        last_applied_update``) — the V1 save boundary.
        """

        state = self._state
        last = state.last_applied_update
        if self._accumulator.accumulation_position != 0:
            raise D0TrainingError(
                "checkpoint capture requires accumulation_position == 0; got "
                f"{self._accumulator.accumulation_position}"
            )
        if self._accumulator.global_update != last:
            raise D0TrainingError(
                f"accumulator global_update {self._accumulator.global_update} != "
                f"last_applied_update {last}"
            )
        parameters = tuple(
            _captured_parameter(name, parameter)
            for name, parameter in sorted(state.model.named_parameters())
        )
        scheduler_scalar = state.schedule.scalar_state(last_applied_update=last)
        scheduler_tensor = CapturedTensor(
            logical_name="scheduler.state",
            dtype="float32",
            shape=(1,),
            raw_bytes=np.asarray([scheduler_scalar.last_learning_rate], dtype="<f4").tobytes(),
        )
        return CapturedCheckpointState(
            parameters=parameters,
            buffers=(),
            alias_groups=(),
            optimizer=None,
            scheduler=scheduler_tensor,
            scaler=None,
            rng_bundle_bytes=self._rng_bundle_bytes(),
            optimizer_slots=tuple(
                CapturedTensor(
                    logical_name=name,
                    dtype="float32",
                    shape=tuple(tensor.shape),
                    raw_bytes=tensor.contiguous().numpy().astype("<f4", copy=False).tobytes(),
                )
                for name, tensor in sorted(state.optimizer.slot_tensors().items())
            ),
            optimizer_scalar_state=safe_value_from_native(state.optimizer.scalar_state()),
            scheduler_scalar_state=safe_value_from_native(
                {
                    "last_applied_update": scheduler_scalar.last_applied_update,
                    "last_learning_rate": scheduler_scalar.last_learning_rate,
                }
            ),
            data_cursor=data_cursor,
            counters=CounterSnapshot(
                global_update=last,
                completed_microsteps=last * self._accumulator.microsteps_per_update,
                accumulation_position=0,
                accepted_samples=data_cursor.accepted_samples,
                accepted_sequences=data_cursor.accepted_sequences,
                processed_tokens=self._accumulator.processed_tokens,
            ),
            model_descriptor=self._model_descriptor(),
            optimizer_descriptor=state.optimizer.descriptor(),
            scheduler_descriptor=self._scheduler_descriptor(),
            scaler_descriptor=None,
            rng_descriptor=RngDescriptor(adapter_set=(), framework_versions=()),
            data_descriptor=self._data_descriptor(data_cursor),
            topology_descriptor=_TOPOLOGY,
        )


class D0StateFactory:
    """:class:`StateFactory` implementation restoring into fresh D0 objects.

    ``state_factory`` builds a brand-new :class:`D0TrainingState` (fresh
    model, optimizer, schedule); the transaction then overwrites it with the
    checkpoint's exact state. After :meth:`commit_to_live`, the runner adopts
    ``factory.target``.
    """

    def __init__(self, state_factory: Callable[[], D0TrainingState]) -> None:
        self._state_factory = state_factory
        self.target: D0TrainingState | None = None
        self.committed = False
        self.reverted = False

    def create(self) -> D0TrainingState:
        self.target = self._state_factory()
        return self.target

    def apply_parameters(self, target: D0TrainingState, tensors: dict[str, CapturedTensor]) -> None:
        by_name = dict(target.model.named_parameters())
        for name, captured in tensors.items():
            parameter = by_name.get(name)
            if parameter is None:
                raise D0TrainingError(f"checkpoint parameter {name!r} not in fresh model")
            restored = torch.frombuffer(bytearray(captured.raw_bytes), dtype=torch.float32).reshape(
                tuple(captured.shape)
            )
            with torch.no_grad():
                parameter.copy_(restored)

    def apply_buffers(self, target: D0TrainingState, tensors: dict[str, CapturedTensor]) -> None:
        if tensors:
            raise D0TrainingError("D0 model captured unexpected buffers")

    def apply_optimizer(self, target: D0TrainingState, tensor: CapturedTensor | None) -> None:
        if tensor is not None:
            raise D0TrainingError("D0 uses slot-based optimizer state, not a legacy tensor")

    def apply_optimizer_state(
        self,
        target: D0TrainingState,
        state_tensor: CapturedTensor | None,
        slots: dict[tuple[int, str, str], CapturedTensor],
        scalar_state: Any,
    ) -> None:
        if state_tensor is not None:
            raise D0TrainingError("unexpected legacy optimizer state tensor")
        by_key = {
            f"optimizer.{slot_name}.{param_name}": torch.frombuffer(
                bytearray(captured.raw_bytes), dtype=torch.float32
            ).reshape(tuple(captured.shape))
            for (_group, param_name, slot_name), captured in slots.items()
        }
        target.optimizer.apply_state(by_key, safe_value_to_native(scalar_state))

    def apply_scheduler(self, target: D0TrainingState, tensor: CapturedTensor | None) -> None:
        if tensor is None:
            raise D0TrainingError("D0 scheduler captures a state tensor")

    def apply_scheduler_state(self, target: D0TrainingState, bundle: Any) -> None:
        scalar = getattr(bundle, "scalar_state", None)
        native = safe_value_to_native(scalar) if scalar is not None else {}
        last = native.get("last_applied_update")
        if type(last) is not int or last < 0:
            raise D0TrainingError("restored scheduler scalar lacks last_applied_update")
        # The float64 scalar LR is authoritative over the float32 tensor
        # (smoke precedent: a restored scheduler matches uninterrupted
        # byte-for-byte; the schedule derives it from the same frozen formula).
        target.last_applied_update = last

    def apply_scaler(self, target: D0TrainingState, tensor: CapturedTensor | None) -> None:
        if tensor is not None:
            raise D0TrainingError("D0 uses bf16 autocast: no scaler state")

    def apply_data_cursor(self, target: D0TrainingState, cursor: DataCursor) -> None:
        target.restored_cursor = cursor

    def apply_counters(self, target: D0TrainingState, counters: CounterSnapshot) -> None:
        target.restored_counters = counters

    def commit_to_live(self, target: D0TrainingState) -> D0TrainingState:
        self.committed = True
        self.target = target
        return target

    def revert_commit_to_live(self, target: D0TrainingState) -> None:
        self.committed = False
        self.reverted = True

    @property
    def restored_cursor(self) -> DataCursor:
        target = self.target
        if target is None or target.restored_cursor is None:
            raise D0TrainingError("no data cursor was restored")
        return target.restored_cursor

    @property
    def restored_counters(self) -> CounterSnapshot:
        target = self.target
        if target is None or target.restored_counters is None:
            raise D0TrainingError("no counters were restored")
        return target.restored_counters
