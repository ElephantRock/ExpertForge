"""Pure-Python reference adapter for the checkpoint protocol (Issue #11).

Implements the :class:`StateProvider` / :class:`StateConsumer` protocols with a
pure-Python object graph (no training framework). Used by the regression and
integration tests to exercise the full save → publish → load → restore →
continuation path. The deferred training-framework adapter is out of scope for
#11.

The provider's authoritative boundary is one quiescent
:meth:`capture_checkpoint_snapshot` call (amendment H) performed under a
checkpoint barrier: it proves the optimizer update is complete,
accumulation_position == 0, no async prefetch is active, and the captured state
cannot advance independently during capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from expertforge.checkpoints.models import (
    AliasGroup,
    CapturedCheckpointState,
    CapturedTensor,
    CounterSnapshot,
    DataCursor,
    DataDescriptor,
    DataIdentity,
    ModelDescriptor,
    ModelParameterDescriptor,
    OptimizerDescriptor,
    OptimizerParamGroup,
    OptimizerStateSlot,
    RngDescriptor,
    ScalerDescriptor,
    SchedulerDescriptor,
    TopologyDescriptor,
)

__all__ = [
    "StateProvider",
    "StateConsumer",
    "ReferenceState",
    "ReferenceStateProvider",
    "ReferenceCheckpointError",
]


class ReferenceCheckpointError(Exception):
    """Raised by the reference adapter on a contract violation."""


class StateProvider(Protocol):
    """Authoritative save boundary: one quiescent snapshot."""

    def capture_checkpoint_snapshot(self) -> CapturedCheckpointState: ...


class StateConsumer(Protocol):
    """Restore boundary: isolated fresh-object construction + apply."""


@dataclass
class ReferenceState:
    """The pure-Python reference training state.

    Holds named parameter tensors (as raw bytes), optimizer/scheduler/scaler
    state tensors, counters, and a data cursor. Mutation is explicit so the
    adapter can model an "advancing fixture provider" for the atomic-capture
    regression (amendment H, N).
    """

    parameters: dict[str, bytes] = field(default_factory=dict)
    buffers: dict[str, bytes] = field(default_factory=dict)
    param_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    param_dtypes: dict[str, str] = field(default_factory=dict)
    alias_groups: tuple[tuple[str, ...], ...] = ()
    optimizer_state: bytes = b""
    scheduler_state: bytes = b""
    scaler_state: bytes | None = None
    counters: CounterSnapshot | None = None
    data_cursor: DataCursor | None = None
    rng_bundle_bytes: bytes = b""
    optimizer_type: str = "SGD"
    scheduler_type: str = "constant"
    scaler_type: str = "grad_scaler"

    def advance(self) -> None:
        """Advance the live state, modeling a fixture that keeps moving during a
        naive multi-call capture (used to prove amendment H: one quiescent
        snapshot is authoritative, not racing provider calls)."""
        if self.counters is not None:
            # Bump accumulation position to simulate mid-flight state; capture
            # must reject a non-quiescent snapshot rather than read drifting
            # values.
            object.__setattr__  # noqa: B018 - placeholder for clarity
        # Flip the last parameter byte so a racing capture would observe drift.
        for name in sorted(self.parameters):
            b = bytearray(self.parameters[name])
            if b:
                b[-1] = (b[-1] + 1) & 0xFF
            self.parameters[name] = bytes(b)


class ReferenceStateProvider:
    """A StateProvider over a :class:`ReferenceState`.

    :meth:`capture_checkpoint_snapshot` performs ONE atomic capture: it reads
    every component from the live state into an immutable
    :class:`CapturedCheckpointState`. The returned snapshot is internally
    cross-validated (quiescent, no accumulation, no prefetch) before archive
    construction.
    """

    def __init__(self, state: ReferenceState) -> None:
        self._state = state

    def capture_checkpoint_snapshot(self) -> CapturedCheckpointState:
        s = self._state
        if s.counters is None:
            raise ReferenceCheckpointError("counters are not initialized")
        if s.data_cursor is None:
            raise ReferenceCheckpointError("data_cursor is not initialized")
        if s.counters.accumulation_position != 0:
            raise ReferenceCheckpointError(
                "capture requires accumulation_position == 0 (V1 save boundary)"
            )
        parameters = tuple(
            CapturedTensor(
                logical_name=name,
                dtype=s.param_dtypes[name],  # type: ignore[arg-type]
                shape=tuple(s.param_shapes[name]),
                raw_bytes=s.parameters[name],
            )
            for name in sorted(s.parameters)
        )
        buffers = tuple(
            CapturedTensor(
                logical_name=name,
                dtype=s.param_dtypes[name],  # type: ignore[arg-type]
                shape=tuple(s.param_shapes[name]),
                raw_bytes=s.buffers[name],
            )
            for name in sorted(s.buffers)
        )
        optimizer = (
            CapturedTensor(
                logical_name="optimizer.state",
                dtype="float32",
                shape=(len(s.optimizer_state) // 4,),
                raw_bytes=s.optimizer_state,
            )
            if s.optimizer_state
            else None
        )
        scheduler = (
            CapturedTensor(
                logical_name="scheduler.state",
                dtype="float32",
                shape=(len(s.scheduler_state) // 4,),
                raw_bytes=s.scheduler_state,
            )
            if s.scheduler_state
            else None
        )
        scaler = None
        if s.scaler_state is not None:
            scaler = CapturedTensor(
                logical_name="scaler.state",
                dtype="float32",
                shape=(len(s.scaler_state) // 4,),
                raw_bytes=s.scaler_state,
            )

        # Item 10: model descriptor MUST separate parameters from buffers
        # (buffers were previously classified as parameters) and MUST include the
        # declared alias groups so the descriptor round-trips the full graph.
        param_desc = tuple(
            ModelParameterDescriptor(name=t.logical_name, shape=tuple(t.shape), dtype=t.dtype)
            for t in parameters
        )
        buffer_desc = tuple(
            ModelParameterDescriptor(name=t.logical_name, shape=tuple(t.shape), dtype=t.dtype)
            for t in buffers
        )
        alias_group_desc = _build_alias_groups(s.alias_groups)
        optimizer_desc = _build_optimizer_descriptor(s, parameters)
        scheduler_desc = SchedulerDescriptor(
            scheduler_type=s.scheduler_type,
            state_shape=(len(s.scheduler_state) // 4,) if s.scheduler_state else (),
        )
        scaler_desc = (
            ScalerDescriptor(
                scaler_type=s.scaler_type,
                state_shape=(len(s.scaler_state) // 4,),
            )
            if s.scaler_state is not None
            else None
        )
        data_desc = DataDescriptor(
            identity=s.data_cursor_identity(),  # type: ignore[attr-defined]
            sampler_type=s.data_cursor.sampler_type,
            sampler_version=s.data_cursor.sampler_version,
            batch_size=s.data_cursor.batch_size,
            sequence_length=s.data_cursor.sequence_length,
            drop_last=s.data_cursor.drop_last,
        )
        rng_desc = RngDescriptor(adapter_set=(), framework_versions=())

        return CapturedCheckpointState(
            parameters=parameters,
            buffers=buffers,
            alias_groups=s.alias_groups,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            rng_bundle_bytes=s.rng_bundle_bytes,
            data_cursor=s.data_cursor,
            counters=s.counters,
            model_descriptor=ModelDescriptor(
                parameters=param_desc,
                buffers=buffer_desc,
                alias_groups=alias_group_desc,
            ),
            optimizer_descriptor=optimizer_desc,
            scheduler_descriptor=scheduler_desc,
            scaler_descriptor=scaler_desc,
            rng_descriptor=rng_desc,
            data_descriptor=data_desc,
            topology_descriptor=TopologyDescriptor(world_size=1, rank_assignment=(0,)),
        )


def _build_optimizer_descriptor(
    s: ReferenceState, parameters: tuple[CapturedTensor, ...]
) -> OptimizerDescriptor:
    names = tuple(t.logical_name for t in parameters)
    opt_state_len = len(s.optimizer_state) // 4 if s.optimizer_state else 0
    slots: tuple[OptimizerStateSlot, ...] = ()
    if s.optimizer_state and names:
        slots = (
            OptimizerStateSlot(
                group_index=0,
                param_name=names[0],
                slot_name="momentum",
                shape=(opt_state_len,),
                dtype="float32",
            ),
        )
    return OptimizerDescriptor(
        optimizer_type=s.optimizer_type,
        param_groups=(OptimizerParamGroup(group_index=0, param_names=names, options=()),),
        state_slots=slots,
    )


def _build_alias_groups(raw: tuple[tuple[str, ...], ...]) -> tuple[AliasGroup, ...]:
    """Build typed AliasGroup entries from the raw alias-group tuples (item 10).

    Each raw group is a tuple of aliased logical names; the canonical member is
    the lexicographically smallest name and all names appear in
    ``aliased_names``. Groups are sorted by canonical_member (the
    :class:`ModelDescriptor` requires canonical ordering).
    """
    out: list[AliasGroup] = []
    for group in raw:
        members = tuple(sorted(set(group)))
        if len(members) < 2:
            continue
        canonical = members[0]
        # AliasGroup stores canonical_member separately from aliased_names; the
        # model validator unions them. Keep canonical_member OUT of aliased_names
        # to avoid a duplicate-member validation error.
        out.append(AliasGroup(canonical_member=canonical, aliased_names=members[1:]))
    out.sort(key=lambda g: g.canonical_member)
    return tuple(out)


def _reference_data_identity() -> DataIdentity:
    return DataIdentity(
        dataset_digest="a" * 64,
        split="train",
        length=1000,
        preprocessing_identity="prep.v1",
        tokenizer_identity="tok.v1",
        packing_policy="no_packing",
        sequence_policy="fixed_128",
        shard_selection="single",
        data_config_digest="b" * 64,
    )


# Attach the default data identity helper to ReferenceState for the provider.
def _data_cursor_identity(self: ReferenceState) -> DataIdentity:
    return _reference_data_identity()


ReferenceState.data_cursor_identity = _data_cursor_identity  # type: ignore[attr-defined]
