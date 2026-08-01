"""Restore transaction: factory/commit/abort with RNG rollback (Issue #11, K).

Exact restoration requires a real transaction boundary:

1. Decode and validate the **complete archive** before mutating any live state.
   A mandatory :func:`check_compatibility` gate (amendment L/M) requires the
   archive to be ``exact`` against the expected descriptor before any state is
   applied to fresh objects.
2. Construct fresh model/optimizer/scheduler/scaler/data state via a factory.
3. Apply and validate all decoded state to the fresh objects (parameters,
   buffers, optimizer, scheduler, scaler, data cursor, counters) — in order.
4. Restore RNG **last** so loading operations do not alter the resumed RNG
   stream. Prevalidate the target :class:`RngStateBundle`; capture the current
   bundle immediately before commit. RNG restoration is the FINAL step of the
   two-phase commit: non-RNG state is applied to fresh objects first, and only
   after RNG restoration succeeds is the non-RNG swap committed to live. If RNG
   restoration fails, the previous RNG bundle is restored AND the non-RNG swap
   is reverted so pre-restore live state remains authoritative.
5. Expose an atomic :meth:`commit` / idempotent :meth:`abort` boundary. Existing
   live state is untouched before commit. Cleanup failures are attached
   diagnostically and never replace the original exception.

Failure-injection points (:attr:`RestoreTransaction.fail_after`) exercise every
stage, including RNG rollback.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from expertforge.checkpoints.models import (
    CapturedTensor,
    CompatibilityDescriptor,
    CompatibilityResult,
    CounterSnapshot,
    DataCursor,
    OptimizerDescriptor,
    SafeValue,
    ScalerDescriptor,
    SchedulerDescriptor,
    TensorDtype,
    TensorMemberRef,
)
from expertforge.checkpoints.store import CheckpointArchive
from expertforge.rng.state import RngStateBundle

# A resolved optimizer state bundle: the legacy single tensor (when present), the
# full multi-slot tensors keyed by (group, param, slot), and the structured
# scalar state (e.g. step counters). Factories consume this via
# :meth:`StateFactory.apply_optimizer_state` (item 3).
OptimizerStateBundle = tuple[
    CapturedTensor | None,
    dict[tuple[int, str, str], CapturedTensor],
    SafeValue | None,
]


__all__ = [
    "RestoreError",
    "RestoreIncompatibleError",
    "RestoreStage",
    "RestoredState",
    "StateConsumer",
    "StateFactory",
    "RestoreTransaction",
    "decode_component_json",
    "build_restored_tensors",
    "TensorComponentRef",
    "ModelState",
    "OptimizerState",
    "OptimizerStateBundle",
    "SchedulerState",
    "SchedulerStateBundle",
    "ScalerState",
]


class RestoreError(Exception):
    """Raised when a restore step fails (state remains untouched)."""


class RestoreIncompatibleError(RestoreError):
    """Raised when the mandatory compatibility gate is not ``exact`` (item 8)."""


RestoreStage = Literal[
    "compatibility",
    "factory",
    "parameters",
    "buffers",
    "optimizer",
    "scheduler",
    "scaler",
    "data_cursor",
    "counters",
    "rng_prevalidate",
    "rng_capture_previous",
    "commit_non_rng",
    "rng_restore",
]


class StateFactory(Protocol):
    """Constructs isolated fresh objects for the transaction (amendment K)."""

    def create(self) -> Any: ...

    def apply_parameters(self, target: Any, tensors: dict[str, CapturedTensor]) -> None: ...

    def apply_buffers(self, target: Any, tensors: dict[str, CapturedTensor]) -> None: ...

    def apply_optimizer(self, target: Any, tensor: CapturedTensor | None) -> None: ...

    def apply_optimizer_state(
        self,
        target: Any,
        state_tensor: CapturedTensor | None,
        slots: dict[tuple[int, str, str], CapturedTensor],
        scalar_state: SafeValue | None,
    ) -> None: ...

    def apply_scheduler(self, target: Any, tensor: CapturedTensor | None) -> None: ...

    def apply_scaler(self, target: Any, tensor: CapturedTensor | None) -> None: ...

    def apply_data_cursor(self, target: Any, cursor: DataCursor) -> None: ...

    def apply_counters(self, target: Any, counters: CounterSnapshot) -> None: ...

    def commit_to_live(self, target: Any) -> Any: ...

    def revert_commit_to_live(self, target: Any) -> None:
        """Revert a previously-applied non-RNG swap when RNG restoration fails.

        Called ONLY after :meth:`commit_to_live` succeeded but RNG restoration
        subsequently failed. The default implementation is a no-op for factories
        whose ``commit_to_live`` does not mutate shared live state; factories
        that swap live handles MUST override this to restore the pre-commit
        handles (item 1).
        """


class StateConsumer(Protocol):
    """A minimal consumer protocol for the restored state."""


class RestoredState:
    """The committed restored state handed back to the caller on success."""

    __slots__ = ("target", "manifest", "artifact_id")

    def __init__(self, *, target: Any, manifest: Any, artifact_id: str) -> None:
        self.target = target
        self.manifest = manifest
        self.artifact_id = artifact_id


# ---------------------------------------------------------------------------
# Strict component payload schemas (item 12)
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Frozen, extra-forbid, strict base for component payload schemas."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
        populate_by_name=True,
    )


class TensorComponentRef(_StrictModel):
    """A strict tensor reference inside a component payload.

    Cross-bound to the manifest's :class:`TensorMemberRef` via
    :meth:`RestoreTransaction`'s member-graph validation: the ``member_name``,
    ``dtype``, ``shape``, and ``logical_names`` MUST match exactly one
    manifest tensor member.
    """

    member_name: str = Field(..., min_length=1)
    logical_names: tuple[str, ...] = Field(..., min_length=1)
    dtype: TensorDtype
    shape: tuple[int, ...] = Field(default_factory=tuple)


class ModelState(_StrictModel):
    """The strict ``state/model.json`` component payload (item 12)."""

    schema_: Literal["expertforge.checkpoint-model-state"] = Field(
        alias="schema", default="expertforge.checkpoint-model-state"
    )
    version: Literal[1] = 1
    parameters: tuple[TensorComponentRef, ...] = Field(default_factory=tuple)
    buffers: tuple[TensorComponentRef, ...] = Field(default_factory=tuple)


class OptimizerState(_StrictModel):
    """The strict ``state/optimizer.json`` component payload (item 12).

    Carries the legacy single ``state_tensor`` (when the optimizer has exactly
    one tensor slot), the optional multi-slot ``state_slots`` (general case),
    the optional structured ``scalar_state``, and the optimizer descriptor.
    """

    schema_: Literal["expertforge.checkpoint-optimizer-state"] = Field(
        alias="schema", default="expertforge.checkpoint-optimizer-state"
    )
    version: Literal[1] = 1
    state_tensor: TensorComponentRef | None = None
    state_slots: tuple[TensorComponentRef, ...] = Field(default_factory=tuple)
    scalar_state: SafeValue | None = None
    descriptor: OptimizerDescriptor
    has_state: bool = True


class SchedulerState(_StrictModel):
    """The strict ``state/scheduler.json`` component payload (item 12)."""

    schema_: Literal["expertforge.checkpoint-scheduler-state"] = Field(
        alias="schema", default="expertforge.checkpoint-scheduler-state"
    )
    version: Literal[1] = 1
    state_tensor: TensorComponentRef | None = None
    scalar_state: SafeValue | None = None
    descriptor: SchedulerDescriptor
    has_state: bool = True


class SchedulerStateBundle(_StrictModel):
    """A resolved scheduler state bundle (item 4).

    Carries the optional scheduler state tensor AND the optional structured
    scalar state (e.g. a step counter, last-epoch LR, or other non-tensor
    scheduler bookkeeping). Either may be None:

    - tensor-only scheduler (the legacy case): ``tensor`` set, ``scalar_state``
      None;
    - scalar-only scheduler (a scheduler with no tensor state, e.g. a
      step-counter-only scheduler): ``tensor`` None, ``scalar_state`` set;
    - mixed: both set.

    Factories consume this via :meth:`StateFactory.apply_scheduler_state`. The
    restore path routes a scalar-bearing scheduler through the full-state method
    so scalar state is never silently dropped (mirroring the optimizer item-3
    fix).
    """

    tensor: CapturedTensor | None = None
    scalar_state: SafeValue | None = None


class ScalerState(_StrictModel):
    """The strict ``state/scaler.json`` component payload (item 12)."""

    schema_: Literal["expertforge.checkpoint-scaler-state"] = Field(
        alias="schema", default="expertforge.checkpoint-scaler-state"
    )
    version: Literal[1] = 1
    state_tensor: TensorComponentRef | None = None
    descriptor: ScalerDescriptor
    has_state: bool = True


def decode_component_json(raw: bytes) -> dict[str, Any]:
    """Decode a ``state/<role>.json`` component with strict canonical checks.

    Rejects duplicate keys, non-UTF-8 input, and non-canonical JSON
    representation. Duplicate-key ``ValueError`` from the object-pairs hook is
    converted to a typed :class:`RestoreError` so it cannot escape the typed
    corruption boundary (item 12).
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise RestoreError(f"component JSON is not valid UTF-8: {e}") from e
    try:
        data = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as e:
        raise RestoreError(f"component JSON is not valid JSON: {e}") from e
    except ValueError as e:
        raise RestoreError(f"component JSON rejected: {e}") from e
    if not isinstance(data, dict):
        raise RestoreError("component JSON root must be an object")
    # Require canonical compact sorted JSON.
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    if canonical != raw:
        raise RestoreError("component JSON is not canonical compact sorted JSON")
    return data


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is not allowed: {key}")
        result[key] = value
    return result


def build_restored_tensors(
    archive: CheckpointArchive,
) -> tuple[
    dict[str, CapturedTensor],
    dict[str, CapturedTensor],
    dict[str, CapturedTensor],
    OptimizerStateBundle,
    SchedulerStateBundle,
    dict[str, int],
]:
    """Decode the model/optimizer/scheduler/scaler component refs and materialize
    the captured tensors from the archive's tensor members.

    Returns ``(parameters, buffers, state_tensors_by_role, optimizer_bundle,
    scheduler_bundle, member_index)`` where:

    - ``optimizer_bundle`` is ``(legacy_tensor, slots, scalar_state)`` (item 3);
    - ``scheduler_bundle`` is a :class:`SchedulerStateBundle` carrying the
      scheduler tensor (when present) AND the structured scalar_state (item 4):
      tensor-only, scalar-only, and mixed schedulers all round-trip.

    Every component tensor reference is strict-schema-validated (item 5) and
    cross-bound to the manifest's :class:`TensorMemberRef`: ``member_name``,
    ``dtype``, ``shape``, and ``logical_names`` must match exactly one manifest
    member. For alias groups, the tensor is inserted for EVERY logical name in
    the group (item 3). Buffers are materialized and returned so the caller can
    apply them (item 2).
    """
    manifest = archive.manifest
    members_by_name: dict[str, TensorMemberRef] = {
        m.member_name: m for m in manifest.tensor_members
    }
    model_payload = decode_component_json(archive.component("model"))
    try:
        # Item 5: strict model_validate so JSON lists are NOT coerced to the
        # frozen tuple fields. The frozen extra-forbid strict model rejects
        # unknown keys, wrong types, AND non-tuple list inputs.
        model_state = ModelState.model_validate_json(
            json.dumps(model_payload, sort_keys=True, separators=(",", ":")), strict=True
        )
    except ValidationError as e:
        raise RestoreError(f"model component failed strict validation: {e}") from e

    parameters: dict[str, CapturedTensor] = {}
    for entry in model_state.parameters:
        tensor = _materialize(archive, entry, members_by_name)
        # Item 3: set the tensor for EVERY logical name in the alias group.
        for name in entry.logical_names:
            parameters[name] = tensor
    # Buffers: materialize and insert for every logical name (item 2/3).
    buffers: dict[str, CapturedTensor] = {}
    for entry in model_state.buffers:
        tensor = _materialize(archive, entry, members_by_name)
        for name in entry.logical_names:
            buffers[name] = tensor

    state_tensors: dict[str, CapturedTensor] = {}
    optimizer_slots: dict[tuple[int, str, str], CapturedTensor] = {}
    optimizer_legacy: CapturedTensor | None = None
    optimizer_scalar: SafeValue | None = None
    # Item 4: scheduler carries an optional tensor AND an optional scalar_state.
    scheduler_tensor: CapturedTensor | None = None
    scheduler_scalar: SafeValue | None = None
    for role in ("optimizer", "scheduler", "scaler"):
        comp_ref = next((c for c in manifest.state_components if c.role == role), None)
        if comp_ref is None:
            continue
        payload = decode_component_json(archive.component(role))
        try:
            if role == "optimizer":
                parsed: OptimizerState | SchedulerState | ScalerState = (
                    OptimizerState.model_validate_json(
                        json.dumps(payload, sort_keys=True, separators=(",", ":")),
                        strict=True,
                    )
                )
            elif role == "scheduler":
                parsed = SchedulerState.model_validate_json(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")), strict=True
                )
            else:
                parsed = ScalerState.model_validate_json(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")), strict=True
                )
        except ValidationError as e:
            raise RestoreError(f"{role} component failed strict validation: {e}") from e
        # A stateless component (has_state=False) carries no tensor/scalar.
        if not parsed.has_state:
            continue
        if role == "optimizer":
            # Item 3: materialize the legacy single tensor AND every multi-slot
            # tensor, plus capture the structured scalar state. Each slot tensor
            # is keyed by its descriptor (group, param, slot) tuple so the
            # factory can apply them by slot identity. Cast parsed to the
            # optimizer type — the branch above assigned an OptimizerState.
            opt_state: OptimizerState = parsed  # type: ignore[assignment]
            if opt_state.state_tensor is not None:
                optimizer_legacy = _materialize(archive, opt_state.state_tensor, members_by_name)
            opt_descriptor = manifest.compatibility.optimizer_descriptor
            for slot_ref in opt_state.state_slots:
                slot_tensor = _materialize(archive, slot_ref, members_by_name)
                # Item 3: resolve the slot's group_index against the descriptor
                # (the authoritative source) rather than hardcoding group_index=0.
                slot_key = _slot_key_from_logical_names(
                    slot_ref.logical_names, slot_ref.member_name, opt_descriptor
                )
                optimizer_slots[slot_key] = slot_tensor
            optimizer_scalar = opt_state.scalar_state
            # Also expose the legacy tensor under the role for compatibility.
            if optimizer_legacy is not None:
                state_tensors["optimizer"] = optimizer_legacy
            continue
        if role == "scheduler":
            # Item 4: build the SchedulerStateBundle from the optional tensor AND
            # the optional scalar_state. A scalar-only scheduler (no tensor) is
            # valid and no longer raises; a mixed scheduler carries both.
            sched_state: SchedulerState = parsed  # type: ignore[assignment]
            if sched_state.state_tensor is not None:
                scheduler_tensor = _materialize(archive, sched_state.state_tensor, members_by_name)
            scheduler_scalar = sched_state.scalar_state
            # A scheduler with has_state=True must carry at least one of
            # tensor/scalar; an empty has_state=True scheduler is corruption.
            if scheduler_tensor is None and scheduler_scalar is None:
                raise RestoreError(
                    "scheduler component has_state=True but carries neither a "
                    "state_tensor nor a scalar_state"
                )
            # Keep the legacy tensor view for backward-compatible consumers.
            if scheduler_tensor is not None:
                state_tensors["scheduler"] = scheduler_tensor
            continue
        ref = parsed.state_tensor
        if ref is None:
            raise RestoreError(f"component {role!r} is missing state_tensor")
        tensor = _materialize(archive, ref, members_by_name)
        state_tensors[role] = tensor

    scheduler_bundle = SchedulerStateBundle(tensor=scheduler_tensor, scalar_state=scheduler_scalar)

    member_index = {m.member_name: i for i, m in enumerate(manifest.tensor_members)}
    return (
        parameters,
        buffers,
        state_tensors,
        (optimizer_legacy, optimizer_slots, optimizer_scalar),
        scheduler_bundle,
        member_index,
    )


def _slot_key_from_logical_names(
    logical_names: tuple[str, ...],
    member_name: str,
    opt_descriptor: OptimizerDescriptor,
) -> tuple[int, str, str]:
    """Derive the (group_index, param_name, slot_name) key for an optimizer slot.

    Item 3: the group_index is resolved from the optimizer descriptor — the
    authoritative source — by matching the slot's ``(param_name, slot_name)``
    (parsed from the canonical logical name
    ``optimizer.<slot_name>.<param_name>``) to a declared
    :class:`OptimizerStateSlot`. This preserves GROUP IDENTITY for multi-group
    optimizers: a slot whose (param, slot) appears in group 1 is keyed by
    group_index=1, not silently collapsed to group 0.

    The logical name does NOT carry the group index (the descriptor does), so
    matching is by (param, slot). If the same (param, slot) is declared in
    multiple groups, the match is ambiguous and the slot is rejected — providers
    must disambiguate via distinct slot_name or param_name per group. A slot
    declared in exactly one group resolves to that group's index.
    """
    if not logical_names:
        raise RestoreError(f"optimizer slot {member_name!r} has no logical_names")
    name = logical_names[0]
    parts = name.split(".")
    # Expected form: optimizer.<slot_name>.<param_name> (param may contain dots).
    if len(parts) < 3 or parts[0] != "optimizer":
        raise RestoreError(
            f"optimizer slot logical_name {name!r} must be 'optimizer.<slot_name>.<param_name>'"
        )
    slot_name = parts[1]
    param_name = ".".join(parts[2:])
    if not slot_name or not param_name:
        raise RestoreError(f"optimizer slot logical_name {name!r} is malformed")
    # Resolve the group_index against the descriptor (the authoritative source).
    matches = [
        s.group_index
        for s in opt_descriptor.state_slots
        if s.param_name == param_name and s.slot_name == slot_name
    ]
    if not matches:
        raise RestoreError(
            f"optimizer slot ({param_name!r}, {slot_name!r}) is not declared in the "
            "optimizer descriptor state_slots"
        )
    if len(set(matches)) > 1:
        raise RestoreError(
            f"optimizer slot ({param_name!r}, {slot_name!r}) is ambiguous: declared in "
            f"groups {sorted(set(matches))}; disambiguate via distinct slot/param names"
        )
    return (matches[0], param_name, slot_name)


def _apply_optimizer_state_to_factory(
    factory: Any,
    target: Any,
    state_tensor: CapturedTensor | None,
    slots: dict[tuple[int, str, str], CapturedTensor],
    scalar_state: SafeValue | None,
) -> None:
    """Apply the full optimizer state bundle to ``factory`` (item 3).

    The optimizer carries structured state beyond a single legacy tensor:
    multi-slot tensors (keyed by ``(group_index, param_name, slot_name)``) and
    a structured ``scalar_state``. When EITHER is present, the factory MUST
    implement :meth:`apply_optimizer_state`; falling back to the legacy
    single-tensor :meth:`apply_optimizer` would SILENTLY DROP the slots/scalars,
    producing an incomplete restore. This is now a hard error rather than a
    silent fallback (item 3).

    When the optimizer carries ONLY the legacy single tensor (no slots, no
    scalar_state), the legacy :meth:`apply_optimizer` path is still permitted so
    minimal reference factories keep working.
    """
    has_structured = bool(slots) or scalar_state is not None
    apply_state = getattr(factory, "apply_optimizer_state", None)
    if callable(apply_state):
        apply_state(target, state_tensor, slots, scalar_state)
        return
    if has_structured:
        raise RestoreError(
            "optimizer carries structured state (multi-slot tensors or scalar_state) "
            "but the factory does not implement apply_optimizer_state; refusing to "
            "silently drop slots/scalars via the legacy apply_optimizer fallback"
        )
    # Legacy path: single tensor only, no structured state.
    factory.apply_optimizer(target, state_tensor)


def _apply_scheduler_state_to_factory(
    factory: Any,
    target: Any,
    bundle: SchedulerStateBundle,
) -> None:
    """Apply the scheduler state bundle to ``factory`` (item 4).

    The scheduler may carry a tensor, a scalar_state, or both. When the bundle
    carries scalar_state, the factory MUST implement
    :meth:`apply_scheduler_state` so the scalar state is consumed (falling back
    to the legacy :meth:`apply_scheduler` would silently drop it). A tensor-only
    scheduler may use the legacy path.
    """
    has_scalar = bundle.scalar_state is not None
    apply_state = getattr(factory, "apply_scheduler_state", None)
    if callable(apply_state):
        apply_state(target, bundle)
        return
    if has_scalar:
        raise RestoreError(
            "scheduler carries scalar_state but the factory does not implement "
            "apply_scheduler_state; refusing to silently drop scalar state via "
            "the legacy apply_scheduler fallback"
        )
    # Legacy path: tensor only (or empty), no scalar state.
    factory.apply_scheduler(target, bundle.tensor)


def _materialize(
    archive: CheckpointArchive,
    entry: TensorComponentRef,
    members_by_name: dict[str, TensorMemberRef],
) -> CapturedTensor:
    """Materialize a :class:`CapturedTensor` from a component ref + member bytes.

    Cross-binds the component reference to the manifest's
    :class:`TensorMemberRef` (item 12): member_name, dtype, shape, and
    logical_names must all match. This prevents the same authenticated bytes from
    being reinterpreted under different component metadata.
    """
    member_ref = members_by_name.get(entry.member_name)
    if member_ref is None:
        raise RestoreError(f"tensor member {entry.member_name!r} is not declared in the manifest")
    if entry.dtype != member_ref.dtype:
        raise RestoreError(
            f"component ref dtype {entry.dtype!r} != manifest dtype {member_ref.dtype!r} "
            f"for {entry.member_name!r}"
        )
    if tuple(entry.shape) != tuple(member_ref.shape):
        raise RestoreError(
            f"component ref shape {tuple(entry.shape)!r} != manifest shape "
            f"{tuple(member_ref.shape)!r} for {entry.member_name!r}"
        )
    if tuple(entry.logical_names) != tuple(member_ref.logical_names):
        raise RestoreError(
            f"component ref logical_names {tuple(entry.logical_names)!r} != manifest "
            f"logical_names {tuple(member_ref.logical_names)!r} for {entry.member_name!r}"
        )
    raw = archive.tensor_member_buffer(entry.member_name)
    if len(raw) != member_ref.member_byte_size:
        raise RestoreError(
            f"tensor member {entry.member_name!r} byte size changed: "
            f"{len(raw)} != {member_ref.member_byte_size}"
        )
    if entry.dtype == "bool" and any(value not in (0, 1) for value in raw):
        raise RestoreError(f"bool tensor member {entry.member_name!r} contains non-canonical bytes")
    if isinstance(raw, bytes):
        return CapturedTensor(
            logical_name=entry.logical_names[0],
            dtype=entry.dtype,
            shape=tuple(entry.shape),
            raw_bytes=raw,
        )
    # CapturedTensor's public validator is intentionally bytes-only. The archive
    # has already authenticated size/digest and the checks above enforce the
    # remaining raw-byte invariants, so construct the framework-neutral record
    # with a read-only mmap-backed memoryview rather than copying up to 1 GiB.
    return CapturedTensor.model_construct(
        logical_name=entry.logical_names[0],
        dtype=entry.dtype,
        shape=tuple(entry.shape),
        byte_order="little",
        layout="contiguous",
        raw_bytes=cast(Any, raw),
        source_device=None,
    )


class RestoreTransaction:
    """The factory/commit/abort restore transaction (amendment K).

    Existing live state is untouched before :meth:`commit`. The transaction:

    - runs a mandatory compatibility gate: the archive must be ``exact`` against
      the expected descriptor before ANY state application (item 8, amendment
      L/M);
    - constructs fresh objects via :class:`StateFactory`;
    - applies parameters, buffers, optimizer, scheduler, scaler, data cursor,
      counters (in order) to the fresh objects;
    - prevalidates the target :class:`RngStateBundle`;
    - captures the current RNG bundle immediately before commit;
    - two-phase commit (item 1): commits the non-RNG swap, THEN restores RNG
      last. If RNG restoration fails, restores the previous RNG bundle AND
      reverts the non-RNG swap via :meth:`StateFactory.revert_commit_to_live` so
      the pre-restore live state is authoritative.

    :attr:`fail_after` and :attr:`rng_restore_should_fail` are failure-injection
    points used by the focused rollback tests.
    """

    def __init__(
        self,
        *,
        archive: CheckpointArchive,
        factory: StateFactory,
        rng_bundle_loader: Callable[[], RngStateBundle],
        rng_consumer: Callable[[RngStateBundle], None],
        expected_descriptor: CompatibilityDescriptor,
        compatibility_checker: Callable[
            [CheckpointArchive, CompatibilityDescriptor], CompatibilityResult
        ]
        | None = None,
    ) -> None:
        self._archive = archive
        self._factory = factory
        self._rng_bundle_loader = rng_bundle_loader
        self._rng_consumer = rng_consumer
        # Item 1: expected_descriptor is REQUIRED (no default None). There is no
        # ungated restore path: every transaction must prove the archive is
        # exactly compatible with the consumer's descriptor before any state is
        # applied.
        self._expected_descriptor = expected_descriptor
        self._compatibility_checker = compatibility_checker
        # Failure injection: if set, raise RestoreError after this stage.
        self.fail_after: RestoreStage | None = None
        # Pluggable RNG capture-previous (for the focused rollback test that
        # simulates RNG restore failure after capturing the previous bundle).
        self.rng_restore_should_fail: bool = False
        self._committed = False
        self._aborted = False
        self._target: Any = None
        self._previous_rng: RngStateBundle | None = None
        self._prepared_rng: RngStateBundle | None = None
        # Whether the non-RNG swap was committed to live (for rollback on RNG
        # failure).
        self._non_rng_committed: bool = False
        self._live: Any = None
        # Item 4: diagnostics collected during independent two-domain rollback.
        # Attached to the original failure, never replacing it.
        self.cleanup_diagnostics: tuple[str, ...] = ()

    def require_exact_compatibility(self, expected: CompatibilityDescriptor) -> CompatibilityResult:
        """Mandatory compatibility gate (item 8).

        Runs :func:`check_compatibility` and raises
        :class:`RestoreIncompatibleError` unless the result is ``exact``. MUST be
        invoked before any state is applied. :meth:`prepare` calls this
        automatically when ``expected_descriptor`` is supplied.
        """
        if self._compatibility_checker is not None:
            result = self._compatibility_checker(self._archive, expected)
        else:
            from expertforge.checkpoints.store import check_compatibility

            result = check_compatibility(self._archive.manifest.compatibility, expected)
        self._maybe_fail("compatibility")
        if result.status != "exact":
            raise RestoreIncompatibleError(
                f"checkpoint is not exactly compatible with the expected descriptor "
                f"(status={result.status!r}, {len(result.mismatches)} mismatches)"
            )
        return result

    def prepare(self) -> None:
        """Apply all non-RNG state to fresh objects (no live mutation yet).

        ALWAYS runs the mandatory compatibility gate first (item 1): the
        expected descriptor is a required constructor input, so every restore
        proves the archive is exactly compatible before any state is applied.
        """
        self._check_open()
        # Item 1: the gate is unconditional (expected_descriptor is required).
        assert self._expected_descriptor is not None
        self.require_exact_compatibility(self._expected_descriptor)
        try:
            self._target = self._factory.create()
        except Exception as e:
            raise RestoreError(f"factory.create failed: {e}") from e
        self._maybe_fail("factory")

        (
            parameters,
            buffers,
            state_tensors,
            optimizer_bundle,
            scheduler_bundle,
            _,
        ) = build_restored_tensors(self._archive)

        # Apply parameters.
        self._factory.apply_parameters(self._target, parameters)
        self._maybe_fail("parameters")

        # Apply buffers (item 2): materialized buffers are now restored, not
        # dropped.
        self._factory.apply_buffers(self._target, buffers)
        self._maybe_fail("buffers")

        # Apply optimizer state (item 3): the full multi-slot tensors plus the
        # structured scalar state are applied together so a real multi-slot
        # optimizer round-trips. Structured state REQUIRES apply_optimizer_state
        # (no silent legacy fallback that drops slots/scalars).
        opt_legacy, opt_slots, opt_scalar = optimizer_bundle
        _apply_optimizer_state_to_factory(
            self._factory, self._target, opt_legacy, opt_slots, opt_scalar
        )
        self._maybe_fail("optimizer")

        # Apply scheduler state (item 4): the full SchedulerStateBundle (tensor
        # + scalar_state) is applied via apply_scheduler_state so scalar state is
        # never silently dropped. A scalar-bearing scheduler routes through the
        # full-state method; a tensor-only scheduler may use the legacy path.
        _apply_scheduler_state_to_factory(self._factory, self._target, scheduler_bundle)
        self._maybe_fail("scheduler")

        # Apply scaler state (if present).
        scaler_tensor = state_tensors.get("scaler")
        self._factory.apply_scaler(self._target, scaler_tensor)
        self._maybe_fail("scaler")

        # Apply data cursor.
        cursor_payload = decode_component_json(self._archive.component("data_cursor"))
        cursor = DataCursor.model_validate(cursor_payload, strict=True)
        self._factory.apply_data_cursor(self._target, cursor)
        self._maybe_fail("data_cursor")

        # Apply counters.
        counters_payload = decode_component_json(self._archive.component("counters"))
        counters = CounterSnapshot.model_validate(counters_payload, strict=True)
        self._factory.apply_counters(self._target, counters)
        self._maybe_fail("counters")

    def commit(self) -> RestoredState:
        """Atomically commit the prepared state + RNG (two-phase, items 1 + 4).

        Phase 1: commit the non-RNG swap to live (``commit_to_live``).
        Phase 2: restore RNG LAST. The ENTIRE post-commit path — including the
        ``commit_non_rng`` bookkeeping stage and the RNG restore — is wrapped in
        rollback handling (item 4): if ANY step after ``commit_to_live`` succeeds
        fails, BOTH the RNG domain and the non-RNG domain are rolled back
        INDEPENDENTLY (one cleanup failure must not prevent the other cleanup
        attempt), and cleanup diagnostics are ATTACHED to the original exception
        rather than replacing it. The pre-restore live state is authoritative.
        Existing live state is untouched before phase 1.
        """
        self._check_open()
        if self._target is None:
            raise RestoreError("commit() called before prepare()")
        # Prevalidate the target RNG bundle.
        rng_bytes = self._archive.component("rng")
        try:
            self._prepared_rng = RngStateBundle.from_json_bytes(rng_bytes)
        except Exception as e:
            raise RestoreError(f"rng prevalidate failed: {e}") from e
        self._maybe_fail("rng_prevalidate")

        # Capture the current (pre-commit) RNG bundle immediately before commit.
        self._previous_rng = self._rng_bundle_loader()
        self._maybe_fail("rng_capture_previous")

        # Phase 1: commit the prepared non-RNG state to live.
        try:
            self._live = self._factory.commit_to_live(self._target)
        except Exception as e:
            # Non-RNG commit failed before any RNG mutation; nothing to roll
            # back. Existing live state is untouched.
            raise RestoreError(f"commit_to_live failed: {e}") from e
        self._non_rng_committed = True

        # Item 4: wrap the ENTIRE post-commit path (commit_non_rng bookkeeping +
        # RNG restore) in rollback handling. A failure at ANY of these stages
        # rolls back BOTH domains independently so pre-restore live state is
        # authoritative, with cleanup diagnostics attached (not replacing the
        # original exception).
        try:
            self._maybe_fail("commit_non_rng")
            # Phase 2: restore RNG LAST.
            if self.rng_restore_should_fail:
                raise RestoreError("injected rng_restore failure")
            self._rng_consumer(self._prepared_rng)
        except Exception as e:
            self._rollback_phase2_independent()
            raise RestoreError(
                f"post-commit failure; rolled back RNG and non-RNG domains independently: {e}"
            ) from e

        self._committed = True
        return RestoredState(
            target=self._live,
            manifest=self._archive.manifest,
            artifact_id=self._archive.artifact_id,
        )

    def abort(self) -> None:
        """Idempotently abort the transaction.

        Discards the prepared fresh objects. Existing live state is untouched.
        """
        if self._aborted:
            return
        self._aborted = True
        self._target = None
        self._prepared_rng = None
        # If we captured the previous RNG but never committed, there is nothing
        # to roll back (we never mutated the live RNG stream).

    def _rollback_phase2_independent(self) -> None:
        """Revert BOTH post-commit domains INDEPENDENTLY after a failure (item 4).

        The RNG domain (the previously-captured bundle) and the non-RNG domain
        (the committed ``commit_to_live`` swap) are each reverted in their OWN
        try/except so a cleanup failure in ONE domain does not prevent the
        cleanup attempt in the OTHER. Cleanup diagnostics are recorded on
        :attr:`cleanup_diagnostics` and ATTACHED to the caller's exception
        (via raise ... from), never replacing the original failure.

        At the ``commit_non_rng`` stage the RNG has been captured but not yet
        mutated, so the RNG revert is a no-op (restoring the previous bundle
        onto an unchanged stream is harmless and idempotent); only the non-RNG
        swap needs reverting. At the ``rng_restore`` stage both domains may have
        partial effects and both are reverted.
        """
        rng_error: Exception | None = None
        nonrng_error: Exception | None = None
        # Restore the RNG domain (it was the last thing mutated, if at all).
        if self._previous_rng is not None:
            try:
                self._rng_consumer(self._previous_rng)
            except Exception as e:  # pragma: no cover - defensive
                rng_error = e
        # Revert the non-RNG swap if it was committed — independently.
        if self._non_rng_committed:
            try:
                self._factory.revert_commit_to_live(self._live)
                self._non_rng_committed = False
            except Exception as e:  # pragma: no cover - defensive
                nonrng_error = e
        # Record diagnostics for observation; they do NOT replace the original.
        diags: list[str] = []
        if rng_error is not None:
            diags.append(f"rng rollback failed: {rng_error}")
        if nonrng_error is not None:
            diags.append(f"non-rng revert failed: {nonrng_error}")
        self.cleanup_diagnostics = tuple(diags)

    def _maybe_fail(self, stage: RestoreStage) -> None:
        if self.fail_after == stage:
            raise RestoreError(f"injected failure after stage {stage!r}")

    def _check_open(self) -> None:
        if self._committed:
            raise RestoreError("transaction already committed")
        if self._aborted:
            raise RestoreError("transaction already aborted")
