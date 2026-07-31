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
from typing import Any, Literal, Protocol

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
    "SchedulerState",
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
    dict[str, CapturedTensor], dict[str, CapturedTensor], dict[str, CapturedTensor], dict[str, int]
]:
    """Decode the model/optimizer/scheduler/scaler component refs and materialize
    the captured tensors from the archive's tensor members.

    Returns ``(parameters, buffers, state_tensors_by_role, member_index)``.

    Every component tensor reference is strict-schema-validated (item 12) and
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
        # Non-strict: JSON lists coerce to the frozen tuple fields. The frozen
        # extra-forbid model still rejects unknown keys and wrong types.
        model_state = ModelState.model_validate(model_payload, strict=False)
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
    for role in ("optimizer", "scheduler", "scaler"):
        comp_ref = next((c for c in manifest.state_components if c.role == role), None)
        if comp_ref is None:
            continue
        payload = decode_component_json(archive.component(role))
        try:
            if role == "optimizer":
                parsed: OptimizerState | SchedulerState | ScalerState = (
                    OptimizerState.model_validate(payload, strict=False)
                )
            elif role == "scheduler":
                parsed = SchedulerState.model_validate(payload, strict=False)
            else:
                parsed = ScalerState.model_validate(payload, strict=False)
        except ValidationError as e:
            raise RestoreError(f"{role} component failed strict validation: {e}") from e
        # A stateless component (has_state=False) carries no tensor.
        if not parsed.has_state:
            continue
        ref = parsed.state_tensor
        if ref is None:
            raise RestoreError(f"component {role!r} is missing state_tensor")
        tensor = _materialize(archive, ref, members_by_name)
        state_tensors[role] = tensor

    member_index = {m.member_name: i for i, m in enumerate(manifest.tensor_members)}
    return parameters, buffers, state_tensors, member_index


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
    raw = archive.tensor_member(entry.member_name)
    return CapturedTensor(
        logical_name=entry.logical_names[0],
        dtype=entry.dtype,
        shape=tuple(entry.shape),
        raw_bytes=raw,
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
        expected_descriptor: CompatibilityDescriptor | None = None,
        compatibility_checker: Callable[
            [CheckpointArchive, CompatibilityDescriptor], CompatibilityResult
        ]
        | None = None,
    ) -> None:
        self._archive = archive
        self._factory = factory
        self._rng_bundle_loader = rng_bundle_loader
        self._rng_consumer = rng_consumer
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

        Runs the mandatory compatibility gate first (item 8) when an expected
        descriptor was supplied to the constructor.
        """
        self._check_open()
        if self._expected_descriptor is not None:
            self.require_exact_compatibility(self._expected_descriptor)
        try:
            self._target = self._factory.create()
        except Exception as e:
            raise RestoreError(f"factory.create failed: {e}") from e
        self._maybe_fail("factory")

        parameters, buffers, state_tensors, _ = build_restored_tensors(self._archive)

        # Apply parameters.
        self._factory.apply_parameters(self._target, parameters)
        self._maybe_fail("parameters")

        # Apply buffers (item 2): materialized buffers are now restored, not
        # dropped.
        self._factory.apply_buffers(self._target, buffers)
        self._maybe_fail("buffers")

        # Apply optimizer state.
        self._factory.apply_optimizer(self._target, state_tensors.get("optimizer"))
        self._maybe_fail("optimizer")

        # Apply scheduler state.
        self._factory.apply_scheduler(self._target, state_tensors.get("scheduler"))
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
        """Atomically commit the prepared state + RNG (two-phase, item 1).

        Phase 1: commit the non-RNG swap to live (``commit_to_live``).
        Phase 2: restore RNG LAST. If phase 2 fails, the previous RNG bundle is
        restored AND the phase-1 non-RNG swap is reverted via
        :meth:`StateFactory.revert_commit_to_live`, so the pre-restore live
        state is authoritative. Existing live state is untouched before phase 1.
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
        self._maybe_fail("commit_non_rng")

        # Phase 2: restore RNG LAST. If it fails, restore the previous RNG
        # bundle AND revert the phase-1 non-RNG swap so pre-restore live state
        # is authoritative (item 1).
        if self.rng_restore_should_fail:
            self._rollback_phase2()
            raise RestoreError("injected rng_restore failure")
        try:
            self._rng_consumer(self._prepared_rng)
        except Exception as e:
            cleanup_error: Exception | None = None
            try:
                self._rollback_phase2()
            except Exception as cleanup_exc:  # pragma: no cover - defensive
                cleanup_error = cleanup_exc
            if cleanup_error is not None:
                raise RestoreError(
                    "rng restore failed; rng rollback AND non-RNG revert failed"
                ) from e
            raise RestoreError(
                f"rng restore failed; previous RNG bundle restored and non-RNG swap reverted: {e}"
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

    def _rollback_phase2(self) -> None:
        """Revert the phase-2 effects after RNG failure (item 1).

        Restores the previously-captured RNG bundle (amendment K) AND reverts
        the phase-1 non-RNG swap so pre-restore live state is authoritative.
        """
        # Restore RNG first (it was the last thing mutated).
        if self._previous_rng is not None:
            self._rng_consumer(self._previous_rng)
        # Then revert the non-RNG swap if it was committed.
        if self._non_rng_committed:
            self._factory.revert_commit_to_live(self._live)
            self._non_rng_committed = False

    def _maybe_fail(self, stage: RestoreStage) -> None:
        if self.fail_after == stage:
            raise RestoreError(f"injected failure after stage {stage!r}")

    def _check_open(self) -> None:
        if self._committed:
            raise RestoreError("transaction already committed")
        if self._aborted:
            raise RestoreError("transaction already aborted")
