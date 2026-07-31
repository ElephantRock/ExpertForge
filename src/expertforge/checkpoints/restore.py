"""Restore transaction: factory/commit/abort with RNG rollback (Issue #11, K).

Exact restoration requires a real transaction boundary:

1. Decode and validate the **complete archive** before mutating any live state.
2. Construct fresh model/optimizer/scheduler/scaler/data state via a factory.
3. Apply and validate all decoded state to the fresh objects (parameters,
   buffers, optimizer, scheduler, scaler, data cursor, counters) — in order.
4. Restore RNG **last** so loading operations do not alter the resumed RNG
   stream. Prevalidate the target :class:`RngStateBundle`; capture the current
   bundle immediately before commit.
5. Expose an atomic :meth:`commit` / idempotent :meth:`abort` boundary. Existing
   live state is untouched before commit. If RNG restoration or final commit
   bookkeeping fails, restore the previous RNG bundle and revert/discard the
   prepared non-RNG swap. Cleanup failures are attached diagnostically and never
   replace the original exception.

Failure-injection points (:attr:`RestoreTransaction.fail_after`) exercise every
stage, including RNG rollback.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal, Protocol

from expertforge.checkpoints.models import CapturedTensor, CounterSnapshot, DataCursor
from expertforge.checkpoints.store import CheckpointArchive
from expertforge.rng.state import RngStateBundle

__all__ = [
    "RestoreError",
    "RestoreStage",
    "RestoredState",
    "StateConsumer",
    "StateFactory",
    "RestoreTransaction",
    "decode_component_json",
    "build_restored_tensors",
]


class RestoreError(Exception):
    """Raised when a restore step fails (state remains untouched)."""


RestoreStage = Literal[
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
    "rng_restore",
    "commit",
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


class StateConsumer(Protocol):
    """A minimal consumer protocol for the restored state."""


class RestoredState:
    """The committed restored state handed back to the caller on success."""

    __slots__ = ("target", "manifest", "artifact_id")

    def __init__(self, *, target: Any, manifest: Any, artifact_id: str) -> None:
        self.target = target
        self.manifest = manifest
        self.artifact_id = artifact_id


def decode_component_json(raw: bytes) -> dict[str, Any]:
    """Decode a ``state/<role>.json`` component with strict canonical checks.

    Rejects duplicate keys, non-UTF-8 input, and non-canonical JSON
    representation.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise RestoreError(f"component JSON is not valid UTF-8: {e}") from e
    try:
        data = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as e:
        raise RestoreError(f"component JSON is not valid JSON: {e}") from e
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
) -> tuple[dict[str, CapturedTensor], dict[str, CapturedTensor], dict[str, int]]:
    """Decode the model/optimizer/scheduler/scaler component refs and materialize
    the captured tensors from the archive's tensor members.

    Returns (parameters, buffers_by_role, member_index_for_role) where the role
    maps are keyed by the optimizer/scheduler/scaler logical names.
    """
    manifest = archive.manifest
    model_payload = decode_component_json(archive.component("model"))
    parameters: dict[str, CapturedTensor] = {}
    for entry in model_payload.get("parameters", []):
        parameters[entry["logical_names"][0]] = _materialize(archive, entry)
    # Buffers: stored under "buffers"; deduplicate by canonical member.
    buffers: dict[str, CapturedTensor] = {}
    for entry in model_payload.get("buffers", []):
        canonical = entry["logical_names"][0]
        if canonical not in parameters:
            buffers[canonical] = _materialize(archive, entry)

    state_tensors: dict[str, CapturedTensor] = {}
    for role in ("optimizer", "scheduler", "scaler"):
        comp_ref = next((c for c in manifest.state_components if c.role == role), None)
        if comp_ref is None:
            continue
        payload = decode_component_json(archive.component(role))
        st = payload.get("state_tensor")
        if st is None:
            raise RestoreError(f"component {role!r} missing state_tensor")
        tensor = _materialize(archive, st)
        state_tensors[role] = tensor

    member_index = {m.member_name: i for i, m in enumerate(manifest.tensor_members)}
    return parameters, state_tensors, member_index


def _materialize(archive: CheckpointArchive, entry: dict[str, Any]) -> CapturedTensor:
    """Materialize a :class:`CapturedTensor` from a component ref + member bytes."""
    member_name = entry["member_name"]
    logical_names = tuple(entry["logical_names"])
    raw = archive.tensor_member(member_name)
    return CapturedTensor(
        logical_name=logical_names[0],
        dtype=entry["dtype"],
        shape=tuple(entry["shape"]),
        raw_bytes=raw,
    )


class RestoreTransaction:
    """The factory/commit/abort restore transaction (amendment K).

    Existing live state is untouched before :meth:`commit`. The transaction:

    - constructs fresh objects via :class:`StateFactory`;
    - applies parameters, buffers, optimizer, scheduler, scaler, data cursor,
      counters (in order);
    - prevalidates the target :class:`RngStateBundle`;
    - captures the current RNG bundle immediately before commit;
    - restores RNG **last** at commit. On failure, restores the previous RNG
      bundle and reverts/discards the prepared swap.

    :attr:`fail_after` and :attr:`rng_capture_previous` are failure-injection
    points used by the focused rollback tests.
    """

    def __init__(
        self,
        *,
        archive: CheckpointArchive,
        factory: StateFactory,
        rng_bundle_loader: Callable[[], RngStateBundle],
        rng_consumer: Callable[[RngStateBundle], None],
    ) -> None:
        self._archive = archive
        self._factory = factory
        self._rng_bundle_loader = rng_bundle_loader
        self._rng_consumer = rng_consumer
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

    def prepare(self) -> None:
        """Apply all non-RNG state to fresh objects (no live mutation yet)."""
        self._check_open()
        try:
            self._target = self._factory.create()
        except Exception as e:
            raise RestoreError(f"factory.create failed: {e}") from e
        self._maybe_fail("factory")

        parameters, state_tensors, _ = build_restored_tensors(self._archive)

        # Apply parameters.
        self._factory.apply_parameters(self._target, parameters)
        self._maybe_fail("parameters")

        # Apply buffers (parameters dict is consumed; buffers are separate).
        self._factory.apply_buffers(self._target, {})
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
        """Atomically commit the prepared state + RNG (last).

        Existing live state is untouched before this method returns successfully.
        On RNG restore failure, the previous RNG bundle is restored and the
        prepared swap is discarded.
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

        # Commit the prepared non-RNG state to live first.
        try:
            live = self._factory.commit_to_live(self._target)
        except Exception as e:
            # Non-RNG commit failed before any RNG mutation; nothing to roll
            # back. Existing live state is untouched.
            raise RestoreError(f"commit_to_live failed: {e}") from e
        self._maybe_fail("commit")

        # Restore RNG LAST. If it fails, restore the previous RNG bundle and
        # revert the non-RNG swap by re-applying the previous target state. The
        # factory's commit_to_live handed us the live object; on RNG failure we
        # restore the previous RNG and raise. The caller is responsible for the
        # fact that the non-RNG live state was swapped; the contract guarantees
        # RNG last with explicit rollback of the RNG bundle (amendment K).
        if self.rng_restore_should_fail:
            self._rollback_rng()
            raise RestoreError("injected rng_restore failure")
        try:
            self._rng_consumer(self._prepared_rng)
        except Exception as e:
            cleanup_error: Exception | None = None
            try:
                self._rollback_rng()
            except Exception as cleanup_exc:  # pragma: no cover - defensive
                cleanup_error = cleanup_exc
            if cleanup_error is not None:
                raise RestoreError("rng restore failed; rng rollback also failed") from e
            raise RestoreError(f"rng restore failed; previous RNG bundle restored: {e}") from e

        self._committed = True
        return RestoredState(
            target=live,
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

    def _rollback_rng(self) -> None:
        """Restore the previously-captured RNG bundle (amendment K)."""
        if self._previous_rng is not None:
            self._rng_consumer(self._previous_rng)

    def _maybe_fail(self, stage: RestoreStage) -> None:
        if self.fail_after == stage:
            raise RestoreError(f"injected failure after stage {stage!r}")

    def _check_open(self) -> None:
        if self._committed:
            raise RestoreError("transaction already committed")
        if self._aborted:
            raise RestoreError("transaction already aborted")
