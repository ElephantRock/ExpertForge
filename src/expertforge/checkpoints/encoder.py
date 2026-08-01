"""Archive encoder: build manifest + members from a captured snapshot (Issue #11).

Bridges :class:`CapturedCheckpointState` (amendment H), the frozen manifest
models, and the deterministic ustar tar writer. The encoder:

- assigns canonical tensor member indices/paths/sizes/digests (amendment E);
- deduplicates alias-group tensors (one payload per group);
- builds the exact member graph (amendment B): ``manifest.json`` first, then
  ``state/<role>.json`` components in fixed order, then ``tensors/<idx>.bin``;
- authenticates every non-manifest member via the manifest (role/path/size/sha256).

No ``metadata.json``. No self-referential checkpoint id (amendment A).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from expertforge.artifacts.models import ParentReference
from expertforge.checkpoints.models import (
    CHECKPOINT_ARCHIVE_FORMAT_VERSION,
    MAX_COMPONENT_MEMBER_BYTES,
    MAX_TENSOR_COUNT,
    MAX_TENSOR_MEMBER_BYTES,
    CapturedCheckpointState,
    CapturedTensor,
    CheckpointManifest,
    CompatibilityDescriptor,
    StateComponentRef,
    TensorMemberRef,
    canonical_json_bytes,
)
from expertforge.checkpoints.tar_writer import TarMember, build_ustar_archive
from expertforge.config.resolve import ResolutionEnvelope, canonical_bytes
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.provenance.record import ProvenanceRecord
from expertforge.rng.state import RngStateBundle

__all__ = [
    "EncoderError",
    "ArchiveMembers",
    "ArchiveParts",
    "build_archive",
    "build_archive_parts",
    "encode_safe_value",
]

# Fixed canonical component order (amendment B role order).
_COMPONENT_ORDER: tuple[str, ...] = (
    "identity",
    "configuration",
    "provenance",
    "rng",
    "data_cursor",
    "counters",
    "model",
    "optimizer",
    "scheduler",
    "scaler",
)


class EncoderError(Exception):
    """Raised when a captured snapshot cannot be encoded into an archive."""


def _resolve_rng_bytes(
    captured: CapturedCheckpointState, rng_bundle: RngStateBundle | None
) -> bytes:
    """Resolve the RNG member bytes from the atomic captured snapshot (item 4).

    The snapshot's own ``captured.rng_bundle_bytes`` are authoritative: they were
    captured atomically alongside the rest of the state. A separately supplied
    ``rng_bundle``, if any, MUST decode to the exact same bytes; any
    disagreement raises :class:`EncoderError` rather than silently persisting a
    bundle from a different instant (which would combine model/cursor/counters
    from one instant with RNG from another and break deterministic continuation).
    """
    captured_bytes = captured.rng_bundle_bytes
    if rng_bundle is not None:
        supplied_bytes = rng_bundle.to_deterministic_json()
        if supplied_bytes != captured_bytes:
            raise EncoderError(
                "supplied rng_bundle disagrees with captured.rng_bundle_bytes; "
                "the atomic snapshot is authoritative"
            )
    return captured_bytes


def encode_safe_value(value: Any) -> Any:
    """Best-effort SafeValue JSON encoder for option-dict scalars.

    Maps native Python scalars to the canonical SafeValue ``kind`` tag. Used to
    encode optimizer option values; nested mappings/sequences recurse.
    """
    from expertforge.checkpoints.models import (
        _SafeBool,
        _SafeBytes,
        _SafeFloat,
        _SafeInt,
        _SafeMapping,
        _SafeNull,
        _SafeSequence,
        _SafeString,
    )

    if value is None:
        return _SafeNull().model_dump(mode="json")
    if isinstance(value, bool):
        return _SafeBool(value=value).model_dump(mode="json")
    if isinstance(value, int):
        return _SafeInt(width_bits=64, signed=True, value=value).model_dump(mode="json")
    if isinstance(value, float):
        packed = struct_pack_float(value)
        return _SafeFloat(width_bits=64, bit_pattern=packed).model_dump(mode="json")
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        return _SafeBytes(
            value=__import__("base64").b64encode(raw).decode("ascii"),
            byte_length=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        ).model_dump(mode="json")
    if isinstance(value, str):
        return _SafeString(value=value).model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        seq = _SafeSequence(value=tuple(_safe_value_obj(v) for v in value))
        return seq.model_dump(mode="json")
    if isinstance(value, dict):
        items = sorted(value.items())
        mp = _SafeMapping(value=tuple((k, _safe_value_obj(v)) for k, v in items))
        return mp.model_dump(mode="json")
    raise EncoderError(f"cannot encode value of type {type(value).__name__!r}")


def _safe_value_obj(value: Any) -> Any:
    """Build a SafeValue-typed model object (not the json dump)."""
    from expertforge.checkpoints.models import (
        _SafeBool,
        _SafeBytes,
        _SafeFloat,
        _SafeInt,
        _SafeMapping,
        _SafeNull,
        _SafeSequence,
        _SafeString,
    )

    if value is None:
        return _SafeNull()
    if isinstance(value, bool):
        return _SafeBool(value=value)
    if isinstance(value, int):
        return _SafeInt(width_bits=64, signed=True, value=value)
    if isinstance(value, float):
        return _SafeFloat(width_bits=64, bit_pattern=struct_pack_float(value))
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        return _SafeBytes(
            value=__import__("base64").b64encode(raw).decode("ascii"),
            byte_length=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    if isinstance(value, str):
        return _SafeString(value=value)
    if isinstance(value, (list, tuple)):
        return _SafeSequence(value=tuple(_safe_value_obj(v) for v in value))
    if isinstance(value, dict):
        items = sorted(value.items())
        return _SafeMapping(value=tuple((k, _safe_value_obj(v)) for k, v in items))
    raise EncoderError(f"cannot encode value of type {type(value).__name__!r}")


def struct_pack_float(value: float) -> int:
    """Return the exact 64-bit IEEE-754 pattern of ``value`` (NaN/Inf/-0 exact)."""
    import struct

    return int.from_bytes(struct.pack(">d", value), "big", signed=False)


class ArchiveMembers:
    """The fully-built archive: manifest + ordered members + raw tar bytes."""

    __slots__ = ("manifest", "members", "tar_bytes", "content_digest", "byte_size")

    def __init__(
        self,
        *,
        manifest: CheckpointManifest,
        members: list[TarMember],
        tar_bytes: bytes,
    ) -> None:
        self.manifest = manifest
        self.members = members
        self.tar_bytes = tar_bytes
        h = hashlib.sha256(tar_bytes)
        self.content_digest = f"sha256:{h.hexdigest()}"
        self.byte_size = len(tar_bytes)


class ArchiveParts:
    """The ordered manifest + members WITHOUT the materialized tar bytes (item 8).

    Used by the streaming save path (:meth:`CheckpointStore.save`): the manifest
    and the ordered :class:`TarMember` list are computed once, then the archive
    is streamed to a temp file via :func:`stream_ustar_archive` while the SHA-256
    digest is updated incrementally. The complete tar is never held in memory.

    ``content_digest`` / ``byte_size`` are filled in by the streaming writer as
    it emits the bytes; the in-memory :class:`ArchiveMembers` form remains
    available via :func:`build_archive` for tests and the small/deterministic
    path.
    """

    __slots__ = ("manifest", "members", "content_digest", "byte_size")

    def __init__(
        self,
        *,
        manifest: CheckpointManifest,
        members: list[TarMember],
    ) -> None:
        self.manifest = manifest
        self.members = members
        self.content_digest = ""
        self.byte_size = 0


def _component_path(role: str) -> str:
    return f"state/{role}.json"


def _component_ref(role: str, payload: bytes) -> StateComponentRef:
    return StateComponentRef(
        role=role,  # type: ignore[arg-type]
        member_name=_component_path(role),
        member_sha256=hashlib.sha256(payload).hexdigest(),
        member_byte_size=len(payload),
    )


def _resolve_tensors(
    captured: CapturedCheckpointState,
) -> tuple[dict[str, CapturedTensor], dict[str, tuple[str, ...]]]:
    """Return (canonical_tensors_by_storage_name, name_to_member_name).

    Alias groups collapse to a single canonical storage tensor. Every logical
    name in a group binds to the same member.
    """
    all_tensors: dict[str, CapturedTensor] = {}
    for t in (*captured.parameters, *captured.buffers):
        all_tensors[t.logical_name] = t
    # Also include optimizer/scheduler/scaler state tensors under their names.
    state_tensors: tuple[CapturedTensor | None, ...] = (
        captured.optimizer,
        captured.scheduler,
        captured.scaler,
    )
    for state_tensor in state_tensors:
        if state_tensor is not None:
            if state_tensor.logical_name in all_tensors:
                raise EncoderError(f"tensor logical name collision: {state_tensor.logical_name!r}")
            all_tensors[state_tensor.logical_name] = state_tensor
    # Item 13: include multi-slot optimizer state tensors under their names.
    for slot_tensor in captured.optimizer_slots:
        if slot_tensor.logical_name in all_tensors:
            raise EncoderError(f"tensor logical name collision: {slot_tensor.logical_name!r}")
        all_tensors[slot_tensor.logical_name] = slot_tensor

    name_to_member: dict[str, tuple[str, ...]] = {}
    canonical_tensors: dict[str, CapturedTensor] = {}
    consumed_names: set[str] = set()

    # Process alias groups first: one canonical member per group.
    for group in captured.alias_groups:
        members = tuple(sorted({*group}))
        canonical_name = members[0]
        if canonical_name not in all_tensors:
            raise EncoderError(f"alias group canonical member {canonical_name!r} not captured")
        canonical_tensors[canonical_name] = all_tensors[canonical_name]
        for n in members:
            name_to_member[n] = members
            consumed_names.add(n)

    # Non-aliased tensors: each gets its own canonical member.
    for name, tensor in all_tensors.items():
        if name in consumed_names:
            continue
        canonical_tensors[name] = tensor
        name_to_member[name] = (name,)
        consumed_names.add(name)

    return canonical_tensors, name_to_member


def build_archive_parts(
    *,
    identity: AttemptIdentityRecord,
    captured: CapturedCheckpointState,
    configuration_envelope: ResolutionEnvelope,
    provenance: ProvenanceRecord,
    rng_bundle: RngStateBundle | None = None,
    parent: ParentReference | None = None,
    created_at_utc: datetime,
) -> ArchiveParts:
    """Compute the manifest + ordered members WITHOUT materializing the tar.

    This is the streaming-friendly core of :func:`build_archive` (item 8): every
    byte that ends up in the archive is determined here (manifest, component
    payloads, tensor payloads), but the complete ustar byte stream is NOT
    assembled. :meth:`CheckpointStore.save` streams these parts directly to a
    temp file via :func:`stream_ustar_archive` while hashing incrementally, so
    the whole archive is never held in Python heap.

    Returns an :class:`ArchiveParts` whose ``content_digest`` / ``byte_size`` are
    left empty (filled in by the streaming writer) — the in-memory digest/size
    are available via :func:`build_archive`.
    """
    config_bytes = canonical_bytes(configuration_envelope)
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    config_fingerprint = identity.fingerprint_digest_str()

    # Item 4: the RNG member is the snapshot's own captured bytes. A separately
    # supplied bundle, if any, must agree byte-for-byte; reject disagreement
    # rather than silently persisting a bundle from a different instant.
    rng_bytes = _resolve_rng_bytes(captured, rng_bundle)

    # --- Component payloads (canonical JSON for the SafeState domain). -------
    identity_bytes = identity.to_deterministic_json()
    configuration_payload = canonical_json_bytes(
        {
            "schema": "expertforge.checkpoint-configuration",
            "version": 1,
            "content_sha256": config_digest,
            "content_base64": __import__("base64").b64encode(config_bytes).decode("ascii"),
            "byte_length": len(config_bytes),
            "fingerprint": config_fingerprint,
        }
    )
    provenance_bytes = provenance.to_deterministic_json()
    data_cursor_bytes = canonical_json_bytes(captured.data_cursor.model_dump(mode="json"))
    counters_bytes = canonical_json_bytes(captured.counters.model_dump(mode="json"))
    # Model/optimizer/scheduler/scaler component payloads encode the captured
    # tensor's logical-name -> member binding plus the SafeState references.
    canonical_tensors, name_to_member = _resolve_tensors(captured)

    # Assign canonical tensor member indices (sorted by canonical storage name).
    canonical_names_sorted = sorted(canonical_tensors.keys())
    if len(canonical_names_sorted) > MAX_TENSOR_COUNT:
        raise EncoderError(f"too many tensors ({len(canonical_names_sorted)})")
    canonical_to_index: dict[str, int] = {name: i for i, name in enumerate(canonical_names_sorted)}

    tensor_members: list[TensorMemberRef] = []
    tensors_by_index: dict[int, tuple[CapturedTensor, tuple[str, ...]]] = {}
    for name in canonical_names_sorted:
        idx = canonical_to_index[name]
        tensor = canonical_tensors[name]
        member_name = f"tensors/{idx}.bin"
        if len(tensor.raw_bytes) > MAX_TENSOR_MEMBER_BYTES:
            raise EncoderError(f"tensor member {name!r} exceeds {MAX_TENSOR_MEMBER_BYTES} bytes")
        members_for_storage = name_to_member[name]
        tensor_members.append(
            TensorMemberRef(
                member_name=member_name,
                member_sha256=tensor.member_sha256(),
                member_byte_size=tensor.member_byte_size(),
                dtype=tensor.dtype,
                shape=tensor.shape,
                logical_names=members_for_storage,
            )
        )
        tensors_by_index[idx] = (tensor, members_for_storage)

    # Model/optimizer/scheduler/scaler JSON: name -> member reference.
    def _tensor_ref_payload(name: str) -> dict[str, Any]:
        storage_members = name_to_member[name]
        canonical_name = storage_members[0]
        idx = canonical_to_index[canonical_name]
        return {
            "member_name": f"tensors/{idx}.bin",
            "logical_names": list(storage_members),
            "dtype": all_tensors_dtype(name, captured),
            "shape": all_tensors_shape(name, captured),
        }

    model_payload = canonical_json_bytes(
        {
            "schema": "expertforge.checkpoint-model-state",
            "version": 1,
            "parameters": [_tensor_ref_payload(t.logical_name) for t in captured.parameters],
            "buffers": [_tensor_ref_payload(t.logical_name) for t in captured.buffers],
        }
    )

    # Item 5: optimizer/scheduler/scaler members are ALWAYS emitted as valid
    # canonical JSON carrying the component's descriptor, even when stateless
    # (no tensors). A ``has_state`` flag distinguishes stateless components from
    # those carrying tensor/scalar state. This prevents the previously-emitted
    # empty ``b""`` member from breaking restoration before apply_* can run.
    opt_has_state = (
        captured.optimizer is not None
        or bool(captured.optimizer_slots)
        or captured.optimizer_scalar_state is not None
    )
    optimizer_obj: dict[str, Any] = {
        "schema": "expertforge.checkpoint-optimizer-state",
        "version": 1,
        "descriptor": captured.optimizer_descriptor.model_dump(mode="json"),
        "has_state": opt_has_state,
    }
    if captured.optimizer is not None:
        optimizer_obj["state_tensor"] = _tensor_ref_payload(captured.optimizer.logical_name)
    if captured.optimizer_slots:
        optimizer_obj["state_slots"] = [
            _tensor_ref_payload(t.logical_name) for t in captured.optimizer_slots
        ]
    if captured.optimizer_scalar_state is not None:
        optimizer_obj["scalar_state"] = captured.optimizer_scalar_state.model_dump(mode="json")
    optimizer_payload = canonical_json_bytes(optimizer_obj)

    sched_has_state = captured.scheduler is not None or captured.scheduler_scalar_state is not None
    scheduler_obj: dict[str, Any] = {
        "schema": "expertforge.checkpoint-scheduler-state",
        "version": 1,
        "descriptor": captured.scheduler_descriptor.model_dump(mode="json"),
        "has_state": sched_has_state,
    }
    if captured.scheduler is not None:
        scheduler_obj["state_tensor"] = _tensor_ref_payload(captured.scheduler.logical_name)
    if captured.scheduler_scalar_state is not None:
        scheduler_obj["scalar_state"] = captured.scheduler_scalar_state.model_dump(mode="json")
    scheduler_payload = canonical_json_bytes(scheduler_obj)

    scaler_payload = b""
    if captured.scaler_descriptor is not None:
        scaler_obj: dict[str, Any] = {
            "schema": "expertforge.checkpoint-scaler-state",
            "version": 1,
            "descriptor": captured.scaler_descriptor.model_dump(mode="json"),
            "has_state": captured.scaler is not None,
        }
        if captured.scaler is not None:
            scaler_obj["state_tensor"] = _tensor_ref_payload(captured.scaler.logical_name)
        scaler_payload = canonical_json_bytes(scaler_obj)

    # Validate component member size bounds.
    component_payloads: list[tuple[str, bytes]] = [
        ("identity", identity_bytes),
        ("configuration", configuration_payload),
        ("provenance", provenance_bytes),
        ("rng", rng_bytes),
        ("data_cursor", data_cursor_bytes),
        ("counters", counters_bytes),
        ("model", model_payload),
        ("optimizer", optimizer_payload),
        ("scheduler", scheduler_payload),
    ]
    if scaler_payload:
        component_payloads.append(("scaler", scaler_payload))

    for role, payload in component_payloads:
        if len(payload) > MAX_COMPONENT_MEMBER_BYTES:
            raise EncoderError(
                f"component member state/{role}.json exceeds {MAX_COMPONENT_MEMBER_BYTES} bytes"
            )

    # --- Build the manifest --------------------------------------------------
    compatibility = _build_compatibility_descriptor(
        identity=identity,
        captured=captured,
        config_fingerprint=config_fingerprint,
    )

    state_components: list[StateComponentRef] = [
        _component_ref(role, payload) for role, payload in component_payloads
    ]

    manifest = CheckpointManifest(
        manifest_schema="expertforge.checkpoint-manifest",
        manifest_schema_version=1,
        archive_format_version=CHECKPOINT_ARCHIVE_FORMAT_VERSION,
        state_format_version=1,
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        specification_fingerprint=config_fingerprint,
        parent_artifact_id=parent.artifact_id if parent else None,
        parent_run_id=parent.run_id if parent else None,
        parent_attempt_id=parent.attempt_id if parent else None,
        created_at_utc=created_at_utc.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        counters=captured.counters,
        configuration_fingerprint=config_fingerprint,
        configuration_content_sha256=config_digest,
        compatibility=compatibility,
        state_components=tuple(state_components),
        tensor_members=tuple(tensor_members),
    )

    # --- Assemble members in the exact canonical order (amendment B). --------
    manifest_bytes = canonical_json_bytes(json.loads(manifest.model_dump_json()))
    if len(manifest_bytes) > 16 * 1024 * 1024:
        raise EncoderError("manifest exceeds 16 MiB")

    members: list[TarMember] = [TarMember("manifest.json", manifest_bytes)]
    for role, payload in component_payloads:
        members.append(TarMember(_component_path(role), payload))
    for idx in range(len(canonical_names_sorted)):
        tensor, _ = tensors_by_index[idx]
        members.append(TarMember(f"tensors/{idx}.bin", tensor.raw_bytes))

    return ArchiveParts(manifest=manifest, members=members)


def build_archive(
    *,
    identity: AttemptIdentityRecord,
    captured: CapturedCheckpointState,
    configuration_envelope: ResolutionEnvelope,
    provenance: ProvenanceRecord,
    rng_bundle: RngStateBundle | None = None,
    parent: ParentReference | None = None,
    created_at_utc: datetime,
) -> ArchiveMembers:
    """Build the full checkpoint archive from a quiescent captured snapshot.

    The archive manifest authenticates every non-manifest member. The #10
    content digest (computed over the full tar byte stream) authenticates the
    manifest. No self-referential checkpoint id is stored (amendment A).

    The RNG member is serialized from the snapshot's own
    ``captured.rng_bundle_bytes`` (item 4): the bytes captured atomically by
    :meth:`StateProvider.capture_checkpoint_snapshot`. If a ``rng_bundle`` is
    additionally supplied, it MUST decode to the exact same bytes; any
    disagreement raises :class:`EncoderError` rather than silently persisting a
    bundle from a different instant.

    This materializes the complete tar in memory; for the streaming save path
    (Issue #11 item 8) use :func:`build_archive_parts` +
    :func:`stream_ustar_archive`.
    """
    parts = build_archive_parts(
        identity=identity,
        captured=captured,
        configuration_envelope=configuration_envelope,
        provenance=provenance,
        rng_bundle=rng_bundle,
        parent=parent,
        created_at_utc=created_at_utc,
    )
    tar_bytes = build_ustar_archive(parts.members)
    if len(tar_bytes) > 64 * 1024 * 1024 * 1024:
        raise EncoderError("archive exceeds 64 GiB")
    return ArchiveMembers(manifest=parts.manifest, members=parts.members, tar_bytes=tar_bytes)


def all_tensors_dtype(name: str, captured: CapturedCheckpointState) -> str:
    """Resolve the dtype of any captured tensor by logical name (item 3).

    Searches parameters, buffers, the legacy optimizer/scheduler/scaler tensors,
    AND the multi-slot ``optimizer_slots`` (the previously-missed case).
    """
    for t in (
        *captured.parameters,
        *captured.buffers,
        captured.optimizer,
        captured.scheduler,
        captured.scaler,
        *captured.optimizer_slots,
    ):
        if t is not None and t.logical_name == name:
            return t.dtype
    raise EncoderError(f"unknown tensor {name!r}")


def all_tensors_shape(name: str, captured: CapturedCheckpointState) -> list[int]:
    """Resolve the shape of any captured tensor by logical name (item 3).

    Searches parameters, buffers, the legacy optimizer/scheduler/scaler tensors,
    AND the multi-slot ``optimizer_slots`` (the previously-missed case).
    """
    for t in (
        *captured.parameters,
        *captured.buffers,
        captured.optimizer,
        captured.scheduler,
        captured.scaler,
        *captured.optimizer_slots,
    ):
        if t is not None and t.logical_name == name:
            return list(t.shape)
    raise EncoderError(f"unknown tensor {name!r}")


def _build_compatibility_descriptor(
    *,
    identity: AttemptIdentityRecord,
    captured: CapturedCheckpointState,
    config_fingerprint: str,
) -> CompatibilityDescriptor:
    return CompatibilityDescriptor(
        specification_fingerprint=config_fingerprint,
        model_descriptor=captured.model_descriptor,
        optimizer_descriptor=captured.optimizer_descriptor,
        scheduler_descriptor=captured.scheduler_descriptor,
        scaler_descriptor=captured.scaler_descriptor,
        rng_descriptor=captured.rng_descriptor,
        data_descriptor=captured.data_descriptor,
        topology_descriptor=captured.topology_descriptor,
    )
