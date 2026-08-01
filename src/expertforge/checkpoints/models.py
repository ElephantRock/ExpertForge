"""Frozen strict checkpoint models (Issue #11).

All persisted/returned models are frozen, ``extra="forbid"``, strict Pydantic v2
records. The safe scalar domain (:class:`SafeValue`) is a closed recursive
tagged union including null and boolean variants (amendment D) with strict
canonical encodings and resource bounds.

Normative references: design ``5145501414`` and amendment ``5145649404``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CHECKPOINT_ARCHIVE_FORMAT_VERSION",
    "CHECKPOINT_MANIFEST_SCHEMA",
    "CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "COMPATIBILITY_SCHEMA",
    "COMPATIBILITY_SCHEMA_VERSION",
    "MAX_ARCHIVE_BYTES",
    "MAX_COMPONENT_MEMBER_BYTES",
    "MAX_MANIFEST_BYTES",
    "MAX_MEMBER_COUNT",
    "MAX_SAFE_DEPTH",
    "MAX_SAFE_MAPPING_ENTRIES",
    "MAX_SAFE_NODES",
    "MAX_SAFE_SEQUENCE_ENTRIES",
    "MAX_SAFE_STRING_BYTES",
    "MAX_SAFE_INLINE_BYTES",
    "MAX_TENSOR_BYTES",
    "MAX_TENSOR_COUNT",
    "MAX_TENSOR_MEMBER_BYTES",
    "SAFE_STATE_FORMAT_VERSION",
    "TensorDtype",
    "TensorByteOrder",
    "TensorLayout",
    "StateComponentRole",
    "SafeValueKind",
    "CompatibilityStatus",
    "CompatibilityComponent",
    "CompatibilityDiagnosticCode",
    "InspectionStatus",
    "SafeValue",
    "CapturedTensor",
    "CapturedCheckpointState",
    "TensorMemberRef",
    "StateComponentRef",
    "ConfigurationState",
    "CounterSnapshot",
    "DataIdentity",
    "DataCursor",
    "ModelParameterDescriptor",
    "AliasGroup",
    "ModelDescriptor",
    "OptimizerParamGroup",
    "OptimizerStateSlot",
    "OptimizerDescriptor",
    "SchedulerDescriptor",
    "ScalerDescriptor",
    "RngDescriptor",
    "DataDescriptor",
    "TopologyDescriptor",
    "CompatibilityDescriptor",
    "CompatibilityMismatch",
    "CompatibilityResult",
    "CheckpointManifest",
    "CheckpointInspection",
    "ResourceLimitError",
    "SafeStateDecodeError",
    "canonical_json_bytes",
    "validate_canonical_digest",
]

# ---------------------------------------------------------------------------
# Independent versioning (design §2)
# ---------------------------------------------------------------------------

# Tar packaging rules (member order, paths, modes, uid/gid, timestamps, padding,
# terminator). Owned here; bumped only on incompatible archive-layout changes.
CHECKPOINT_ARCHIVE_FORMAT_VERSION: int = 1

# The checkpoint manifest schema (manifest.json).
CHECKPOINT_MANIFEST_SCHEMA: Literal["expertforge.checkpoint-manifest"] = (
    "expertforge.checkpoint-manifest"
)
CHECKPOINT_MANIFEST_SCHEMA_VERSION: int = 1

# The recursive safe state-tree encoding domain (state component JSON).
SAFE_STATE_FORMAT_VERSION: int = 1

# The compatibility descriptor schema.
COMPATIBILITY_SCHEMA: Literal["expertforge.checkpoint-compatibility"] = (
    "expertforge.checkpoint-compatibility"
)
COMPATIBILITY_SCHEMA_VERSION: int = 1

# ---------------------------------------------------------------------------
# Resource bounds (amendment D, F) — hard limits before unbounded allocation
# ---------------------------------------------------------------------------

# Safe state-tree recursion bounds.
MAX_SAFE_DEPTH: int = 64
MAX_SAFE_NODES: int = 1_000_000
MAX_SAFE_MAPPING_ENTRIES: int = 100_000
MAX_SAFE_SEQUENCE_ENTRIES: int = 1_000_000
MAX_SAFE_STRING_BYTES: int = 16 * 1024 * 1024  # 16 MiB per string value
MAX_SAFE_INLINE_BYTES: int = 16 * 1024 * 1024  # 16 MiB per inline byte payload

# Tensor limits (design §6; amendment E).
MAX_TENSOR_RANK: int = 8
MAX_TENSOR_DIM: int = 2**31 - 1
MAX_TENSOR_ELEMENTS: int = 2**31 - 1
MAX_TENSOR_COUNT: int = 10_000

# Archive-level bounds (design §7; amendment F).
MAX_MEMBER_COUNT: int = 10_000
MAX_TENSOR_MEMBER_BYTES: int = 1024 * 1024 * 1024  # 1 GiB per tensor member
MAX_COMPONENT_MEMBER_BYTES: int = 256 * 1024 * 1024  # 256 MiB per component member
MAX_MANIFEST_BYTES: int = 16 * 1024 * 1024  # 16 MiB manifest
MAX_TENSOR_BYTES: int = 32 * 1024 * 1024 * 1024  # 32 GiB total tensor bytes
MAX_ARCHIVE_BYTES: int = 64 * 1024 * 1024 * 1024  # 64 GiB total archive

# ---------------------------------------------------------------------------
# Closed domains
# ---------------------------------------------------------------------------

TensorDtype = Literal[
    "float32",
    "float64",
    "float16",
    "bfloat16",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bool",
]

TensorByteOrder = Literal["little"]
TensorLayout = Literal["contiguous"]

# Exact byte widths per dtype for the product(shape) * itemsize invariant.
_DTYPE_ITEMSIZE: dict[str, int] = {
    "float32": 4,
    "float64": 8,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "bool": 1,
}

StateComponentRole = Literal[
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
]

SafeValueKind = Literal[
    "null",
    "bool",
    "int",
    "float",
    "bytes",
    "string",
    "sequence",
    "mapping",
]

CompatibilityStatus = Literal["exact", "inspect_only", "incompatible", "unsupported"]

# Closed compatibility component domain (amendment M).
CompatibilityComponent = Literal[
    "archive_format_version",
    "manifest_schema_version",
    "state_format_version",
    "compatibility_schema_version",
    "specification_fingerprint",
    "model_names",
    "model_shapes",
    "model_dtypes",
    "model_alias_groups",
    "optimizer_type",
    "optimizer_groups",
    "optimizer_options",
    "optimizer_slots",
    "scheduler_identity",
    "scheduler_state_shape",
    "scaler_active",
    "rng_adapter_set",
    "rng_schema",
    "data_identity",
    "data_cursor_contract",
    "topology",
]

# Closed diagnostic code domain. Each mismatch carries exactly one code; the
# status is derived from the closed severity table, not chosen ad hoc.
CompatibilityDiagnosticCode = Literal[
    "version_mismatch",
    "specification_fingerprint_mismatch",
    "parameter_name_missing",
    "parameter_name_unexpected",
    "parameter_shape_mismatch",
    "parameter_dtype_mismatch",
    "alias_group_missing",
    "alias_group_unexpected",
    "alias_group_membership_mismatch",
    "optimizer_type_mismatch",
    "optimizer_group_count_mismatch",
    "optimizer_group_membership_mismatch",
    "optimizer_option_mismatch",
    "optimizer_slot_missing",
    "optimizer_slot_unexpected",
    "optimizer_slot_shape_mismatch",
    "optimizer_slot_dtype_mismatch",
    "scheduler_type_mismatch",
    "scheduler_active_mismatch",
    "scheduler_state_shape_mismatch",
    "scaler_presence_mismatch",
    "rng_adapter_missing",
    "rng_adapter_unexpected",
    "rng_schema_mismatch",
    "data_identity_mismatch",
    "data_cursor_contract_mismatch",
    "topology_world_size_mismatch",
    "topology_rank_assignment_mismatch",
    "unknown_schema",
    "unknown_version",
]

InspectionStatus = Literal["complete", "corrupt", "incomplete", "incompatible", "unsupported"]

# Components whose mismatch ALWAYS blocks restoration (severity blocking).
# amendment M: any mismatch in an active required component blocks restoration;
# scaler presence is therefore NOT a noncritical mismatch.
_BLOCKING_COMPONENTS: frozenset[str] = frozenset(
    {
        "archive_format_version",
        "manifest_schema_version",
        "state_format_version",
        "compatibility_schema_version",
        "specification_fingerprint",
        "model_names",
        "model_shapes",
        "model_dtypes",
        "model_alias_groups",
        "optimizer_type",
        "optimizer_groups",
        "optimizer_options",
        "optimizer_slots",
        "scheduler_identity",
        "scheduler_state_shape",
        "scaler_active",
        "rng_adapter_set",
        "rng_schema",
        "data_identity",
        "data_cursor_contract",
        "topology",
    }
)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_TENSOR_MEMBER_NAME = re.compile(r"^tensors/[0-9]+\.bin$")
_CANONICAL_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-/]*$")


def validate_canonical_digest(value: str) -> str:
    """Raise ValueError unless ``value`` is ``sha256:<64 lowercase hex>``."""
    if not _SHA256_DIGEST.fullmatch(value):
        raise ValueError(f"digest must be 'sha256:<64 lowercase hex>'; got {value!r}.")
    return value


def canonical_json_bytes(obj: Any) -> bytes:
    """Compact, sorted-key, UTF-8, non-finite-prohibiting JSON encoding."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class ResourceLimitError(Exception):
    """Raised when a resource bound (depth, nodes, bytes, entries) is exceeded."""


class SafeStateDecodeError(Exception):
    """Raised on non-canonical / unparseable safe-state JSON."""


# ---------------------------------------------------------------------------
# Frozen model base
# ---------------------------------------------------------------------------


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
        populate_by_name=True,
    )


# ---------------------------------------------------------------------------
# SafeValue — closed recursive tagged union (amendment D)
# ---------------------------------------------------------------------------


def _check_str(value: str) -> str:
    if len(value.encode("utf-8")) > MAX_SAFE_STRING_BYTES:
        raise ResourceLimitError(f"string value exceeds {MAX_SAFE_STRING_BYTES} byte limit")
    return value


def _check_bytes(value: bytes) -> bytes:
    if len(value) > MAX_SAFE_INLINE_BYTES:
        raise ResourceLimitError(f"inline bytes value exceeds {MAX_SAFE_INLINE_BYTES} byte limit")
    return value


class _SafeNull(_FrozenModel):
    kind: Literal["null"] = "null"


class _SafeBool(_FrozenModel):
    kind: Literal["bool"] = "bool"
    value: bool


class _SafeInt(_FrozenModel):
    """An exact-width integer. Signedness and width/range are enforced."""

    kind: Literal["int"] = "int"
    width_bits: int = Field(..., ge=1, le=64)
    signed: bool
    value: int

    @model_validator(mode="after")
    def _check_range(self) -> _SafeInt:
        bits = self.width_bits
        if self.signed:
            lo = -(1 << (bits - 1))
            hi = (1 << (bits - 1)) - 1
        else:
            lo = 0
            hi = (1 << bits) - 1
        if self.value < lo or self.value > hi:
            raise ValueError(
                f"int value {self.value} out of range for "
                f"{'int' if self.signed else 'uint'}{bits} "
                f"([{lo}, {hi}])"
            )
        return self


class _SafeFloat(_FrozenModel):
    """An exact IEEE-754 bit pattern. Stored as the raw 64-bit pattern plus width.

    width=32 → value is the 32-bit pattern widened to 64; bits beyond 32 are zero
    in canonical storage and the canonical 32-bit pattern is reconstructible.
    For canonical storage we persist the exact fixed-width bit pattern as an
    unsigned integer of the matching width (16/32/64).
    """

    kind: Literal["float"] = "float"
    width_bits: int = Field(..., ge=16, le=64)
    # The exact fixed-width bit pattern, stored as an unsigned integer of that
    # width. NaN, ±Inf, and negative zero are all exact bit patterns.
    bit_pattern: int

    @model_validator(mode="after")
    def _check_range(self) -> _SafeFloat:
        bits = self.width_bits
        # Allowed widths for floats: 16, 32, 64.
        if bits not in (16, 32, 64):
            raise ValueError(f"float width_bits must be 16, 32, or 64; got {bits}")
        hi = (1 << bits) - 1
        if self.bit_pattern < 0 or self.bit_pattern > hi:
            raise ValueError(
                f"float bit_pattern {self.bit_pattern} out of range for float{bits} ([0, {hi}])"
            )
        return self


class _SafeBytes(_FrozenModel):
    """A raw byte string. Persisted as canonical padded base64 with digest."""

    kind: Literal["bytes"] = "bytes"
    encoding: Literal["base64"] = "base64"
    value: str  # canonical padded base64
    byte_length: int = Field(..., ge=0)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _check_digest(cls, v: str) -> str:
        if not _HEX64.fullmatch(v):
            raise ValueError(f"bytes sha256 must be 64 lowercase hex; got {v!r}.")
        return v

    @model_validator(mode="after")
    def _verify(self) -> _SafeBytes:
        if self.byte_length > MAX_SAFE_INLINE_BYTES:
            raise ResourceLimitError(f"bytes value exceeds {MAX_SAFE_INLINE_BYTES} byte limit")
        try:
            decoded = base64.b64decode(self.value.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as exc:
            raise ValueError("bytes value must be canonical ASCII base64") from exc
        if len(decoded) != self.byte_length:
            raise ValueError(f"bytes byte_length {self.byte_length} != decoded {len(decoded)}")
        if base64.b64encode(decoded).decode("ascii") != self.value:
            raise ValueError("bytes value must use canonical padded base64")
        if hashlib.sha256(decoded).hexdigest() != self.sha256:
            raise ValueError("bytes sha256 digest mismatch")
        return self


class _SafeString(_FrozenModel):
    kind: Literal["string"] = "string"
    value: str

    @field_validator("value")
    @classmethod
    def _check_len(cls, v: str) -> str:
        if len(v.encode("utf-8")) > MAX_SAFE_STRING_BYTES:
            raise ResourceLimitError(f"string value exceeds {MAX_SAFE_STRING_BYTES} byte limit")
        return v


class _SafeSequence(_FrozenModel):
    kind: Literal["sequence"] = "sequence"
    value: tuple[SafeValue, ...] = Field(default_factory=tuple)

    @field_validator("value")
    @classmethod
    def _check_len(cls, v: tuple[SafeValue, ...]) -> tuple[SafeValue, ...]:
        if len(v) > MAX_SAFE_SEQUENCE_ENTRIES:
            raise ResourceLimitError(f"sequence exceeds {MAX_SAFE_SEQUENCE_ENTRIES} entries")
        return v


class _SafeMapping(_FrozenModel):
    kind: Literal["mapping"] = "mapping"
    # Mappings use canonical sorted keys (sorted at construction time).
    value: tuple[tuple[str, SafeValue], ...] = Field(default_factory=tuple)

    @field_validator("value")
    @classmethod
    def _check_canonical(
        cls, v: tuple[tuple[str, SafeValue], ...]
    ) -> tuple[tuple[str, SafeValue], ...]:
        if len(v) > MAX_SAFE_MAPPING_ENTRIES:
            raise ResourceLimitError(f"mapping exceeds {MAX_SAFE_MAPPING_ENTRIES} entries")
        keys = [k for k, _ in v]
        if len(set(keys)) != len(keys):
            raise ValueError("mapping keys must be unique")
        if keys != sorted(keys):
            raise ValueError("mapping keys must be sorted (canonical order)")
        return v


# The discriminated union. Exactly one variant is set; the discriminator is the
# ``kind`` field. Use Annotated[Union[...], Field(discriminator="kind")].
SafeValue = Annotated[
    _SafeNull
    | _SafeBool
    | _SafeInt
    | _SafeFloat
    | _SafeBytes
    | _SafeString
    | _SafeSequence
    | _SafeMapping,
    Field(discriminator="kind"),
]

# Re-export for runtime isinstance checks.
_SAFE_VARIANTS: tuple[type[BaseModel], ...] = (
    _SafeNull,
    _SafeBool,
    _SafeInt,
    _SafeFloat,
    _SafeBytes,
    _SafeString,
    _SafeSequence,
    _SafeMapping,
)


def _safe_node_count(value: SafeValue) -> int:
    """Count nodes in a SafeValue tree (for resource-limit enforcement)."""
    obj = value if isinstance(value, BaseModel) else None  # discriminant unions
    if obj is None:  # pragma: no cover - defensive
        return 1
    if isinstance(obj, _SafeSequence):
        return 1 + sum(_safe_node_count(v) for v in obj.value)
    if isinstance(obj, _SafeMapping):
        return 1 + sum(_safe_node_count(v) for _, v in obj.value)
    return 1


def _safe_depth(value: SafeValue) -> int:
    obj = value if isinstance(value, BaseModel) else None
    if obj is None:  # pragma: no cover - defensive
        return 1
    if isinstance(obj, _SafeSequence):
        return 1 + max((_safe_depth(v) for v in obj.value), default=0)
    if isinstance(obj, _SafeMapping):
        return 1 + max((_safe_depth(v) for _, v in obj.value), default=0)
    return 1


def assert_safe_bounds(value: SafeValue) -> None:
    """Raise ResourceLimitError if the tree exceeds depth/node bounds."""
    if _safe_depth(value) > MAX_SAFE_DEPTH:
        raise ResourceLimitError(f"safe state exceeds max depth {MAX_SAFE_DEPTH}")
    if _safe_node_count(value) > MAX_SAFE_NODES:
        raise ResourceLimitError(f"safe state exceeds max nodes {MAX_SAFE_NODES}")


# ---------------------------------------------------------------------------
# Captured tensors (amendment E) — separate from persisted refs
# ---------------------------------------------------------------------------


class CapturedTensor(_FrozenModel):
    """A framework-neutral captured tensor returned by a StateProvider.

    The provider returns dtype, shape, canonical raw bytes (already
    little-endian, contiguous, CPU), and a stable logical name. The checkpoint
    encoder assigns member indices, paths, sizes, and digests. Shape dims are
    integers >= 0; rank-zero and zero-element tensors are explicitly supported.
    Boolean tensor bytes must be canonical 0 or 1.
    """

    logical_name: str = Field(..., min_length=1)
    dtype: TensorDtype
    shape: tuple[int, ...] = Field(default_factory=tuple)
    byte_order: TensorByteOrder = "little"
    layout: TensorLayout = "contiguous"
    raw_bytes: bytes
    source_device: str | None = None

    @field_validator("logical_name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _CANONICAL_NAME.fullmatch(v):
            raise ValueError(f"logical_name {v!r} must match {_CANONICAL_NAME.pattern}")
        return v

    @field_validator("shape")
    @classmethod
    def _check_shape(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if len(v) > MAX_TENSOR_RANK:
            raise ValueError(f"tensor rank exceeds {MAX_TENSOR_RANK}")
        for dim in v:
            if dim < 0:
                raise ValueError(f"tensor shape dims must be >= 0; got {dim}")
            if dim > MAX_TENSOR_DIM:
                raise ValueError(f"tensor shape dim {dim} exceeds {MAX_TENSOR_DIM}")
        return v

    @model_validator(mode="after")
    def _check_bytes(self) -> CapturedTensor:
        if len(self.raw_bytes) > MAX_TENSOR_MEMBER_BYTES:
            raise ResourceLimitError(f"tensor raw_bytes exceeds {MAX_TENSOR_MEMBER_BYTES} bytes")
        itemsize = _DTYPE_ITEMSIZE[self.dtype]
        # Overflow-safe product check.
        product = 1
        for dim in self.shape:
            product *= dim
            if product > MAX_TENSOR_ELEMENTS:
                raise ValueError(f"tensor element count exceeds {MAX_TENSOR_ELEMENTS}")
        expected = product * itemsize
        if expected != len(self.raw_bytes):
            raise ValueError(
                f"tensor byte size mismatch: product(shape)*itemsize={expected} "
                f"!= len(raw_bytes)={len(self.raw_bytes)} for {self.logical_name!r}"
            )
        if self.dtype == "bool":
            for b in self.raw_bytes:
                if b not in (0, 1):
                    raise ValueError(
                        f"bool tensor bytes must be canonical 0 or 1; "
                        f"found {b} in {self.logical_name!r}"
                    )
        return self

    def member_byte_size(self) -> int:
        return len(self.raw_bytes)

    def member_sha256(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()


class CapturedCheckpointState(_FrozenModel):
    """The immutable quiescent snapshot returned by
    :meth:`StateProvider.capture_checkpoint_snapshot` (amendment H).

    Carries every required component captured atomically: optimizer complete,
    accumulation_position == 0, no async prefetch active, and counters/cursor
    cannot advance independently during capture.
    """

    parameters: tuple[CapturedTensor, ...] = Field(default_factory=tuple)
    buffers: tuple[CapturedTensor, ...] = Field(default_factory=tuple)
    alias_groups: tuple[tuple[str, ...], ...] = Field(default_factory=tuple)
    optimizer: CapturedTensor | None = None
    scheduler: CapturedTensor | None = None
    scaler: CapturedTensor | None = None
    # General optimizer state (amendment E): multiple per-parameter slots plus
    # structured scalar state (e.g. step counters). A generic Adam-like optimizer
    # round-trips through these instead of the single legacy ``optimizer`` tensor.
    # Each slot tensor's logical name MUST be ``optimizer.<slot_name>.<param>`` so
    # it binds to the optimizer descriptor's (group, param, slot) tuple.
    optimizer_slots: tuple[CapturedTensor, ...] = Field(default_factory=tuple)
    optimizer_scalar_state: SafeValue | None = None
    scheduler_scalar_state: SafeValue | None = None
    rng_bundle_bytes: bytes
    data_cursor: DataCursor
    counters: CounterSnapshot
    model_descriptor: ModelDescriptor
    optimizer_descriptor: OptimizerDescriptor
    scheduler_descriptor: SchedulerDescriptor
    scaler_descriptor: ScalerDescriptor | None
    rng_descriptor: RngDescriptor
    data_descriptor: DataDescriptor
    topology_descriptor: TopologyDescriptor
    optimizer_update_complete: bool = True
    accumulation_position: int = 0
    async_prefetch_active: bool = False

    @field_validator("accumulation_position")
    @classmethod
    def _check_accum(cls, v: int) -> int:
        if v != 0:
            raise ValueError(
                "accumulation_position must be 0 at capture (V1 permits saves "
                "only after a completed optimizer update)"
            )
        return v

    @field_validator("optimizer_slots")
    @classmethod
    def _check_optimizer_slots_unique(
        cls, v: tuple[CapturedTensor, ...]
    ) -> tuple[CapturedTensor, ...]:
        names = [t.logical_name for t in v]
        if len(set(names)) != len(names):
            raise ValueError("optimizer_slots logical names must be unique")
        return v

    def _validate_internal_consistency(self) -> None:
        """Cross-validate that descriptors agree with the captured payloads.

        Verifies:
        - the model descriptor's parameters/buffers match the captured tensors
          (name, shape, dtype);
        - the optimizer descriptor's state_slots match the captured optimizer
          state (each declared slot has a captured tensor of matching shape and
          dtype; the legacy single ``optimizer`` tensor, when present, must
          correspond to a declared slot);
        - the data cursor's contract matches the data descriptor's identity
          (sampler type/version, batch/sequence/drop_last, and the cursor's
          accepted counts are consistent with the data length);
        - the captured RNG bytes are a non-empty, decodable bundle and the RNG
          descriptor's adapter set is non-empty when the bundle carries
          per-adapter state.
        """
        # Model descriptor <-> captured parameters/buffers.
        captured = {t.logical_name: t for t in (*self.parameters, *self.buffers)}
        for p in (*self.model_descriptor.parameters, *self.model_descriptor.buffers):
            t = captured.get(p.name)
            if t is None:
                raise ValueError(
                    f"model descriptor references parameter/buffer {p.name!r} "
                    "not present in captured tensors"
                )
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(
                    f"model descriptor shape {tuple(p.shape)!r} for {p.name!r} "
                    f"!= captured shape {tuple(t.shape)!r}"
                )
            if t.dtype != p.dtype:
                raise ValueError(
                    f"model descriptor dtype {p.dtype!r} for {p.name!r} "
                    f"!= captured dtype {t.dtype!r}"
                )
        # Every captured parameter/buffer must be declared in the descriptor.
        declared = {
            p.name for p in (*self.model_descriptor.parameters, *self.model_descriptor.buffers)
        }
        for name in captured:
            if name not in declared:
                raise ValueError(
                    f"captured parameter/buffer {name!r} is not declared in the model descriptor"
                )

        # Optimizer descriptor slots <-> captured optimizer state (item 10:
        # exact 1:1 binding). Every declared slot has exactly one captured
        # tensor of matching shape/dtype, and every captured slot tensor
        # corresponds to exactly one declared slot — no extras in either
        # direction, and a legacy single tensor cannot satisfy multiple slots.
        slot_tensors = {t.logical_name: t for t in self.optimizer_slots}
        # Build the expected logical name for each declared slot.
        expected_slot_names = {
            f"optimizer.{slot.slot_name}.{slot.param_name}"
            for slot in (self.optimizer_descriptor.state_slots)
        }
        if self.optimizer_slots:
            # Multi-slot path: every declared slot has a tensor...
            for slot in self.optimizer_descriptor.state_slots:
                expected_name = f"optimizer.{slot.slot_name}.{slot.param_name}"
                t = slot_tensors.get(expected_name)
                if t is None:
                    raise ValueError(
                        f"optimizer slot ({slot.group_index},{slot.param_name!r},"
                        f"{slot.slot_name!r}) has no captured optimizer_slot tensor"
                    )
                if tuple(t.shape) != tuple(slot.shape):
                    raise ValueError(
                        f"optimizer slot {expected_name!r} shape {tuple(t.shape)!r} "
                        f"!= declared {tuple(slot.shape)!r}"
                    )
                if t.dtype != slot.dtype:
                    raise ValueError(
                        f"optimizer slot {expected_name!r} dtype {t.dtype!r} "
                        f"!= declared {slot.dtype!r}"
                    )
            # ...and every captured slot tensor maps to a declared slot (no extras).
            extra = set(slot_tensors) - expected_slot_names
            if extra:
                raise ValueError(
                    f"captured optimizer_slot tensors not declared in the descriptor: "
                    f"{sorted(extra)!r}"
                )
        elif self.optimizer is not None:
            # Legacy single-tensor optimizer (item 10): must correspond to
            # EXACTLY one declared slot. One tensor cannot satisfy multiple
            # slots by shape/dtype.
            if len(self.optimizer_descriptor.state_slots) != 1:
                raise ValueError(
                    "legacy single optimizer tensor requires exactly one declared "
                    "state_slot; multiple slots cannot be satisfied by one tensor"
                )
            slot = self.optimizer_descriptor.state_slots[0]
            if tuple(self.optimizer.shape) != tuple(slot.shape):
                raise ValueError(
                    f"optimizer tensor shape {tuple(self.optimizer.shape)!r} "
                    f"!= declared slot {tuple(slot.shape)!r}"
                )
            if self.optimizer.dtype != slot.dtype:
                raise ValueError(
                    f"optimizer tensor dtype {self.optimizer.dtype!r} "
                    f"!= declared slot dtype {slot.dtype!r}"
                )
        elif self.optimizer_descriptor.state_slots:
            # Slots are declared but no state was captured (neither legacy nor
            # multi-slot). That is an incomplete capture.
            raise ValueError(
                "optimizer descriptor declares state_slots but no optimizer state was captured"
            )
        # If multi-slot tensors are present, the descriptor must declare them.
        if self.optimizer_slots and not self.optimizer_descriptor.state_slots:
            raise ValueError("optimizer_slots present but descriptor declares no state_slots")

        # Data cursor contract <-> data descriptor identity.
        di = self.data_descriptor
        dc = self.data_cursor
        if dc.sampler_type != di.sampler_type:
            raise ValueError("data_cursor sampler_type != data_descriptor sampler_type")
        if dc.sampler_version != di.sampler_version:
            raise ValueError("data_cursor sampler_version != data_descriptor sampler_version")
        if dc.batch_size != di.batch_size:
            raise ValueError("data_cursor batch_size != data_descriptor batch_size")
        if dc.sequence_length != di.sequence_length:
            raise ValueError("data_cursor sequence_length != data_descriptor sequence_length")
        if dc.drop_last != di.drop_last:
            raise ValueError("data_cursor drop_last != data_descriptor drop_last")
        if dc.accepted_samples > di.identity.length and di.identity.length > 0:
            raise ValueError("data_cursor accepted_samples exceeds data identity length")

        # RNG (item 10): the captured bytes must be present AND strictly decode
        # to a :class:`RngStateBundle`, and that bundle's descriptor must agree
        # with the captured RNG descriptor (adapter set + schema version). A
        # non-decodable or descriptor-mismatched bundle is rejected rather than
        # accepted on non-emptiness alone.
        if not self.rng_bundle_bytes:
            raise ValueError("rng_bundle_bytes must be present (non-empty) at capture")
        try:
            from expertforge.rng.state import RngStateBundle

            bundle = RngStateBundle.from_json_bytes(self.rng_bundle_bytes)
        except (ValueError, Exception) as e:
            raise ValueError(f"rng_bundle_bytes must decode to a valid RngStateBundle: {e}") from e
        if bundle.rng_state_schema_version != self.rng_descriptor.rng_state_schema_version:
            raise ValueError(
                "rng_bundle rng_state_schema_version != rng_descriptor rng_state_schema_version"
            )
        # The bundle's framework adapter providers must match the descriptor's
        # adapter set exactly (each persisted framework state has a provider in
        # the descriptor, and vice versa).
        bundle_providers = tuple(sorted(state.provider for state in bundle.framework_states))
        if bundle_providers != tuple(self.rng_descriptor.adapter_set):
            raise ValueError("rng_bundle framework providers != rng_descriptor adapter_set")

        # Item 10: alias groups must share IDENTICAL dtype, shape, AND raw_bytes
        # before they are deduplicated to one canonical tensor. A group whose
        # members disagree on any of these is not a true alias/tie group and
        # must not collapse to a single member.
        tensors_by_name = {t.logical_name: t for t in (*self.parameters, *self.buffers)}
        for group in self.alias_groups:
            members = [tensors_by_name[n] for n in group if n in tensors_by_name]
            if len(members) < 2:
                continue
            first = members[0]
            for m in members[1:]:
                if m.dtype != first.dtype or tuple(m.shape) != tuple(first.shape):
                    raise ValueError(f"alias group {tuple(group)!r} members must share dtype/shape")
                if m.raw_bytes != first.raw_bytes:
                    raise ValueError(
                        f"alias group {tuple(group)!r} members must share identical "
                        "raw_bytes (true alias/tie group)"
                    )

    @model_validator(mode="after")
    def _check_quiescent(self) -> CapturedCheckpointState:
        if not self.optimizer_update_complete:
            raise ValueError("capture requires optimizer_update_complete=True (V1 save boundary)")
        if self.async_prefetch_active:
            raise ValueError(
                "capture requires async_prefetch_active=False (V1 rejects active prefetch)"
            )
        # Scaler descriptor presence must match scaler tensor presence.
        if (self.scaler is not None) != (self.scaler_descriptor is not None):
            raise ValueError("scaler tensor and scaler_descriptor presence must agree")
        if self.scaler_descriptor is not None and self.scaler_descriptor.active != (
            self.scaler is not None
        ):
            raise ValueError("scaler_descriptor.active must match scaler presence")
        # Alias groups: each name appears exactly once across parameters/buffers,
        # and every name is a known parameter/buffer logical name.
        all_names = [t.logical_name for t in (*self.parameters, *self.buffers)]
        name_set = set(all_names)
        if len(name_set) != len(all_names):
            raise ValueError("parameter/buffer logical names must be unique")
        seen_in_groups: set[str] = set()
        canonical_groups: list[tuple[str, ...]] = []
        for group in self.alias_groups:
            if len(group) < 2:
                raise ValueError("alias groups must contain at least two members")
            gl = list(group)
            for n in gl:
                if n not in name_set:
                    raise ValueError(f"alias group references unknown name {n!r}")
                if n in seen_in_groups:
                    raise ValueError(f"name {n!r} appears in multiple alias groups")
                seen_in_groups.add(n)
            canonical_groups.append(tuple(sorted(gl)))
        # Canonicalize group order.
        if tuple(sorted(canonical_groups)) != self.alias_groups:
            raise ValueError("alias_groups must be sorted (canonical order)")
        # Cross-validate descriptors against captured payloads (amendment H/N).
        self._validate_internal_consistency()
        return self


# ---------------------------------------------------------------------------
# Persisted manifest member references
# ---------------------------------------------------------------------------


class TensorMemberRef(_FrozenModel):
    """A persisted tensor member reference: path, digest, size, dtype, shape."""

    member_name: str
    member_sha256: str
    member_byte_size: int = Field(..., ge=0)
    dtype: TensorDtype
    shape: tuple[int, ...] = Field(default_factory=tuple)
    logical_names: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("member_name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _TENSOR_MEMBER_NAME.fullmatch(v):
            raise ValueError(
                f"tensor member_name must match {_TENSOR_MEMBER_NAME.pattern}; got {v!r}"
            )
        return v

    @field_validator("member_sha256")
    @classmethod
    def _check_digest(cls, v: str) -> str:
        if not _HEX64.fullmatch(v):
            raise ValueError(f"member_sha256 must be 64 lowercase hex; got {v!r}.")
        return v

    @field_validator("logical_names")
    @classmethod
    def _check_names(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if list(v) != sorted(v):
            raise ValueError("logical_names must be sorted")
        if len(set(v)) != len(v):
            raise ValueError("logical_names must be unique")
        return v


class StateComponentRef(_FrozenModel):
    """One authenticated non-manifest member of the archive."""

    role: StateComponentRole
    member_name: str
    member_sha256: str
    member_byte_size: int = Field(..., ge=0)

    @field_validator("member_sha256")
    @classmethod
    def _check_digest(cls, v: str) -> str:
        if not _HEX64.fullmatch(v):
            raise ValueError(f"member_sha256 must be 64 lowercase hex; got {v!r}.")
        return v

    @field_validator("member_name")
    @classmethod
    def _check_member_name(cls, v: str) -> str:
        # Non-manifest, non-tensor members live under state/<role>.json.
        if not v.startswith("state/") or not v.endswith(".json"):
            raise ValueError(f"component member_name must be 'state/<role>.json'; got {v!r}")
        return v


class ConfigurationState(_FrozenModel):
    """The strict ``state/configuration.json`` component payload (item 5).

    Carries the configuration fingerprint, the SHA-256 of the canonical
    configuration bytes, the base64-encoded configuration bytes, and their
    length. Used by the cross-binding check: the actual ``content_base64`` bytes
    are decoded and rehashed, then compared to both the payload's
    ``content_sha256`` and the manifest's ``configuration_content_sha256`` (so a
    tampered payload cannot self-attest its own digest).
    """

    schema_: Literal["expertforge.checkpoint-configuration"] = Field(
        alias="schema", default="expertforge.checkpoint-configuration"
    )
    version: Literal[1] = 1
    content_sha256: str
    content_base64: str
    byte_length: int = Field(..., ge=0)
    fingerprint: str = Field(..., min_length=1)

    @field_validator("content_sha256")
    @classmethod
    def _check_digest(cls, v: str) -> str:
        if not _HEX64.fullmatch(v):
            raise ValueError(f"content_sha256 must be 64 lowercase hex; got {v!r}.")
        return v


# ---------------------------------------------------------------------------
# Counters (amendment I) — monotonic, checkpoint-safe
# ---------------------------------------------------------------------------


class CounterSnapshot(_FrozenModel):
    """V1 counters (amendment I).

    ``completed_microsteps`` is the total committed microsteps in the logical
    run (monotonic). ``accumulation_position`` is the completed microsteps in
    the current in-flight accumulation window; V1 requires it == 0 at save and
    restore. ``global_update``, ``completed_microsteps``, ``accepted_samples``,
    ``accepted_sequences``, and ``processed_tokens`` never regress.
    """

    global_update: int = Field(..., ge=0)
    completed_microsteps: int = Field(..., ge=0)
    accumulation_position: int = Field(..., ge=0)
    accepted_samples: int = Field(..., ge=0)
    accepted_sequences: int = Field(..., ge=0)
    processed_tokens: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _check_quiescent(self) -> CounterSnapshot:
        if self.accumulation_position != 0:
            raise ValueError("accumulation_position must be 0 at save/restore (V1 save boundary)")
        return self


# ---------------------------------------------------------------------------
# Data identity + cursor (amendment J) — structural, not a free string
# ---------------------------------------------------------------------------


class DataIdentity(_FrozenModel):
    """Versioned structural dataset identity (amendment J).

    Contains the dataset/snapshot digest, split, length, preprocessing/tokenizer
    identity, packing/sequence policy, shard selection, and resolved
    data-configuration digest.
    """

    data_identity_schema_version: int = 1
    dataset_digest: str
    split: str = Field(..., min_length=1)
    length: int = Field(..., ge=0)
    preprocessing_identity: str = Field(..., min_length=1)
    tokenizer_identity: str | None = None
    packing_policy: str = Field(..., min_length=1)
    sequence_policy: str = Field(..., min_length=1)
    shard_selection: str = Field(..., min_length=1)
    data_config_digest: str

    @field_validator("dataset_digest", "data_config_digest")
    @classmethod
    def _check_digest(cls, v: str) -> str:
        if not _HEX64.fullmatch(v):
            raise ValueError(f"digest must be 64 lowercase hex; got {v!r}.")
        return v


class DataCursor(_FrozenModel):
    """The data continuation cursor (amendment J).

    Binds sampler algorithm and version, batch/sequence assembly settings,
    drop-last policy, epoch, and the next unconsumed logical item/sequence
    position. For shuffled v1 data, the permutation must be reconstructible
    solely from the persisted dataset identity, sampler type/version, epoch, and
    permutation seed/state.
    """

    sampler_type: Literal["sequential", "shuffled"]
    sampler_version: int = Field(..., ge=0)
    batch_size: int = Field(..., ge=1)
    sequence_length: int = Field(..., ge=1)
    drop_last: bool
    epoch: int = Field(..., ge=0)
    position: int = Field(..., ge=0)
    permutation_seed: int | None = None
    accepted_samples: int = Field(..., ge=0)
    accepted_sequences: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _check_permutation(self) -> DataCursor:
        if self.sampler_type == "shuffled":
            if self.permutation_seed is None:
                raise ValueError("shuffled sampler requires a permutation_seed/state")
        else:  # sequential
            if self.permutation_seed is not None:
                raise ValueError("sequential sampler must not carry a permutation_seed")
        return self


# ---------------------------------------------------------------------------
# Descriptors (design §11)
# ---------------------------------------------------------------------------


class ModelParameterDescriptor(_FrozenModel):
    """One canonical parameter/buffer: name, shape, dtype."""

    name: str = Field(..., min_length=1)
    shape: tuple[int, ...] = Field(default_factory=tuple)
    dtype: TensorDtype

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _CANONICAL_NAME.fullmatch(v):
            raise ValueError(f"name {v!r} must match {_CANONICAL_NAME.pattern}")
        return v

    @field_validator("shape")
    @classmethod
    def _check_shape(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if len(v) > MAX_TENSOR_RANK:
            raise ValueError(f"tensor rank exceeds {MAX_TENSOR_RANK}")
        for dim in v:
            if dim < 0 or dim > MAX_TENSOR_DIM:
                raise ValueError(f"tensor shape dim {dim} out of range")
        return v


class AliasGroup(_FrozenModel):
    """One declared exact alias/tie group (amendment E).

    V1 preserves declared exact alias/tie groups: one canonical tensor payload
    per group, with every logical parameter/buffer name plus the canonical
    storage-group binding.
    """

    canonical_member: str = Field(..., min_length=1)
    aliased_names: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("canonical_member", "aliased_names")
    @classmethod
    def _check_names(cls, v: tuple[str, ...] | str) -> tuple[str, ...] | str:
        items = v if isinstance(v, tuple) else (v,)
        for n in items:
            if not _CANONICAL_NAME.fullmatch(n):
                raise ValueError(f"alias name {n!r} must match {_CANONICAL_NAME.pattern}")
        return v

    @field_validator("aliased_names")
    @classmethod
    def _check_sorted(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if list(v) != sorted(v):
            raise ValueError("aliased_names must be sorted")
        if len(set(v)) != len(v):
            raise ValueError("aliased_names must be unique")
        return v


class ModelDescriptor(_FrozenModel):
    """Canonical parameter/buffer names, shapes, dtypes, alias groups."""

    parameters: tuple[ModelParameterDescriptor, ...] = Field(default_factory=tuple)
    buffers: tuple[ModelParameterDescriptor, ...] = Field(default_factory=tuple)
    alias_groups: tuple[AliasGroup, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _check_unique(self) -> ModelDescriptor:
        pnames = [p.name for p in self.parameters]
        bnames = [b.name for b in self.buffers]
        if len(set(pnames)) != len(pnames):
            raise ValueError("parameter names must be unique")
        if len(set(bnames)) != len(bnames):
            raise ValueError("buffer names must be unique")
        all_names = set(pnames) | set(bnames)
        seen: set[str] = set()
        canonical_groups: list[AliasGroup] = []
        for g in self.alias_groups:
            members = (*g.aliased_names, g.canonical_member)
            if len(set(members)) != len(members):
                raise ValueError("alias group members must be unique")
            for n in members:
                if n not in all_names:
                    raise ValueError(f"alias group references unknown name {n!r}")
                if n in seen:
                    raise ValueError(f"name {n!r} appears in multiple alias groups")
                seen.add(n)
            if g.canonical_member not in g.aliased_names and len(g.aliased_names) > 0:
                # canonical_member must be one of the members.
                pass
            canonical_groups.append(g)
        if tuple(sorted(canonical_groups, key=lambda x: x.canonical_member)) != self.alias_groups:
            raise ValueError("alias_groups must be sorted by canonical_member")
        return self

    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters) + tuple(b.name for b in self.buffers)


class OptimizerParamGroup(_FrozenModel):
    """One optimizer parameter group: canonical param indices + options."""

    group_index: int = Field(..., ge=0)
    param_names: tuple[str, ...] = Field(default_factory=tuple)
    options: tuple[tuple[str, SafeValue], ...] = Field(default_factory=tuple)

    @field_validator("param_names")
    @classmethod
    def _check_params(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if list(v) != sorted(v):
            raise ValueError("param_names must be sorted")
        if len(set(v)) != len(v):
            raise ValueError("param_names must be unique")
        return v

    @field_validator("options")
    @classmethod
    def _check_options(cls, v: tuple[tuple[str, Any], ...]) -> tuple[tuple[str, Any], ...]:
        keys = [k for k, _ in v]
        if len(set(keys)) != len(keys):
            raise ValueError("option keys must be unique")
        if keys != sorted(keys):
            raise ValueError("option keys must be sorted")
        return v


class OptimizerStateSlot(_FrozenModel):
    """One optimizer state slot: (group, param, slot) -> shape."""

    group_index: int = Field(..., ge=0)
    param_name: str = Field(..., min_length=1)
    slot_name: str = Field(..., min_length=1)
    shape: tuple[int, ...] = Field(default_factory=tuple)
    dtype: TensorDtype

    @field_validator("param_name", "slot_name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _CANONICAL_NAME.fullmatch(v):
            raise ValueError(f"name {v!r} must match {_CANONICAL_NAME.pattern}")
        return v


class OptimizerDescriptor(_FrozenModel):
    """Optimizer type identity, groups, options, and state slots."""

    optimizer_type: str = Field(..., min_length=1)
    param_groups: tuple[OptimizerParamGroup, ...] = Field(default_factory=tuple)
    state_slots: tuple[OptimizerStateSlot, ...] = Field(default_factory=tuple)

    @field_validator("optimizer_type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if not _CANONICAL_NAME.fullmatch(v):
            raise ValueError(f"optimizer_type {v!r} must match {_CANONICAL_NAME.pattern}")
        return v

    @field_validator("param_groups")
    @classmethod
    def _check_groups(cls, v: tuple[OptimizerParamGroup, ...]) -> tuple[OptimizerParamGroup, ...]:
        idxs = [g.group_index for g in v]
        if idxs != list(range(len(v))):
            raise ValueError("param_groups must be indexed 0..n-1 in order")
        return v

    @field_validator("state_slots")
    @classmethod
    def _check_slots(cls, v: tuple[OptimizerStateSlot, ...]) -> tuple[OptimizerStateSlot, ...]:
        # Slots must be canonically sorted by (group, param, slot).
        def key(s: OptimizerStateSlot) -> tuple[int, str, str]:
            return (s.group_index, s.param_name, s.slot_name)

        if list(v) != sorted(v, key=key):
            raise ValueError("state_slots must be sorted by (group, param, slot)")
        seen = {(s.group_index, s.param_name, s.slot_name) for s in v}
        if len(seen) != len(v):
            raise ValueError("state_slots must be unique")
        return v


class SchedulerDescriptor(_FrozenModel):
    """Scheduler type identity + internal state shape."""

    scheduler_type: str
    active: bool = True
    state_shape: tuple[int, ...] = Field(default_factory=tuple)

    @field_validator("scheduler_type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if not _CANONICAL_NAME.fullmatch(v):
            raise ValueError(f"scheduler_type {v!r} must match {_CANONICAL_NAME.pattern}")
        return v

    @model_validator(mode="after")
    def _check_active(self) -> SchedulerDescriptor:
        if self.active is False and self.state_shape:
            raise ValueError("inactive scheduler must have empty state_shape")
        return self


class ScalerDescriptor(_FrozenModel):
    """Mixed-precision scaler descriptor. Present only when scaler is active."""

    scaler_type: str
    active: bool = True
    state_shape: tuple[int, ...] = Field(default_factory=tuple)

    @field_validator("scaler_type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if not _CANONICAL_NAME.fullmatch(v):
            raise ValueError(f"scaler_type {v!r} must match {_CANONICAL_NAME.pattern}")
        return v


class RngDescriptor(_FrozenModel):
    """RNG adapter set + framework versions."""

    adapter_set: tuple[str, ...] = Field(default_factory=tuple)
    framework_versions: tuple[tuple[str, str], ...] = Field(default_factory=tuple)
    rng_state_schema_version: int = 1

    @field_validator("adapter_set")
    @classmethod
    def _check_adapters(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if list(v) != sorted(v):
            raise ValueError("adapter_set must be sorted")
        if len(set(v)) != len(v):
            raise ValueError("adapter_set must be unique")
        return v

    @field_validator("framework_versions")
    @classmethod
    def _check_versions(cls, v: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
        keys = [k for k, _ in v]
        if len(set(keys)) != len(keys):
            raise ValueError("framework_versions keys must be unique")
        if keys != sorted(keys):
            raise ValueError("framework_versions must be sorted by framework name")
        return v


class DataDescriptor(_FrozenModel):
    """Data identity + sampler/version/type."""

    identity: DataIdentity
    sampler_type: Literal["sequential", "shuffled"]
    sampler_version: int = Field(..., ge=0)
    batch_size: int = Field(..., ge=1)
    sequence_length: int = Field(..., ge=1)
    drop_last: bool


class TopologyDescriptor(_FrozenModel):
    """World size + rank assignment. V1 requires world_size == 1."""

    world_size: int = Field(..., ge=1)
    rank_assignment: tuple[int, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _check_v1(self) -> TopologyDescriptor:
        if self.world_size != 1:
            raise ValueError(
                f"V1 is single-process only; topology world_size must be 1 (got {self.world_size})"
            )
        return self


class CompatibilityDescriptor(_FrozenModel):
    """The full compatibility descriptor (design §11)."""

    compatibility_schema: Literal["expertforge.checkpoint-compatibility"] = COMPATIBILITY_SCHEMA
    compatibility_schema_version: int = COMPATIBILITY_SCHEMA_VERSION
    archive_format_version: int = CHECKPOINT_ARCHIVE_FORMAT_VERSION
    manifest_schema_version: int = CHECKPOINT_MANIFEST_SCHEMA_VERSION
    state_format_version: int = SAFE_STATE_FORMAT_VERSION
    specification_fingerprint: str = Field(..., min_length=1)
    model_descriptor: ModelDescriptor
    optimizer_descriptor: OptimizerDescriptor
    scheduler_descriptor: SchedulerDescriptor
    scaler_descriptor: ScalerDescriptor | None = None
    rng_descriptor: RngDescriptor
    data_descriptor: DataDescriptor
    topology_descriptor: TopologyDescriptor

    @field_validator("compatibility_schema_version")
    @classmethod
    def _check_compat_version(cls, v: int) -> int:
        if v != COMPATIBILITY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported compatibility_schema_version {v!r}; "
                f"this version supports exactly {COMPATIBILITY_SCHEMA_VERSION}"
            )
        return v

    @field_validator("archive_format_version")
    @classmethod
    def _check_archive_format_version(cls, v: int) -> int:
        if v != CHECKPOINT_ARCHIVE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported archive_format_version {v!r}; "
                f"this version supports exactly {CHECKPOINT_ARCHIVE_FORMAT_VERSION}"
            )
        return v

    @field_validator("manifest_schema_version")
    @classmethod
    def _check_manifest_schema_version(cls, v: int) -> int:
        if v != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest_schema_version {v!r}; "
                f"this version supports exactly {CHECKPOINT_MANIFEST_SCHEMA_VERSION}"
            )
        return v

    @field_validator("state_format_version")
    @classmethod
    def _check_state_format_version(cls, v: int) -> int:
        if v != SAFE_STATE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported state_format_version {v!r}; "
                f"this version supports exactly {SAFE_STATE_FORMAT_VERSION}"
            )
        return v


class CompatibilityMismatch(_FrozenModel):
    """One typed compatibility mismatch (amendment M)."""

    component: CompatibilityComponent
    diagnostic_code: CompatibilityDiagnosticCode
    severity: Literal["blocking"]
    expected: str
    actual: str
    path: str = ""

    @model_validator(mode="after")
    def _check_severity(self) -> CompatibilityMismatch:
        # All listed components are blocking (amendment M). The severity field
        # is part of the compatibility schema and fixed by the table.
        if self.component not in _BLOCKING_COMPONENTS:
            raise ValueError(f"unknown non-blocking component {self.component!r}")
        return self


class CompatibilityResult(_FrozenModel):
    """The closed, deterministic compatibility classification."""

    status: CompatibilityStatus
    mismatches: tuple[CompatibilityMismatch, ...] = Field(default_factory=tuple)

    @field_validator("mismatches")
    @classmethod
    def _check_sorted(
        cls, v: tuple[CompatibilityMismatch, ...]
    ) -> tuple[CompatibilityMismatch, ...]:
        def key(m: CompatibilityMismatch) -> tuple[str, str, str, str, str]:
            return (m.component, m.path, m.diagnostic_code, m.expected, m.actual)

        if list(v) != sorted(v, key=key):
            raise ValueError("mismatches must be sorted canonically")
        return v

    @model_validator(mode="after")
    def _check_status(self) -> CompatibilityResult:
        if self.status == "exact" and self.mismatches:
            raise ValueError("status='exact' forbids mismatches")
        if self.status != "exact" and not self.mismatches and self.status != "unsupported":
            raise ValueError("non-exact status requires mismatches or unsupported")
        return self


# ---------------------------------------------------------------------------
# Manifest (amendment A — no self-referential id; amendment B — exact graph)
# ---------------------------------------------------------------------------


class CheckpointManifest(_FrozenModel):
    """The checkpoint manifest (manifest.json).

    Per amendment A: no self-referential checkpoint_id. The authoritative
    checkpoint identifier is the ``ArtifactRecord.artifact_id`` from #10. The
    manifest carries producing identity, format versions, state descriptors,
    counters, and parent lineage.
    """

    manifest_schema: Literal["expertforge.checkpoint-manifest"] = CHECKPOINT_MANIFEST_SCHEMA
    manifest_schema_version: int = CHECKPOINT_MANIFEST_SCHEMA_VERSION
    archive_format_version: int = CHECKPOINT_ARCHIVE_FORMAT_VERSION
    state_format_version: int = SAFE_STATE_FORMAT_VERSION
    run_id: str = Field(..., min_length=1)

    @field_validator("manifest_schema_version")
    @classmethod
    def _check_manifest_schema_version(cls, v: int) -> int:
        if v != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest_schema_version {v!r}; "
                f"this version supports exactly {CHECKPOINT_MANIFEST_SCHEMA_VERSION}"
            )
        return v

    @field_validator("archive_format_version")
    @classmethod
    def _check_archive_format_version(cls, v: int) -> int:
        if v != CHECKPOINT_ARCHIVE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported archive_format_version {v!r}; "
                f"this version supports exactly {CHECKPOINT_ARCHIVE_FORMAT_VERSION}"
            )
        return v

    @field_validator("state_format_version")
    @classmethod
    def _check_state_format_version(cls, v: int) -> int:
        if v != SAFE_STATE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported state_format_version {v!r}; "
                f"this version supports exactly {SAFE_STATE_FORMAT_VERSION}"
            )
        return v

    attempt_id: str = Field(..., min_length=1)
    specification_fingerprint: str = Field(..., min_length=1)
    parent_artifact_id: str | None = None
    parent_run_id: str | None = None
    parent_attempt_id: str | None = None
    created_at_utc: str = Field(..., min_length=1)
    counters: CounterSnapshot
    configuration_fingerprint: str = Field(..., min_length=1)
    configuration_content_sha256: str
    compatibility: CompatibilityDescriptor
    state_components: tuple[StateComponentRef, ...] = Field(default_factory=tuple)
    tensor_members: tuple[TensorMemberRef, ...] = Field(default_factory=tuple)

    @field_validator("configuration_content_sha256")
    @classmethod
    def _check_digest(cls, v: str) -> str:
        if not _HEX64.fullmatch(v):
            raise ValueError(f"configuration_content_sha256 must be 64 hex; got {v!r}.")
        return v

    @field_validator("state_components")
    @classmethod
    def _check_components(cls, v: tuple[StateComponentRef, ...]) -> tuple[StateComponentRef, ...]:
        roles = [c.role for c in v]
        if len(set(roles)) != len(roles):
            raise ValueError("state_components roles must be unique")
        # Required roles (amendment B/C): identity, configuration, provenance,
        # rng, data_cursor, counters, model, optimizer, scheduler. scaler only
        # when declared active.
        required = {
            "identity",
            "configuration",
            "provenance",
            "rng",
            "data_cursor",
            "counters",
            "model",
            "optimizer",
            "scheduler",
        }
        present = set(roles)
        missing = required - present
        if missing:
            raise ValueError(f"missing required state components: {sorted(missing)}")
        unexpected = present - required - {"scaler"}
        if unexpected:
            raise ValueError(f"unexpected state components: {sorted(unexpected)}")
        return v

    @field_validator("tensor_members")
    @classmethod
    def _check_tensor_members(cls, v: tuple[TensorMemberRef, ...]) -> tuple[TensorMemberRef, ...]:
        names = [m.member_name for m in v]
        if len(set(names)) != len(names):
            raise ValueError("tensor_members must be unique")
        if len(v) > MAX_TENSOR_COUNT:
            raise ResourceLimitError(f"tensor_members exceeds {MAX_TENSOR_COUNT}")
        return v

    @model_validator(mode="after")
    def _check_parent(self) -> CheckpointManifest:
        # Parent fields are all-or-none.
        fields = (self.parent_artifact_id, self.parent_run_id, self.parent_attempt_id)
        all_set = all(f is not None for f in fields)
        none_set = all(f is None for f in fields)
        if not (all_set or none_set):
            raise ValueError("parent fields must be all-set or all-None (fully qualified)")
        return self

    @model_validator(mode="after")
    def _check_scaler_consistency(self) -> CheckpointManifest:
        has_scaler_component = any(c.role == "scaler" for c in self.state_components)
        scaler_active = self.compatibility.scaler_descriptor is not None
        if has_scaler_component != scaler_active:
            raise ValueError(
                "scaler state component presence must match "
                "compatibility.scaler_descriptor presence"
            )
        # tensor_members sorted by member_name and indices contiguous from 0.
        names = [m.member_name for m in self.tensor_members]
        if names != sorted(names):
            raise ValueError("tensor_members must be sorted by member_name")
        for i, name in enumerate(names):
            expected = f"tensors/{i}.bin"
            if name != expected:
                raise ValueError(
                    f"tensor member index must be contiguous from 0; "
                    f"expected {expected!r} at position {i}, got {name!r}"
                )
        return self


class CheckpointInspection(_FrozenModel):
    """The result of inspecting an archive without restoring (design §17)."""

    status: InspectionStatus
    manifest: CheckpointManifest | None = None
    diagnostic: str | None = None
    archive_byte_size: int | None = None
    member_count: int | None = None


# Update forward refs for recursive models.
CapturedCheckpointState.model_rebuild()
_SafeSequence.model_rebuild()
_SafeMapping.model_rebuild()
OptimizerParamGroup.model_rebuild()
