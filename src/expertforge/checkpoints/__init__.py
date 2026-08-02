"""Checkpoint serialization and exact restoration (Issue #11).

This package owns checkpoint-safe state-tree encoding, deterministic POSIX
ustar tar packaging, manifest construction/validation, compatibility checking,
and exact-restoration ordering.

It may import frozen identity types (:class:`AttemptIdentityRecord`, #6), the
RNG manager and state bundle (:mod:`expertforge.rng`, #8), resolved
configuration bytes (:mod:`expertforge.config.resolve`, #5), and the public
:class:`ArtifactStore` API (#10) for publication only.

It must not import telemetry internals, artifact-private modules, or arbitrary
training-framework serialization. Training-framework adapters are explicit
consumer/provider interfaces defined in this package; the framework implements
them, not the other way around. A pure-Python reference adapter is provided for
testing; the deferred training-framework adapter is out of scope for #11.

No import-time side effects: importing this package does not touch the
filesystem, mutate process-global RNG state, or execute deserialization.

Normative references:
- Design comment ``5145501414`` (ratified proposal).
- Binding amendment ``5145649404`` (amendments A–N). Where the two conflict,
  the amendment prevails.
"""

from __future__ import annotations

from expertforge.checkpoints.encoder import (
    ArchiveMembers,
    EncoderError,
    build_archive,
    encode_safe_value,
    safe_value_from_native,
    safe_value_to_native,
)
from expertforge.checkpoints.models import (
    CHECKPOINT_ARCHIVE_FORMAT_VERSION,
    CHECKPOINT_MANIFEST_SCHEMA,
    CHECKPOINT_MANIFEST_SCHEMA_VERSION,
    COMPATIBILITY_SCHEMA,
    COMPATIBILITY_SCHEMA_VERSION,
    MAX_ARCHIVE_BYTES,
    MAX_COMPONENT_MEMBER_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_MEMBER_COUNT,
    MAX_SAFE_DEPTH,
    MAX_SAFE_INLINE_BYTES,
    MAX_SAFE_MAPPING_ENTRIES,
    MAX_SAFE_NODES,
    MAX_SAFE_SEQUENCE_ENTRIES,
    MAX_SAFE_STRING_BYTES,
    MAX_TENSOR_BYTES,
    MAX_TENSOR_COUNT,
    MAX_TENSOR_MEMBER_BYTES,
    SAFE_STATE_FORMAT_VERSION,
    AliasGroup,
    CapturedCheckpointState,
    CapturedTensor,
    CheckpointInspection,
    CheckpointManifest,
    CompatibilityDescriptor,
    CompatibilityDiagnosticCode,
    CompatibilityMismatch,
    CompatibilityResult,
    CompatibilityStatus,
    CounterSnapshot,
    DataCursor,
    DataDescriptor,
    DataIdentity,
    InspectionStatus,
    ModelDescriptor,
    ModelParameterDescriptor,
    OptimizerDescriptor,
    OptimizerParamGroup,
    OptimizerStateSlot,
    ResourceLimitError,
    RngDescriptor,
    SafeStateDecodeError,
    SafeValue,
    SafeValueKind,
    ScalerDescriptor,
    SchedulerDescriptor,
    StateComponentRef,
    StateComponentRole,
    TensorByteOrder,
    TensorDtype,
    TensorLayout,
    TensorMemberRef,
    TopologyDescriptor,
    canonical_json_bytes,
    validate_canonical_digest,
)
from expertforge.checkpoints.reference_adapter import (
    ReferenceCheckpointError,
    ReferenceState,
    ReferenceStateProvider,
    StateConsumer,
    StateProvider,
)
from expertforge.checkpoints.restore import (
    ModelState,
    OptimizerState,
    RestoredState,
    RestoreError,
    RestoreIncompatibleError,
    RestoreTransaction,
    ScalerState,
    StateFactory,
    TensorComponentRef,
    build_restored_tensors,
    decode_component_json,
)
from expertforge.checkpoints.store import (
    CheckpointArchive,
    CheckpointComponentError,
    CheckpointCorruptError,
    CheckpointError,
    CheckpointLineageError,
    CheckpointStore,
    CheckpointTopologyError,
    CheckpointVersionError,
    check_compatibility,
)
from expertforge.checkpoints.tar_reader import ParsedMember, TarParseError, parse_ustar_archive
from expertforge.checkpoints.tar_writer import TarMember, TarWriterError, build_ustar_archive

__all__ = [
    # Version / schema constants
    "CHECKPOINT_ARCHIVE_FORMAT_VERSION",
    "CHECKPOINT_MANIFEST_SCHEMA",
    "CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "COMPATIBILITY_SCHEMA",
    "COMPATIBILITY_SCHEMA_VERSION",
    "SAFE_STATE_FORMAT_VERSION",
    # Resource bounds
    "MAX_ARCHIVE_BYTES",
    "MAX_COMPONENT_MEMBER_BYTES",
    "MAX_MANIFEST_BYTES",
    "MAX_MEMBER_COUNT",
    "MAX_SAFE_DEPTH",
    "MAX_SAFE_INLINE_BYTES",
    "MAX_SAFE_MAPPING_ENTRIES",
    "MAX_SAFE_NODES",
    "MAX_SAFE_SEQUENCE_ENTRIES",
    "MAX_SAFE_STRING_BYTES",
    "MAX_TENSOR_BYTES",
    "MAX_TENSOR_COUNT",
    "MAX_TENSOR_MEMBER_BYTES",
    # Closed domains
    "CompatibilityDiagnosticCode",
    "CompatibilityStatus",
    "InspectionStatus",
    "SafeValueKind",
    "StateComponentRole",
    "TensorByteOrder",
    "TensorDtype",
    "TensorLayout",
    # Core models
    "AliasGroup",
    "CapturedCheckpointState",
    "CapturedTensor",
    "CheckpointInspection",
    "CheckpointManifest",
    "CompatibilityDescriptor",
    "CompatibilityMismatch",
    "CompatibilityResult",
    "CounterSnapshot",
    "DataCursor",
    "DataDescriptor",
    "DataIdentity",
    "ModelDescriptor",
    "ModelParameterDescriptor",
    "OptimizerDescriptor",
    "OptimizerParamGroup",
    "OptimizerStateSlot",
    "RngDescriptor",
    "SafeValue",
    "ScalerDescriptor",
    "SchedulerDescriptor",
    "StateComponentRef",
    "TensorMemberRef",
    "TopologyDescriptor",
    # Errors
    "CheckpointComponentError",
    "CheckpointCorruptError",
    "CheckpointError",
    "CheckpointLineageError",
    "CheckpointTopologyError",
    "CheckpointVersionError",
    "EncoderError",
    "ReferenceCheckpointError",
    "ResourceLimitError",
    "RestoreError",
    "RestoreIncompatibleError",
    "SafeStateDecodeError",
    "TarParseError",
    "TarWriterError",
    # Encoder
    "ArchiveMembers",
    "build_archive",
    "encode_safe_value",
    "safe_value_from_native",
    "safe_value_to_native",
    # Store + restore
    "CheckpointArchive",
    "CheckpointStore",
    "OptimizerState",
    "ParsedMember",
    "ModelState",
    "RestoreTransaction",
    "RestoredState",
    "ScalerState",
    "StateConsumer",
    "StateFactory",
    "StateProvider",
    "ReferenceState",
    "ReferenceStateProvider",
    "TensorComponentRef",
    "build_restored_tensors",
    "build_ustar_archive",
    "check_compatibility",
    "decode_component_json",
    "parse_ustar_archive",
    "TarMember",
    # Canonical encoding helpers
    "canonical_json_bytes",
    "validate_canonical_digest",
]
