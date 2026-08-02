"""Artifact management and content hashing (Issue #10).

This package owns artifact registry records and their deterministic
serialization, canonical path derivation, atomic publication (write → hash →
register), content verification, external reference representation, retention
state transitions, and registry durability (append-only JSONL per attempt).

It may import frozen identity types (:class:`AttemptIdentityRecord`) and frozen
provenance types for binding, but must not depend on training frameworks,
telemetry internals, or checkpoint serialization.

No import-time side effects: importing this package does not touch the
filesystem or mutate process-global state.
"""

from __future__ import annotations

from expertforge.artifacts.models import (
    ARTIFACT_BUNDLE_SCHEMA,
    ARTIFACT_BUNDLE_SCHEMA_VERSION,
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_ID_PATTERN,
    ARTIFACT_RECORD_SCHEMA,
    ARTIFACT_RECORD_SCHEMA_VERSION,
    ARTIFACT_REGISTRY_FORMAT_VERSION,
    EXTERNAL_REFERENCE_SCHEMA,
    EXTERNAL_REFERENCE_SCHEMA_VERSION,
    MAX_REGISTRY_ENTRY_BYTES,
    ArtifactCategory,
    ArtifactConflictError,
    ArtifactDescriptor,
    ArtifactFormat,
    ArtifactNotFoundError,
    ArtifactRecord,
    EntryKind,
    ExternalLocationType,
    ExternalReference,
    ExternalReferenceAvailability,
    ParentReference,
    RegistryEntry,
    RegistryStatus,
    RetentionStatus,
    StorageClass,
    VerificationDiagnosticCode,
    VerificationResult,
    allowed_formats_for_category,
    canonical_timestamp,
    descriptor_to_artifact_id,
    is_legal_retention_transition,
    is_legal_storage_transition,
    validate_artifact_id,
    validate_canonical_timestamp,
    validate_sha256_digest,
)
from expertforge.artifacts.paths import PathSafetyError
from expertforge.artifacts.registry import (
    RegistryError,
    RegistryScanResult,
    allocate_and_append,
    compute_next_sequence,
    load_registry,
    registry_path_for_attempt,
    scan_registry,
)
from expertforge.artifacts.secure_store import ArtifactStore
from expertforge.artifacts.store import (
    BLOCK_SIZE,
    ArtifactSealedError,
    ArtifactStoreError,
    RegistryReconciliation,
    sha256_stream,
)

__all__ = [
    "ARTIFACT_BUNDLE_SCHEMA",
    "ARTIFACT_BUNDLE_SCHEMA_VERSION",
    "ARTIFACT_FORMAT_VERSION",
    "ARTIFACT_ID_PATTERN",
    "ARTIFACT_REGISTRY_FORMAT_VERSION",
    "ARTIFACT_RECORD_SCHEMA",
    "ARTIFACT_RECORD_SCHEMA_VERSION",
    "BLOCK_SIZE",
    "EXTERNAL_REFERENCE_SCHEMA",
    "EXTERNAL_REFERENCE_SCHEMA_VERSION",
    "MAX_REGISTRY_ENTRY_BYTES",
    "ArtifactCategory",
    "ArtifactConflictError",
    "ArtifactDescriptor",
    "ArtifactFormat",
    "ArtifactNotFoundError",
    "ArtifactRecord",
    "ArtifactSealedError",
    "ArtifactStore",
    "ArtifactStoreError",
    "EntryKind",
    "ExternalLocationType",
    "ExternalReference",
    "ExternalReferenceAvailability",
    "ParentReference",
    "PathSafetyError",
    "RegistryEntry",
    "RegistryError",
    "RegistryReconciliation",
    "RegistryScanResult",
    "RegistryStatus",
    "RetentionStatus",
    "StorageClass",
    "VerificationDiagnosticCode",
    "VerificationResult",
    "allocate_and_append",
    "allowed_formats_for_category",
    "canonical_timestamp",
    "compute_next_sequence",
    "descriptor_to_artifact_id",
    "is_legal_retention_transition",
    "is_legal_storage_transition",
    "load_registry",
    "registry_path_for_attempt",
    "scan_registry",
    "sha256_stream",
    "validate_artifact_id",
    "validate_canonical_timestamp",
    "validate_sha256_digest",
]
