"""Frozen artifact models, closed domains, and the cross-field transition matrix.

Issue #10 normative contract (design comment ``5138481783`` ratified by
amendment ``5138634449``). All persisted models are deeply immutable, frozen,
``extra="forbid"``, strict Pydantic v2.

The amendment overrides the original design on the following points, which are
reflected here exactly:

- **A.** Full semantic artifact IDs ``artifact-v1-sha256-<64hex>`` derived from a
  canonical immutable descriptor (not a 16-hex content prefix).
- **B.** Fully-qualified ``ParentReference(run_id, attempt_id, artifact_id)``.
- **D.** Sequenced registry-entry envelopes with ``entry_kind`` and
  ``registry_format_version``.
- **F.** Separated ``StorageClass`` (canonical_local|local_cache|external|
  metadata_only) and ``RetentionStatus`` (retained|pending_transfer|
  externally_retained|expired|missing|verification_failed).
- **G.** Identity-bound ``ExternalReference``; credential-bearing locations are
  rejected (not sanitized).
- **J.** Closed ``ArtifactFormat`` domain (json|jsonl|text|binary|tar) and a
  typed ``VerificationResult``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "ARTIFACT_BUNDLE_SCHEMA",
    "ARTIFACT_BUNDLE_SCHEMA_VERSION",
    "ARTIFACT_FORMAT_VERSION",
    "ARTIFACT_ID_PATTERN",
    "ARTIFACT_REGISTRY_FORMAT_VERSION",
    "EXTERNAL_REFERENCE_SCHEMA",
    "EXTERNAL_REFERENCE_SCHEMA_VERSION",
    "ARTIFACT_RECORD_SCHEMA",
    "ARTIFACT_RECORD_SCHEMA_VERSION",
    "MAX_REGISTRY_ENTRY_BYTES",
    "ArtifactCategory",
    "ArtifactConflictError",
    "ArtifactDescriptor",
    "ArtifactFormat",
    "ArtifactNotFoundError",
    "ArtifactRecord",
    "EntryKind",
    "ExternalLocationType",
    "ExternalReference",
    "ExternalReferenceAvailability",
    "ParentReference",
    "RegistryEntry",
    "RegistryStatus",
    "RetentionStatus",
    "StorageClass",
    "VerificationDiagnosticCode",
    "VerificationResult",
    "allowed_formats_for_category",
    "canonical_timestamp",
    "descriptor_to_artifact_id",
    "is_legal_retention_transition",
    "is_legal_storage_transition",
    "validate_artifact_id",
    "validate_canonical_timestamp",
    "validate_sha256_digest",
]

# ---------------------------------------------------------------------------
# Format / schema versions
# ---------------------------------------------------------------------------

ARTIFACT_FORMAT_VERSION: int = 1
# The registry.jsonl envelope/framing version.
ARTIFACT_REGISTRY_FORMAT_VERSION: int = 1
# The immutable artifact.json bundle schema (self-contained per-bundle metadata).
ARTIFACT_BUNDLE_SCHEMA: Literal["expertforge.artifact-record"] = "expertforge.artifact-record"
ARTIFACT_BUNDLE_SCHEMA_VERSION: int = 1
# Legacy alias kept for callers that read the design's ``schema`` name; identical
# canonical value.
ARTIFACT_RECORD_SCHEMA: Literal["expertforge.artifact-record"] = ARTIFACT_BUNDLE_SCHEMA
ARTIFACT_RECORD_SCHEMA_VERSION: int = ARTIFACT_BUNDLE_SCHEMA_VERSION
EXTERNAL_REFERENCE_SCHEMA: Literal["expertforge.external-reference"] = "expertforge.external-reference"
EXTERNAL_REFERENCE_SCHEMA_VERSION: int = 1

# A single registry entry must fit in this many bytes (excluding the newline).
MAX_REGISTRY_ENTRY_BYTES: int = 256 * 1024

# ---------------------------------------------------------------------------
# Closed domains
# ---------------------------------------------------------------------------

ArtifactCategory = Literal[
    "resolved_configuration",
    "provenance",
    "telemetry",
    "checkpoint",
    "experiment_manifest",
    "report",
    "generated_sample",
]

ArtifactFormat = Literal["json", "jsonl", "text", "binary", "tar"]

StorageClass = Literal["canonical_local", "local_cache", "external", "metadata_only"]

RetentionStatus = Literal[
    "retained",
    "pending_transfer",
    "externally_retained",
    "expired",
    "missing",
    "verification_failed",
]

# Registry entry kinds cover the full per-artifact lifecycle.
EntryKind = Literal[
    "initial_publication",
    "external_registration",
    "retention_transition",
    "verification_transition",
]

ExternalLocationType = Literal["uri", "filesystem_path", "object_store"]

ExternalReferenceAvailability = Literal["available", "unavailable", "verification_failed"]

RegistryStatus = Literal["complete", "incomplete", "corrupt"]

VerificationDiagnosticCode = Literal[
    "verified",
    "missing_bundle",
    "missing_content",
    "digest_mismatch",
    "size_mismatch",
    "not_regular_file",
    "bundle_metadata_mismatch",
    "external_reference_not_verified",
]

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_HEX64 = r"[0-9a-f]{64}"
ARTIFACT_ID_PATTERN = re.compile(r"^artifact-v1-sha256-[0-9a-f]{64}$")
_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_CANONICAL_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")

# Schema-version-1 category/format compatibility matrix. Adding a persisted
# format requires a schema amendment/version.
_CATEGORY_FORMATS: dict[ArtifactCategory, frozenset[ArtifactFormat]] = {
    "resolved_configuration": frozenset({"json"}),
    "provenance": frozenset({"json"}),
    "telemetry": frozenset({"jsonl"}),
    "checkpoint": frozenset({"binary", "tar"}),
    "experiment_manifest": frozenset({"json"}),
    "report": frozenset({"json", "text"}),
    "generated_sample": frozenset({"json", "jsonl", "text", "binary", "tar"}),
}


def allowed_formats_for_category(category: ArtifactCategory) -> frozenset[ArtifactFormat]:
    """The closed set of formats a category may carry at schema version 1."""
    return _CATEGORY_FORMATS[category]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_artifact_id(value: str) -> None:
    """Raise ValueError if ``value`` is not a valid ``artifact-v1-sha256-<hex>``."""
    if not ARTIFACT_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"Invalid artifact_id {value!r}; must match {ARTIFACT_ID_PATTERN.pattern}."
        )


def validate_sha256_digest(value: str) -> None:
    """Raise ValueError unless ``value`` is ``sha256:<64 lowercase hex>``."""
    if not _SHA256_DIGEST_PATTERN.fullmatch(value):
        raise ValueError(
            f"digest must be 'sha256:<64 lowercase hex>'; got {value!r}."
        )


def canonical_timestamp(value: datetime) -> str:
    """Format a timezone-aware UTC datetime as fixed-width RFC 3339 with 6 fractional digits."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"timestamp must be timezone-aware; got naive {value!r}.")
    if value.tzinfo.utcoffset(value) != timedelta(0):
        raise ValueError(
            f"timestamp must be UTC (offset 0); got offset {value.tzinfo.utcoffset(value)!r}."
        )
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond:06d}Z"


def validate_canonical_timestamp(value: str) -> None:
    """Raise ValueError unless ``value`` is fixed-width canonical UTC."""
    if not _CANONICAL_TIMESTAMP_PATTERN.fullmatch(value):
        raise ValueError(
            f"timestamp must be fixed-width canonical UTC "
            f"(YYYY-MM-DDTHH:MM:SS.ffffffZ); got {value!r}."
        )


# ---------------------------------------------------------------------------
# Transition matrices (amendment F)
# ---------------------------------------------------------------------------

# Storage transitions. ``temporary`` is never persisted.
_STORAGE_TRANSITIONS: dict[StorageClass, frozenset[StorageClass]] = {
    "canonical_local": frozenset({"canonical_local", "local_cache", "external", "metadata_only"}),
    "local_cache": frozenset({"local_cache", "canonical_local", "external", "metadata_only"}),
    "external": frozenset({"external", "metadata_only"}),
    "metadata_only": frozenset({"metadata_only"}),
}


def is_legal_storage_transition(src: StorageClass, dst: StorageClass) -> bool:
    """Closed storage transition matrix (amendment F). ``temporary`` is never persisted."""
    return dst in _STORAGE_TRANSITIONS[src]


# Retention transitions. ``verification_failed`` records a later failed
# verification of an already-registered artifact and is not a route for
# accepting invalid bytes initially.
_RETENTION_TRANSITIONS: dict[RetentionStatus, frozenset[RetentionStatus]] = {
    # Local canonical artifacts reach externally_retained only via
    # pending_transfer (amendment F: transfer moves through pending_transfer).
    "retained": frozenset(
        {"retained", "pending_transfer", "expired", "missing", "verification_failed"}
    ),
    "pending_transfer": frozenset(
        {"pending_transfer", "retained", "externally_retained", "expired", "missing",
         "verification_failed"}
    ),
    "externally_retained": frozenset(
        {"externally_retained", "expired", "missing", "verification_failed"}
    ),
    "expired": frozenset({"expired", "missing"}),
    "missing": frozenset({"missing", "verification_failed"}),
    "verification_failed": frozenset({"verification_failed", "retained", "missing"}),
}


def is_legal_retention_transition(src: RetentionStatus, dst: RetentionStatus) -> bool:
    """Closed retention transition matrix (amendment F)."""
    return dst in _RETENTION_TRANSITIONS[src]


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class ArtifactConflictError(Exception):
    """Raised when an artifact path exists with conflicting immutable metadata,
    when content/size do not match an expected record, or when identity binding
    is inconsistent."""


class ArtifactNotFoundError(Exception):
    """Raised when an artifact_id has no canonical bundle or registry entry."""


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
# Parent reference (amendment B)
# ---------------------------------------------------------------------------


class ParentReference(_FrozenModel):
    """A fully-qualified cross-attempt parent artifact reference (amendment B).

    Same-attempt parents still persist the full triple; a convenience
    constructor (:meth:`same_attempt`) builds the common case. Parent
    references are evidence links and are never silently rewritten.
    """

    run_id: str = Field(..., min_length=1)
    attempt_id: str = Field(..., min_length=1)
    artifact_id: str = Field(..., min_length=1)

    @field_validator("artifact_id")
    @classmethod
    def _validate_artifact_id(cls, v: str) -> str:
        validate_artifact_id(v)
        return v

    @classmethod
    def same_attempt(
        cls, *, run_id: str, attempt_id: str, artifact_id: str
    ) -> ParentReference:
        """Build a parent reference within the same run/attempt."""
        return cls(run_id=run_id, attempt_id=attempt_id, artifact_id=artifact_id)


# ---------------------------------------------------------------------------
# Artifact descriptor + semantic artifact ID (amendment A)
# ---------------------------------------------------------------------------


class ArtifactDescriptor(_FrozenModel):
    """The canonical immutable descriptor that is hashed to produce ``artifact_id``.

    The descriptor includes at least: registry/artifact schema version, run ID,
    attempt ID, specification fingerprint, category, format, format version,
    content digest, byte size, producing component, and parent reference
    (amendment A). Equal bytes with different category/format/producer/parent/
    run/attempt are distinct logical artifacts.

    Identity fields are bound from the supplied :class:`AttemptIdentityRecord`;
    callers do not provide them independently — the store constructs this from
    the identity plus publication inputs.
    """

    schema_version: int = Field(default=ARTIFACT_BUNDLE_SCHEMA_VERSION)
    run_id: str = Field(..., min_length=1)
    attempt_id: str = Field(..., min_length=1)
    specification_fingerprint: str = Field(..., min_length=1)
    category: ArtifactCategory
    format: ArtifactFormat
    format_version: int = Field(..., ge=1)
    content_digest: str = Field(..., min_length=1)
    byte_size: int = Field(..., ge=0)
    producing_component: str = Field(..., min_length=1)
    parent: ParentReference | None = None

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: int) -> int:
        if v != ARTIFACT_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported artifact descriptor schema_version {v}; "
                f"this version understands {ARTIFACT_BUNDLE_SCHEMA_VERSION}."
            )
        return v

    @field_validator("content_digest")
    @classmethod
    def _validate_content_digest(cls, v: str) -> str:
        validate_sha256_digest(v)
        return v

    @field_validator("format")
    @classmethod
    def _validate_format_for_category(cls, v: ArtifactFormat) -> ArtifactFormat:
        # Cross-field check happens in model_validator (needs category). Here we
        # only assert membership of the closed domain, which Literal already
        # enforces; kept explicit for clarity.
        return v

    @model_validator(mode="after")
    def _enforce_invariants(self) -> ArtifactDescriptor:
        allowed = allowed_formats_for_category(self.category)
        if self.format not in allowed:
            raise ValueError(
                f"format {self.format!r} is not allowed for category "
                f"{self.category!r}; allowed={sorted(allowed)}."
            )
        return self

    def canonical_bytes(self) -> bytes:
        """Compact, sorted-key, UTF-8, non-finite-prohibiting JSON of the descriptor."""
        return _canonical_json_bytes(self.to_descriptor_dict())

    def to_descriptor_dict(self) -> dict[str, Any]:
        """The canonical hashable mapping (sorted on serialization)."""
        return {
            "schema": "expertforge.artifact-descriptor",
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "specification_fingerprint": self.specification_fingerprint,
            "category": self.category,
            "format": self.format,
            "format_version": self.format_version,
            "content_digest": self.content_digest,
            "byte_size": self.byte_size,
            "producing_component": self.producing_component,
            "parent": (None if self.parent is None else self.parent.model_dump(mode="json")),
        }


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def descriptor_to_artifact_id(descriptor: ArtifactDescriptor) -> str:
    """Compute the canonical ``artifact-v1-sha256-<64hex>`` ID from a descriptor."""
    digest = hashlib.sha256(descriptor.canonical_bytes()).hexdigest()
    return f"artifact-v1-sha256-{digest}"


# ---------------------------------------------------------------------------
# ArtifactRecord (immutable per-bundle metadata; amendment F cross-fields)
# ---------------------------------------------------------------------------


class ArtifactRecord(_FrozenModel):
    """Self-contained immutable per-bundle metadata.

    The persisted schema name is ``expertforge.artifact-record`` (aliased from
    ``schema_name`` to avoid shadowing :meth:`BaseModel.schema`). The
    ``artifact_id`` is the full semantic digest (amendment A), bound from the
    descriptor; callers may not set identity fields independently of the
    descriptor that produced them.
    """

    schema_name: str = Field(
        default=ARTIFACT_BUNDLE_SCHEMA, alias="schema"
    )
    schema_version: int = Field(default=ARTIFACT_BUNDLE_SCHEMA_VERSION)
    artifact_id: str = Field(..., min_length=1)
    category: ArtifactCategory
    format: ArtifactFormat
    format_version: int = Field(..., ge=1)
    byte_size: int = Field(..., ge=0)
    content_digest: str = Field(..., min_length=1)
    producing_component: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    attempt_id: str = Field(..., min_length=1)
    specification_fingerprint: str = Field(..., min_length=1)
    created_at_utc: datetime
    relative_path: str = Field(..., min_length=1)
    parent: ParentReference | None = None
    storage_class: StorageClass
    retention: RetentionStatus

    @field_validator("schema_name")
    @classmethod
    def _validate_schema_name(cls, v: str) -> str:
        if v != ARTIFACT_BUNDLE_SCHEMA:
            raise ValueError(
                f"schema must be {ARTIFACT_BUNDLE_SCHEMA!r}; got {v!r}."
            )
        return v

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: int) -> int:
        if v != ARTIFACT_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported artifact-record schema_version {v}; "
                f"this version understands {ARTIFACT_BUNDLE_SCHEMA_VERSION}."
            )
        return v

    @field_validator("artifact_id")
    @classmethod
    def _validate_artifact_id(cls, v: str) -> str:
        validate_artifact_id(v)
        return v

    @field_validator("content_digest")
    @classmethod
    def _validate_content_digest(cls, v: str) -> str:
        validate_sha256_digest(v)
        return v

    @field_validator("created_at_utc")
    @classmethod
    def _validate_created_at(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError(f"created_at_utc must be timezone-aware; got naive {v!r}.")
        if v.tzinfo.utcoffset(v) != timedelta(0):
            raise ValueError(
                f"created_at_utc must be UTC (offset 0); got offset {v.tzinfo.utcoffset(v)!r}."
            )
        return v

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, v: str) -> str:
        # POSIX repo-relative, no traversal, no backslashes, no absolute form.
        normalized = v.replace("\\", "/")
        if not normalized:
            raise ValueError("relative_path must be non-empty.")
        if normalized.startswith("/"):
            raise ValueError(f"relative_path {v!r} must be relative, not absolute.")
        if re.match(r"^[A-Za-z]:/", normalized):
            raise ValueError(f"relative_path {v!r} must be relative, not absolute.")
        parts = normalized.split("/")
        if any(part in ("..", ".") for part in parts):
            raise ValueError(f"relative_path {v!r} contains a '.' or '..' component.")
        if any(part == "" for part in parts):
            raise ValueError(f"relative_path {v!r} contains an empty path component.")
        if normalized != v:
            raise ValueError(f"relative_path {v!r} must use forward slashes only.")
        return v

    @model_validator(mode="after")
    def _enforce_cross_fields(self) -> ArtifactRecord:
        # Category/format compatibility (schema version 1).
        allowed = allowed_formats_for_category(self.category)
        if self.format not in allowed:
            raise ValueError(
                f"format {self.format!r} is not allowed for category "
                f"{self.category!r}; allowed={sorted(allowed)}."
            )
        # Cross-field storage/retention matrix (amendment F).
        self._check_storage_retention(self.storage_class, self.retention)
        return self

    @staticmethod
    def _check_storage_retention(storage: StorageClass, retention: RetentionStatus) -> None:
        # Newly published local artifacts are canonical_local + retained.
        # external requires externally_retained or pending_transfer.
        # metadata_only pairs with missing|expired|verification_failed.
        # verification_failed is never paired with a freshly-accepted payload.
        legal: set[tuple[StorageClass, RetentionStatus]] = {
            ("canonical_local", "retained"),
            ("canonical_local", "pending_transfer"),
            ("canonical_local", "externally_retained"),
            ("canonical_local", "expired"),
            ("canonical_local", "missing"),
            ("canonical_local", "verification_failed"),
            ("local_cache", "retained"),
            ("local_cache", "pending_transfer"),
            ("external", "pending_transfer"),
            ("external", "externally_retained"),
            ("external", "missing"),
            ("external", "verification_failed"),
            ("metadata_only", "missing"),
            ("metadata_only", "expired"),
            ("metadata_only", "verification_failed"),
        }
        if (storage, retention) not in legal:
            raise ValueError(
                f"illegal storage/retention combination "
                f"({storage!r}, {retention!r})."
            )

    @classmethod
    def from_descriptor(
        cls,
        descriptor: ArtifactDescriptor,
        *,
        created_at_utc: datetime,
        relative_path: str,
        storage_class: StorageClass = "canonical_local",
        retention: RetentionStatus = "retained",
    ) -> ArtifactRecord:
        """Build a record whose identity fields are bound from the descriptor.

        Callers may not supply run/attempt/fingerprint/artifact_id independently
        of the descriptor that produced them.
        """
        artifact_id = descriptor_to_artifact_id(descriptor)
        return cls(
            artifact_id=artifact_id,
            category=descriptor.category,
            format=descriptor.format,
            format_version=descriptor.format_version,
            byte_size=descriptor.byte_size,
            content_digest=descriptor.content_digest,
            producing_component=descriptor.producing_component,
            run_id=descriptor.run_id,
            attempt_id=descriptor.attempt_id,
            specification_fingerprint=descriptor.specification_fingerprint,
            created_at_utc=created_at_utc,
            relative_path=relative_path,
            parent=descriptor.parent,
            storage_class=storage_class,
            retention=retention,
        )


# ---------------------------------------------------------------------------
# ExternalReference (amendment G)
# ---------------------------------------------------------------------------


class ExternalReference(_FrozenModel):
    """Identity-bound external reference (amendment G).

    Must include run ID, attempt ID, specification fingerprint, artifact ID,
    expected digest, expected size, and the artifact's category/format/version
    binding. A standalone location without an artifact record is invalid; this
    model is always persisted alongside the matching immutable
    :class:`ArtifactRecord` in one registry entry.

    Credential-bearing locations are **rejected**, not sanitized. For version 1:
    URI/object-store identifiers forbid userinfo, query strings, and fragments;
    filesystem locations must be canonical portable relative form tied to an
    explicit ``external_root_id``.
    """

    schema_name: str = Field(default=EXTERNAL_REFERENCE_SCHEMA, alias="schema")
    schema_version: int = Field(default=EXTERNAL_REFERENCE_SCHEMA_VERSION)
    artifact_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    attempt_id: str = Field(..., min_length=1)
    specification_fingerprint: str = Field(..., min_length=1)
    category: ArtifactCategory
    format: ArtifactFormat
    format_version: int = Field(..., ge=1)
    location_type: ExternalLocationType
    location: str = Field(..., min_length=1)
    external_root_id: str | None = None
    expected_digest: str = Field(..., min_length=1)
    expected_byte_size: int = Field(..., ge=0)
    availability: ExternalReferenceAvailability = "unavailable"
    verified_at_utc: datetime | None = None

    @field_validator("schema_name")
    @classmethod
    def _validate_schema_name(cls, v: str) -> str:
        if v != EXTERNAL_REFERENCE_SCHEMA:
            raise ValueError(
                f"schema must be {EXTERNAL_REFERENCE_SCHEMA!r}; got {v!r}."
            )
        return v

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: int) -> int:
        if v != EXTERNAL_REFERENCE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported external-reference schema_version {v}; "
                f"this version understands {EXTERNAL_REFERENCE_SCHEMA_VERSION}."
            )
        return v

    @field_validator("artifact_id")
    @classmethod
    def _validate_artifact_id(cls, v: str) -> str:
        validate_artifact_id(v)
        return v

    @field_validator("expected_digest")
    @classmethod
    def _validate_expected_digest(cls, v: str) -> str:
        validate_sha256_digest(v)
        return v

    @field_validator("verified_at_utc")
    @classmethod
    def _validate_verified_at(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return v
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError(f"verified_at_utc must be timezone-aware; got naive {v!r}.")
        if v.tzinfo.utcoffset(v) != timedelta(0):
            raise ValueError("verified_at_utc must be UTC (offset 0).")
        return v

    @model_validator(mode="after")
    def _enforce_location_safety(self) -> ExternalReference:
        from expertforge.artifacts.external import reject_credential_bearing_location

        reject_credential_bearing_location(
            location_type=self.location_type,
            location=self.location,
            external_root_id=self.external_root_id,
        )
        # Category/format compatibility for the referenced artifact.
        allowed = allowed_formats_for_category(self.category)
        if self.format not in allowed:
            raise ValueError(
                f"format {self.format!r} is not allowed for category "
                f"{self.category!r}; allowed={sorted(allowed)}."
            )
        return self


# ---------------------------------------------------------------------------
# Registry entry envelope (amendment D)
# ---------------------------------------------------------------------------


class RegistryEntry(_FrozenModel):
    """One sequenced registry-entry envelope line (amendment D).

    ``payload`` is a JSON object carrying either an immutable
    :class:`ArtifactRecord` (or :class:`ExternalReference`) and is validated by
    the registry loader against closed transition rules. The envelope is
    strictly deterministic on serialization.
    """

    registry_format_version: int = Field(default=ARTIFACT_REGISTRY_FORMAT_VERSION)
    sequence: int = Field(..., ge=0)
    run_id: str = Field(..., min_length=1)
    attempt_id: str = Field(..., min_length=1)
    specification_fingerprint: str = Field(..., min_length=1)
    recorded_at_utc: datetime
    entry_kind: EntryKind
    payload: Mapping[str, Any]

    @field_validator("registry_format_version")
    @classmethod
    def _validate_registry_version(cls, v: int) -> int:
        if v != ARTIFACT_REGISTRY_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported registry_format_version {v}; "
                f"this version understands {ARTIFACT_REGISTRY_FORMAT_VERSION}."
            )
        return v

    @field_validator("recorded_at_utc")
    @classmethod
    def _validate_recorded_at(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError(f"recorded_at_utc must be timezone-aware; got naive {v!r}.")
        if v.tzinfo.utcoffset(v) != timedelta(0):
            raise ValueError("recorded_at_utc must be UTC (offset 0).")
        return v

    @field_validator("payload")
    @classmethod
    def _validate_payload(cls, v: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(v, Mapping):
            raise ValueError("payload must be a JSON object.")
        # Materialize to a plain dict so the envelope is JSON-serializable and
        # not a mutable alias of caller state.
        return dict(v)

    def to_deterministic_json(self) -> bytes:
        """Compact, sorted-key, UTF-8, non-finite-prohibiting JSON (one line)."""
        return _canonical_json_bytes(self.to_envelope_dict())

    def to_envelope_dict(self) -> dict[str, Any]:
        return {
            "registry_format_version": self.registry_format_version,
            "sequence": self.sequence,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "specification_fingerprint": self.specification_fingerprint,
            "recorded_at_utc": canonical_timestamp(self.recorded_at_utc),
            "entry_kind": self.entry_kind,
            "payload": dict(self.payload),
        }


# ---------------------------------------------------------------------------
# Verification result (amendment J)
# ---------------------------------------------------------------------------


class VerificationResult(_FrozenModel):
    """Typed verification result (amendment J).

    A bare boolean is insufficient as the authoritative interface. The
    :attr:`status` is True only when the diagnostic is ``verified``.
    """

    status: bool
    artifact_id: str = Field(..., min_length=1)
    diagnostic_code: VerificationDiagnosticCode
    observed_digest: str | None = None
    observed_size: int | None = None

    @field_validator("artifact_id")
    @classmethod
    def _validate_artifact_id(cls, v: str) -> str:
        validate_artifact_id(v)
        return v

    @field_validator("observed_digest")
    @classmethod
    def _validate_observed_digest(cls, v: str | None) -> str | None:
        if v is None:
            return v
        validate_sha256_digest(v)
        return v

    @model_validator(mode="after")
    def _enforce_consistency(self) -> VerificationResult:
        if self.status and self.diagnostic_code != "verified":
            raise ValueError(
                f"status=True requires diagnostic_code='verified'; got {self.diagnostic_code!r}."
            )
        if not self.status and self.diagnostic_code == "verified":
            raise ValueError("diagnostic_code='verified' requires status=True.")
        return self
