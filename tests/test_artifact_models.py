"""Issue #10 artifact model tests (amendments A, B, D, F, G, J).

Covers: model immutability, closed Literal domains, cross-field validators,
the storage/retention transition matrix, full semantic artifact IDs, fully
qualified parent references, typed VerificationResult, and the closed
ArtifactFormat/category compatibility matrix.

Pure-unit tests; no filesystem access.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from expertforge.artifacts.models import (
    ARTIFACT_ID_PATTERN,
    ARTIFACT_REGISTRY_FORMAT_VERSION,
    EXTERNAL_REFERENCE_SCHEMA,
    MAX_REGISTRY_ENTRY_BYTES,
    ArtifactDescriptor,
    ArtifactRecord,
    ExternalReference,
    ParentReference,
    RegistryEntry,
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
from tests._artifact_fixtures import VALID_ATTEMPT, VALID_FP, VALID_RUN, make_identity

_VALID_DIGEST = "sha256:" + "a" * 64
_VALID_ARTIFACT_ID = "artifact-v1-sha256-" + "a" * 64
_VALID_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _descriptor(**overrides: object) -> ArtifactDescriptor:
    base: dict[str, object] = {
        "run_id": VALID_RUN,
        "attempt_id": VALID_ATTEMPT,
        "specification_fingerprint": VALID_FP,
        "category": "report",
        "format": "json",
        "format_version": 1,
        "content_digest": _VALID_DIGEST,
        "byte_size": 10,
        "producing_component": "training",
    }
    base.update(overrides)
    return ArtifactDescriptor(**base)  # type: ignore[arg-type]


class TestClosedDomains:
    def test_category_domain_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            _descriptor(category="not_a_category")

    def test_format_domain_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            _descriptor(format="yaml")

    def test_storage_domain_rejects_temporary(self) -> None:
        # temporary is an internal operational state and is never persisted.
        with pytest.raises(ValidationError):
            ArtifactRecord(
                artifact_id=_VALID_ARTIFACT_ID,
                category="report",
                format="json",
                format_version=1,
                byte_size=1,
                content_digest=_VALID_DIGEST,
                producing_component="training",
                run_id=VALID_RUN,
                attempt_id=VALID_ATTEMPT,
                specification_fingerprint=VALID_FP,
                created_at_utc=_VALID_TS,
                relative_path="artifacts/report/x",
                storage_class="temporary",  # type: ignore[arg-type]
                retention="retained",
            )

    def test_storage_domain_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRecord(
                artifact_id=_VALID_ARTIFACT_ID,
                category="report",
                format="json",
                format_version=1,
                byte_size=1,
                content_digest=_VALID_DIGEST,
                producing_component="training",
                run_id=VALID_RUN,
                attempt_id=VALID_ATTEMPT,
                specification_fingerprint=VALID_FP,
                created_at_utc=_VALID_TS,
                relative_path="artifacts/report/x",
                storage_class="cloud",  # type: ignore[arg-type]
                retention="retained",
            )

    def test_no_directory_format(self) -> None:
        # Amendment I: Issue #10 accepts bytes or regular files only.
        with pytest.raises(ValidationError):
            _descriptor(category="checkpoint", format="directory")


class TestCategoryFormatMatrix:
    @pytest.mark.parametrize(
        ("category", "expected"),
        [
            ("resolved_configuration", {"json"}),
            ("provenance", {"json"}),
            ("telemetry", {"jsonl"}),
            ("checkpoint", {"binary", "tar"}),
            ("experiment_manifest", {"json"}),
            ("report", {"json", "text"}),
            ("generated_sample", {"json", "jsonl", "text", "binary", "tar"}),
        ],
    )
    def test_allowed_formats(self, category: str, expected: set[str]) -> None:
        assert set(allowed_formats_for_category(category)) == expected  # type: ignore[arg-type]

    def test_telemetry_rejects_json(self) -> None:
        with pytest.raises(ValidationError):
            _descriptor(category="telemetry", format="json")

    def test_checkpoint_rejects_json(self) -> None:
        with pytest.raises(ValidationError):
            _descriptor(category="checkpoint", format="json")


class TestImmutability:
    def test_descriptor_is_frozen(self) -> None:
        d = _descriptor()
        with pytest.raises(ValidationError):
            d.category = "provenance"  # type: ignore[misc]

    def test_record_is_frozen(self) -> None:
        rec = ArtifactRecord.from_descriptor(
            _descriptor(), created_at_utc=_VALID_TS, relative_path="artifacts/report/a"
        )
        with pytest.raises(ValidationError):
            rec.retention = "expired"  # type: ignore[misc]

    def test_parent_reference_is_frozen(self) -> None:
        p = ParentReference(
            run_id=VALID_RUN,
            attempt_id=VALID_ATTEMPT,
            artifact_id=_VALID_ARTIFACT_ID,
        )
        with pytest.raises(ValidationError):
            p.artifact_id = "other"  # type: ignore[misc]

    def test_verification_result_is_frozen(self) -> None:
        v = VerificationResult(
            status=True,
            artifact_id=_VALID_ARTIFACT_ID,
            diagnostic_code="verified",
            observed_digest=_VALID_DIGEST,
            observed_size=1,
        )
        with pytest.raises(ValidationError):
            v.status = False  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ParentReference(  # type: ignore[call-arg]
                run_id=VALID_RUN,
                attempt_id=VALID_ATTEMPT,
                artifact_id=_VALID_ARTIFACT_ID,
                extra="no",
            )


class TestSemanticArtifactId:
    def test_id_pattern(self) -> None:
        d = _descriptor()
        aid = descriptor_to_artifact_id(d)
        assert ARTIFACT_ID_PATTERN.fullmatch(aid)

    def test_same_inputs_produce_same_id(self) -> None:
        assert descriptor_to_artifact_id(_descriptor()) == descriptor_to_artifact_id(_descriptor())

    def test_different_category_different_id(self) -> None:
        a = descriptor_to_artifact_id(_descriptor(category="report"))
        b = descriptor_to_artifact_id(_descriptor(category="provenance"))
        assert a != b

    def test_different_producer_different_id(self) -> None:
        a = descriptor_to_artifact_id(_descriptor(producing_component="training"))
        b = descriptor_to_artifact_id(_descriptor(producing_component="eval"))
        assert a != b

    def test_different_run_different_id(self) -> None:
        a = descriptor_to_artifact_id(_descriptor(run_id="run-1"))
        b = descriptor_to_artifact_id(_descriptor(run_id="run-2"))
        assert a != b

    def test_different_parent_different_id(self) -> None:
        p1 = ParentReference(
            run_id=VALID_RUN,
            attempt_id=VALID_ATTEMPT,
            artifact_id=_VALID_ARTIFACT_ID,
        )
        p2 = ParentReference(
            run_id="run-other",
            attempt_id=VALID_ATTEMPT,
            artifact_id=_VALID_ARTIFACT_ID,
        )
        a = descriptor_to_artifact_id(_descriptor(parent=p1))
        b = descriptor_to_artifact_id(_descriptor(parent=p2))
        assert a != b

    def test_equal_bytes_different_descriptor_distinct(self) -> None:
        # Amendment A/L: same bytes (same content_digest) but different
        # immutable descriptor fields produce distinct logical artifact IDs.
        a = descriptor_to_artifact_id(_descriptor(category="report"))
        b = descriptor_to_artifact_id(_descriptor(category="generated_sample"))
        assert a != b

    def test_validate_artifact_id_rejects_prefix(self) -> None:
        # 16-hex prefixes are explicitly rejected.
        with pytest.raises(ValueError):
            validate_artifact_id("report-abcdef0123456789")

    def test_validate_artifact_id_rejects_short(self) -> None:
        with pytest.raises(ValueError):
            validate_artifact_id("artifact-v1-sha256-" + "a" * 16)


class TestParentReference:
    def test_fully_qualified_required(self) -> None:
        p = ParentReference(
            run_id=VALID_RUN,
            attempt_id=VALID_ATTEMPT,
            artifact_id=_VALID_ARTIFACT_ID,
        )
        assert p.run_id == VALID_RUN
        assert p.attempt_id == VALID_ATTEMPT

    def test_invalid_artifact_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ParentReference(run_id=VALID_RUN, attempt_id=VALID_ATTEMPT, artifact_id="not-an-id")

    def test_same_attempt_constructor(self) -> None:
        p = ParentReference.same_attempt(
            run_id=VALID_RUN,
            attempt_id=VALID_ATTEMPT,
            artifact_id=_VALID_ARTIFACT_ID,
        )
        assert p.run_id == VALID_RUN


class TestArtifactRecordCrossFields:
    def _record(self, **overrides: object) -> ArtifactRecord:
        d = _descriptor(**overrides)
        return ArtifactRecord.from_descriptor(
            d, created_at_utc=_VALID_TS, relative_path="artifacts/report/x"
        )

    def test_default_storage_retention(self) -> None:
        rec = self._record()
        assert rec.storage_class == "canonical_local"
        assert rec.retention == "retained"

    def test_external_requires_externally_retained_or_pending(self) -> None:
        # legal
        ArtifactRecord._check_storage_retention("external", "externally_retained")
        ArtifactRecord._check_storage_retention("external", "pending_transfer")
        # illegal
        with pytest.raises(ValueError):
            ArtifactRecord._check_storage_retention("external", "retained")

    def test_metadata_only_requires_missing_expired_or_failed(self) -> None:
        for ok in ("missing", "expired", "verification_failed"):
            ArtifactRecord._check_storage_retention("metadata_only", ok)
        with pytest.raises(ValueError):
            ArtifactRecord._check_storage_retention("metadata_only", "retained")

    def test_canonical_local_allows_retained(self) -> None:
        ArtifactRecord._check_storage_retention("canonical_local", "retained")

    def test_relative_path_rejects_traversal(self) -> None:
        d = _descriptor()
        with pytest.raises(ValidationError):
            ArtifactRecord.from_descriptor(
                d,
                created_at_utc=_VALID_TS,
                relative_path="../escape",
            )

    def test_relative_path_rejects_absolute(self) -> None:
        d = _descriptor()
        with pytest.raises(ValidationError):
            ArtifactRecord.from_descriptor(d, created_at_utc=_VALID_TS, relative_path="/etc/passwd")

    def test_relative_path_rejects_backslashes(self) -> None:
        d = _descriptor()
        with pytest.raises(ValidationError):
            ArtifactRecord.from_descriptor(
                d, created_at_utc=_VALID_TS, relative_path="artifacts\\report\\x"
            )

    def test_relative_path_rejects_dot_component(self) -> None:
        d = _descriptor()
        with pytest.raises(ValidationError):
            ArtifactRecord.from_descriptor(
                d, created_at_utc=_VALID_TS, relative_path="artifacts/./report"
            )

    def test_naive_created_at_rejected(self) -> None:
        d = _descriptor()
        with pytest.raises(ValidationError):
            ArtifactRecord.from_descriptor(
                d,
                created_at_utc=datetime(2026, 1, 1),  # naive
                relative_path="artifacts/report/x",
            )

    def test_non_utc_created_at_rejected(self) -> None:
        d = _descriptor()
        non_utc = timezone(timedelta(hours=2))
        with pytest.raises(ValidationError):
            ArtifactRecord.from_descriptor(
                d,
                created_at_utc=datetime(2026, 1, 1, tzinfo=non_utc),
                relative_path="artifacts/report/x",
            )

    def test_unknown_schema_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRecord(
                schema_version=999,
                artifact_id=_VALID_ARTIFACT_ID,
                category="report",
                format="json",
                format_version=1,
                byte_size=1,
                content_digest=_VALID_DIGEST,
                producing_component="training",
                run_id=VALID_RUN,
                attempt_id=VALID_ATTEMPT,
                specification_fingerprint=VALID_FP,
                created_at_utc=_VALID_TS,
                relative_path="artifacts/report/x",
                storage_class="canonical_local",
                retention="retained",
            )

    def test_bad_content_digest_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRecord(
                artifact_id=_VALID_ARTIFACT_ID,
                category="report",
                format="json",
                format_version=1,
                byte_size=1,
                content_digest="md5:abc",
                producing_component="training",
                run_id=VALID_RUN,
                attempt_id=VALID_ATTEMPT,
                specification_fingerprint=VALID_FP,
                created_at_utc=_VALID_TS,
                relative_path="artifacts/report/x",
                storage_class="canonical_local",
                retention="retained",
            )


class TestTransitionMatrices:
    @pytest.mark.parametrize(
        ("src", "dst", "legal"),
        [
            ("canonical_local", "external", True),
            ("canonical_local", "metadata_only", True),
            ("canonical_local", "local_cache", True),
            ("external", "canonical_local", False),
            ("external", "metadata_only", True),
            ("metadata_only", "canonical_local", False),
            ("metadata_only", "metadata_only", True),
        ],
    )
    def test_storage_transitions(self, src: str, dst: str, legal: bool) -> None:
        assert is_legal_storage_transition(src, dst) is legal  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("src", "dst", "legal"),
        [
            ("retained", "missing", True),
            ("retained", "verification_failed", True),
            ("retained", "pending_transfer", True),
            ("missing", "retained", False),
            ("missing", "verification_failed", True),
            ("expired", "retained", False),
            ("expired", "missing", True),
            ("verification_failed", "retained", True),
            ("externally_retained", "retained", False),
        ],
    )
    def test_retention_transitions(self, src: str, dst: str, legal: bool) -> None:
        assert is_legal_retention_transition(src, dst) is legal  # type: ignore[arg-type]


class TestExternalReference:
    def _kwargs(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "artifact_id": _VALID_ARTIFACT_ID,
            "run_id": VALID_RUN,
            "attempt_id": VALID_ATTEMPT,
            "specification_fingerprint": VALID_FP,
            "category": "report",
            "format": "json",
            "format_version": 1,
            "location_type": "uri",
            "location": "https://example.com/artifact.json",
            "expected_digest": _VALID_DIGEST,
            "expected_byte_size": 10,
        }
        base.update(overrides)
        return base

    def _make(self, **overrides: object) -> ExternalReference:
        # model_validate is the canonical load path and accepts a plain dict,
        # avoiding **dict[str, object] splat type-checking against Literals.
        return ExternalReference.model_validate(self._kwargs(**overrides))

    def test_valid_uri_reference(self) -> None:
        ext = self._make()
        assert ext.schema_name == EXTERNAL_REFERENCE_SCHEMA
        assert ext.availability == "unavailable"

    def test_credential_userinfo_rejected(self) -> None:
        with pytest.raises((ValidationError, Exception)):
            self._make(location="https://user:pw@example.com/x")

    def test_query_string_rejected(self) -> None:
        with pytest.raises((ValidationError, Exception)):
            self._make(location="https://example.com/x?token=secret")

    def test_fragment_rejected(self) -> None:
        with pytest.raises((ValidationError, Exception)):
            self._make(location="https://example.com/x#frag")

    def test_file_scheme_rejected(self) -> None:
        with pytest.raises((ValidationError, Exception)):
            self._make(location="file:///etc/passwd")

    def test_filesystem_path_requires_external_root_id(self) -> None:
        with pytest.raises((ValidationError, Exception)):
            self._make(location_type="filesystem_path", location="data/x.bin")

    def test_filesystem_path_absolute_rejected(self) -> None:
        with pytest.raises((ValidationError, Exception)):
            self._make(
                location_type="filesystem_path",
                location="/etc/passwd",
                external_root_id="root-1",
            )

    def test_filesystem_path_relative_with_root_accepted(self) -> None:
        ext = self._make(
            location_type="filesystem_path",
            location="data/x.bin",
            external_root_id="root-1",
        )
        assert ext.external_root_id == "root-1"

    def test_default_availability_unavailable(self) -> None:
        ext = self._make()
        assert ext.availability == "unavailable"
        assert ext.verified_at_utc is None


class TestVerificationResult:
    def test_verified_result(self) -> None:
        v = VerificationResult(
            status=True,
            artifact_id=_VALID_ARTIFACT_ID,
            diagnostic_code="verified",
            observed_digest=_VALID_DIGEST,
            observed_size=1,
        )
        assert v.status is True

    def test_status_true_requires_verified_diagnostic(self) -> None:
        with pytest.raises(ValidationError):
            VerificationResult(
                status=True,
                artifact_id=_VALID_ARTIFACT_ID,
                diagnostic_code="digest_mismatch",
            )

    def test_verified_diagnostic_requires_status_true(self) -> None:
        with pytest.raises(ValidationError):
            VerificationResult(
                status=False,
                artifact_id=_VALID_ARTIFACT_ID,
                diagnostic_code="verified",
            )

    def test_observed_digest_validated(self) -> None:
        with pytest.raises(ValidationError):
            VerificationResult(
                status=False,
                artifact_id=_VALID_ARTIFACT_ID,
                diagnostic_code="digest_mismatch",
                observed_digest="bad",
            )


class TestRegistryEntryEnvelope:
    def _entry(self, **overrides: object) -> RegistryEntry:
        base: dict[str, object] = {
            "sequence": 0,
            "run_id": VALID_RUN,
            "attempt_id": VALID_ATTEMPT,
            "specification_fingerprint": VALID_FP,
            "recorded_at_utc": _VALID_TS,
            "entry_kind": "initial_publication",
            "payload": {"artifact_id": _VALID_ARTIFACT_ID},
        }
        base.update(overrides)
        return RegistryEntry(**base)  # type: ignore[arg-type]

    def test_envelope_carries_registry_version(self) -> None:
        e = self._entry()
        assert e.registry_format_version == ARTIFACT_REGISTRY_FORMAT_VERSION

    def test_envelope_is_frozen(self) -> None:
        e = self._entry()
        with pytest.raises(ValidationError):
            e.sequence = 5  # type: ignore[misc]

    def test_envelope_deterministic_json(self) -> None:
        a = self._entry().to_deterministic_json()
        b = self._entry().to_deterministic_json()
        assert a == b
        assert b"\n" not in a
        # canonical: no spaces after separators
        assert b", " not in a
        assert b": " not in a

    def test_recorded_at_utc_canonical_timestamp(self) -> None:
        e = self._entry()
        rendered = canonical_timestamp(e.recorded_at_utc)
        validate_canonical_timestamp(rendered)
        assert rendered == "2026-01-01T00:00:00.000000Z"

    def test_naive_recorded_at_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._entry(recorded_at_utc=datetime(2026, 1, 1))

    def test_max_entry_bytes_constant(self) -> None:
        assert MAX_REGISTRY_ENTRY_BYTES == 256 * 1024


class TestCanonicalTimestamp:
    def test_microsecond_precision(self) -> None:
        ts = datetime(2026, 7, 29, 13, 5, 42, 123456, tzinfo=UTC)
        assert canonical_timestamp(ts) == "2026-07-29T13:05:42.123456Z"

    def test_pads_microseconds(self) -> None:
        ts = datetime(2026, 7, 29, 13, 5, 42, 5, tzinfo=UTC)
        assert canonical_timestamp(ts) == "2026-07-29T13:05:42.000005Z"

    def test_rejects_non_utc(self) -> None:
        with pytest.raises(ValueError):
            canonical_timestamp(datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=3))))

    def test_validate_canonical_timestamp_rejects_offset(self) -> None:
        with pytest.raises(ValueError):
            validate_canonical_timestamp("2026-01-01T00:00:00.000000+02:00")

    def test_validate_canonical_timestamp_rejects_no_fraction(self) -> None:
        with pytest.raises(ValueError):
            validate_canonical_timestamp("2026-01-01T00:00:00Z")


class TestIdentityBinding:
    def test_descriptor_fingerprint_from_identity(self) -> None:
        ident = make_identity()
        d = _descriptor(specification_fingerprint=ident.fingerprint_digest_str())
        assert d.specification_fingerprint == ident.fingerprint_digest_str()
        assert d.specification_fingerprint.startswith("spec-v1-sha256-")

    def test_digest_validation_helper(self) -> None:
        validate_sha256_digest(_VALID_DIGEST)
        with pytest.raises(ValueError):
            validate_sha256_digest("sha256:short")
        with pytest.raises(ValueError):
            validate_sha256_digest("sha1:" + "a" * 64)
