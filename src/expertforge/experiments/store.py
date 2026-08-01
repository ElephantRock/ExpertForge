"""Manifest generation, atomic publication, authoritative loading, and inspection."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from expertforge.artifacts import (
    ArtifactNotFoundError,
    ArtifactRecord,
    ArtifactStore,
    ArtifactStoreError,
    ParentReference,
)
from expertforge.experiments.models import (
    AttemptStatus,
    DatasetReference,
    EvaluationSummary,
    EvidenceCompleteness,
    ExperimentClassification,
    ExperimentManifest,
    ManifestDiagnosticCode,
    MaturityStage,
    MissingEvidenceCode,
    ModelIdentity,
    ResearchFamily,
    TokenizerReference,
    TrainingBudget,
)
from expertforge.experiments.serialization import (
    MAX_MANIFEST_BYTES,
    ManifestCorruptError,
    ManifestError,
    ManifestVersionError,
    canonical_manifest_bytes,
    parse_manifest_bytes,
)
from expertforge.identity.lineage import ResumeLineage
from expertforge.identity.record import AttemptIdentityRecord

__all__ = [
    "ManifestBindingError",
    "ManifestGenerator",
    "ManifestInspection",
    "ManifestPublicationError",
    "ParentCheckpointResolver",
    "load_manifest",
    "scan_manifest",
]

ParentCheckpointResolver = Callable[[ResumeLineage], tuple[ArtifactStore, ArtifactRecord]]


class ManifestBindingError(ManifestError):
    """Manifest identity, lineage, or referenced-artifact binding is invalid."""


class ManifestPublicationError(ManifestError):
    """Manifest publication cannot preserve the one-terminal-manifest contract."""


@dataclass(frozen=True)
class ManifestInspection:
    status: Literal["complete", "unsupported", "corrupt"]
    lifecycle_status: AttemptStatus | None = None
    diagnostic_code: str | None = None
    detail: str | None = None


def _same_identity(left: AttemptIdentityRecord, right: AttemptIdentityRecord) -> bool:
    return left.model_dump(mode="json", by_alias=True) == right.model_dump(
        mode="json", by_alias=True
    )


def _read_fd_bounded(fd: int, expected_size: int) -> bytes:
    if expected_size < 0 or expected_size > MAX_MANIFEST_BYTES:
        raise ManifestCorruptError(
            f"manifest size {expected_size} is outside 0..{MAX_MANIFEST_BYTES}."
        )
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        chunk = os.read(fd, min(64 * 1024, remaining))
        if not chunk:
            raise ManifestCorruptError(
                f"manifest truncated: expected {expected_size} bytes."
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(fd, 1):
        raise ManifestCorruptError("manifest content exceeds its recorded byte_size.")
    return b"".join(chunks)


def _assert_record_role(record: ArtifactRecord) -> None:
    if record.category != "experiment_manifest":
        raise ManifestBindingError(
            f"artifact {record.artifact_id} has category={record.category!r}, "
            "expected 'experiment_manifest'."
        )
    if record.format != "json" or record.format_version != 1:
        raise ManifestBindingError("experiment manifest artifact must be json format version 1.")
    if record.producing_component != "experiments":
        raise ManifestBindingError(
            "experiment manifest artifact must be produced by component 'experiments'."
        )


def _assert_record_identity(record: ArtifactRecord, identity: AttemptIdentityRecord) -> None:
    expected = (identity.run_id, identity.attempt_id, identity.fingerprint_digest_str())
    observed = (record.run_id, record.attempt_id, record.specification_fingerprint)
    if observed != expected:
        raise ManifestBindingError("artifact record is not bound to the expected attempt identity.")


def _resolve_current_record(store: ArtifactStore, record: ArtifactRecord) -> ArtifactRecord:
    try:
        current = store.inspect(record.artifact_id)
    except (ArtifactNotFoundError, ArtifactStoreError) as exc:
        raise ManifestBindingError(
            f"referenced artifact {record.artifact_id} is not authoritatively registered: {exc}"
        ) from exc
    if not isinstance(current, ArtifactRecord):
        raise ManifestBindingError(
            f"referenced artifact {record.artifact_id} is external-only, not a canonical record."
        )
    if ExperimentManifest.immutable_artifact_identity(current) != (
        ExperimentManifest.immutable_artifact_identity(record)
    ):
        raise ManifestBindingError(
            f"referenced artifact {record.artifact_id} immutable metadata mismatch."
        )
    try:
        verification = store.verify(record.artifact_id)
    except ArtifactStoreError as exc:
        raise ManifestBindingError(
            f"referenced artifact {record.artifact_id} could not be verified: {exc}"
        ) from exc
    if not verification.status:
        raise ManifestBindingError(
            f"referenced artifact {record.artifact_id} failed verification: "
            f"{verification.diagnostic_code}."
        )
    return current


def _verify_manifest_references(
    manifest: ExperimentManifest,
    store: ArtifactStore,
    *,
    parent_checkpoint_resolver: ParentCheckpointResolver | None,
) -> None:
    for record in manifest.referenced_attempt_artifacts():
        _resolve_current_record(store, record)
    lineage = manifest.identity.lineage
    if lineage is None:
        return
    if parent_checkpoint_resolver is None:
        raise ManifestBindingError(
            "authoritative publication/loading of a resumed manifest requires "
            "parent_checkpoint_resolver."
        )
    parent_store, resolved = parent_checkpoint_resolver(lineage)
    if (
        parent_store.identity.run_id != lineage.parent_run_id
        or parent_store.identity.attempt_id != lineage.parent_attempt_id
    ):
        raise ManifestBindingError("parent checkpoint resolver returned the wrong attempt store.")
    current = _resolve_current_record(parent_store, resolved)
    assert manifest.resume_checkpoint is not None
    if ExperimentManifest.immutable_artifact_identity(current) != (
        ExperimentManifest.immutable_artifact_identity(manifest.resume_checkpoint)
    ):
        raise ManifestBindingError("resolved parent checkpoint does not match manifest lineage.")


class ManifestGenerator:
    """Build and publish terminal manifests from existing typed subsystem records."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._artifact_store = artifact_store

    @property
    def artifact_store(self) -> ArtifactStore:
        return self._artifact_store

    def generate(
        self,
        *,
        identity: AttemptIdentityRecord,
        classification: ExperimentClassification,
        maturity_stage: MaturityStage,
        research_family: ResearchFamily,
        status: AttemptStatus,
        issue_number: int | None = None,
        pull_request_number: int | None = None,
        configuration_artifact: ArtifactRecord | None = None,
        provenance_artifact: ArtifactRecord | None = None,
        dataset: DatasetReference | None = None,
        tokenizer: TokenizerReference | None = None,
        fixture: bool = True,
        model: ModelIdentity | None = None,
        training: TrainingBudget | None = None,
        evaluation: EvaluationSummary | None = None,
        hypothesis: str | None = None,
        control: str | None = None,
        fixed_constraints: tuple[str, ...] = (),
        independent_variable: str | None = None,
        dependent_variables: tuple[str, ...] = (),
        minimum_useful_effect: str | None = None,
        failure_threshold: str | None = None,
        kill_criterion: str | None = None,
        outcome_diagnostic: ManifestDiagnosticCode | None = None,
        known_limitations: tuple[str, ...] = (),
        telemetry_artifacts: tuple[ArtifactRecord, ...] = (),
        checkpoint_artifacts: tuple[ArtifactRecord, ...] = (),
        generated_output_artifacts: tuple[ArtifactRecord, ...] = (),
        review_report_artifacts: tuple[ArtifactRecord, ...] = (),
        resume_checkpoint: ArtifactRecord | None = None,
    ) -> ExperimentManifest:
        if not _same_identity(identity, self._artifact_store.identity):
            raise ManifestBindingError(
                "ManifestGenerator identity must equal the ArtifactStore-bound identity."
            )
        missing: list[MissingEvidenceCode] = []
        if configuration_artifact is None:
            missing.append("configuration_artifact_missing")
        if provenance_artifact is None:
            missing.append("provenance_artifact_missing")
        if not telemetry_artifacts:
            missing.append("telemetry_artifact_missing")
        if dataset is None:
            missing.append("dataset_identity_missing")
        if tokenizer is None:
            missing.append("tokenizer_identity_missing")
        if model is None:
            missing.append("model_identity_missing")
        if training is None:
            missing.append("training_budget_missing")
        missing_codes = tuple(sorted(missing))
        evidence = EvidenceCompleteness(
            status="complete" if not missing_codes else "partial",
            missing=missing_codes,
        )
        return ExperimentManifest(
            identity=identity,
            issue_number=issue_number,
            pull_request_number=pull_request_number,
            maturity_stage=maturity_stage,
            research_family=research_family,
            classification=classification,
            status=status,
            outcome_diagnostic=outcome_diagnostic,
            known_limitations=tuple(sorted(set(known_limitations))),
            evidence=evidence,
            configuration_artifact=configuration_artifact,
            provenance_artifact=provenance_artifact,
            dataset_identity=dataset,
            tokenizer_identity=tokenizer,
            fixture=fixture,
            model_descriptor=model,
            training_budget=training,
            evaluation_summary=evaluation,
            hypothesis=hypothesis,
            control=control,
            fixed_constraints=tuple(sorted(set(fixed_constraints))),
            independent_variable=independent_variable,
            dependent_variables=tuple(sorted(set(dependent_variables))),
            minimum_useful_effect=minimum_useful_effect,
            failure_threshold=failure_threshold,
            kill_criterion=kill_criterion,
            telemetry_artifacts=tuple(
                sorted(telemetry_artifacts, key=lambda item: item.artifact_id)
            ),
            checkpoint_artifacts=tuple(
                sorted(checkpoint_artifacts, key=lambda item: item.artifact_id)
            ),
            generated_output_artifacts=tuple(
                sorted(generated_output_artifacts, key=lambda item: item.artifact_id)
            ),
            review_report_artifacts=tuple(
                sorted(review_report_artifacts, key=lambda item: item.artifact_id)
            ),
            resume_checkpoint=resume_checkpoint,
        )

    def publish(
        self,
        manifest: ExperimentManifest,
        *,
        parent_checkpoint_resolver: ParentCheckpointResolver | None = None,
    ) -> ArtifactRecord:
        if not _same_identity(manifest.identity, self._artifact_store.identity):
            raise ManifestBindingError("manifest identity does not match the ArtifactStore identity.")
        _verify_manifest_references(
            manifest,
            self._artifact_store,
            parent_checkpoint_resolver=parent_checkpoint_resolver,
        )
        existing = self._artifact_store.list_artifacts(category="experiment_manifest")
        payload = canonical_manifest_bytes(manifest)
        if existing:
            if len(existing) != 1:
                raise ManifestPublicationError(
                    "attempt has multiple registered experiment manifests; registry is inconsistent."
                )
            fd, existing_record = self._artifact_store.open_verified_content(
                existing[0].artifact_id
            )
            try:
                existing_payload = _read_fd_bounded(fd, existing_record.byte_size)
            finally:
                os.close(fd)
            if existing_payload == payload:
                return existing_record
            raise ManifestPublicationError(
                "attempt already has a different terminal experiment manifest."
            )
        parent: ParentReference | None = None
        lineage = manifest.identity.lineage
        if lineage is not None:
            assert lineage.parent_attempt_id is not None
            parent = ParentReference(
                run_id=lineage.parent_run_id,
                attempt_id=lineage.parent_attempt_id,
                artifact_id=lineage.parent_checkpoint_id,
            )
        try:
            record = self._artifact_store.publish(
                payload,
                category="experiment_manifest",
                format="json",
                format_version=1,
                producing_component="experiments",
                parent=parent,
            )
        except (ArtifactStoreError, OSError) as exc:
            raise ManifestPublicationError(f"manifest publication failed: {exc}") from exc
        return record

    def finalize_attempt(
        self,
        *,
        parent_checkpoint_resolver: ParentCheckpointResolver | None = None,
        **generate_kwargs: Any,
    ) -> ArtifactRecord:
        """Generate and publish in one call; retained for orchestrator convenience."""
        manifest = self.generate(**generate_kwargs)
        return self.publish(
            manifest,
            parent_checkpoint_resolver=parent_checkpoint_resolver,
        )


def load_manifest(
    artifact_id: str,
    *,
    artifact_store: ArtifactStore,
    expected_identity: AttemptIdentityRecord,
    parent_checkpoint_resolver: ParentCheckpointResolver | None = None,
) -> ExperimentManifest:
    """Authoritatively load a registered manifest through #10's verified descriptor."""
    if not _same_identity(artifact_store.identity, expected_identity):
        raise ManifestBindingError("artifact_store identity does not match expected_identity.")
    try:
        fd, record = artifact_store.open_verified_content(artifact_id)
    except (ArtifactNotFoundError, ArtifactStoreError, OSError) as exc:
        raise ManifestBindingError(f"could not open registered manifest artifact: {exc}") from exc
    try:
        _assert_record_role(record)
        _assert_record_identity(record, expected_identity)
        try:
            st = os.fstat(fd)
        except OSError as exc:
            raise ManifestCorruptError(f"could not stat manifest content: {exc}") from exc
        if not stat.S_ISREG(st.st_mode):
            raise ManifestCorruptError("manifest content is not a regular file.")
        if st.st_size != record.byte_size:
            raise ManifestCorruptError(
                f"manifest byte size {st.st_size} does not match record {record.byte_size}."
            )
        raw = _read_fd_bounded(fd, record.byte_size)
        observed_digest = f"sha256:{hashlib.sha256(raw).hexdigest()}"
        if observed_digest != record.content_digest:
            raise ManifestCorruptError(
                "manifest content digest does not match its artifact record."
            )
    finally:
        os.close(fd)
    manifest = parse_manifest_bytes(raw)
    if not _same_identity(manifest.identity, expected_identity):
        raise ManifestBindingError("manifest identity does not match expected_identity.")
    _verify_manifest_references(
        manifest,
        artifact_store,
        parent_checkpoint_resolver=parent_checkpoint_resolver,
    )
    return manifest


def _open_scan_path(path: Path) -> int:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise OSError(f"manifest scan path {path} is a symlink.")
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags)


def scan_manifest(path: Path) -> ManifestInspection:
    """Non-authoritative bounded diagnostic scan of one manifest JSON file."""
    fd: int | None = None
    try:
        fd = _open_scan_path(path)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return ManifestInspection(
                status="corrupt",
                diagnostic_code="not_regular_file",
                detail=str(path),
            )
        raw = _read_fd_bounded(fd, st.st_size)
        manifest = parse_manifest_bytes(raw)
        return ManifestInspection(status="complete", lifecycle_status=manifest.status)
    except ManifestVersionError as exc:
        return ManifestInspection(
            status="unsupported",
            diagnostic_code="unsupported_version",
            detail=str(exc),
        )
    except (ManifestCorruptError, OSError) as exc:
        return ManifestInspection(
            status="corrupt",
            diagnostic_code="invalid_manifest",
            detail=str(exc),
        )
    finally:
        if fd is not None:
            os.close(fd)
