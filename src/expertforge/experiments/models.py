"""Frozen experiment-manifest models and cross-field invariants (Issue #12)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from expertforge.artifacts import ArtifactRecord, validate_sha256_digest
from expertforge.identity.record import AttemptIdentityRecord

__all__ = [
    "EXPERIMENT_MANIFEST_FORMAT_VERSION",
    "EXPERIMENT_MANIFEST_SCHEMA",
    "EXPERIMENT_MANIFEST_SCHEMA_VERSION",
    "ArtifactRole",
    "AttemptStatus",
    "DatasetReference",
    "EvaluationSummary",
    "EvidenceCompleteness",
    "ExperimentClassification",
    "ExperimentManifest",
    "ManifestDiagnosticCode",
    "MaturityStage",
    "MissingEvidenceCode",
    "ModelIdentity",
    "ParameterCountKind",
    "ResearchFamily",
    "TokenizerReference",
    "TrainingBudget",
]

EXPERIMENT_MANIFEST_SCHEMA: Literal["expertforge.experiment-manifest"] = (
    "expertforge.experiment-manifest"
)
EXPERIMENT_MANIFEST_SCHEMA_VERSION: int = 1
EXPERIMENT_MANIFEST_FORMAT_VERSION: int = 1

MaturityStage = Literal[
    "Milestone 0",
    "D0",
    "D1",
    "D2",
    "M0",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "none/not-applicable",
]
ResearchFamily = Literal[
    "none/not-applicable",
    "F0",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
]
ExperimentClassification = Literal["smoke_test", "formal_experiment"]
AttemptStatus = Literal["completed", "failed", "interrupted"]
ParameterCountKind = Literal["exact", "estimated"]
ManifestDiagnosticCode = Literal[
    "handled_interruption",
    "unhandled_exception",
    "non_finite_loss",
    "non_finite_gradient",
    "abnormal_gradient_norm",
    "optimizer_failure",
    "data_failure",
    "checkpoint_failure",
    "throughput_degradation",
    "out_of_memory",
    "distributed_failure",
    "provider_failure",
    "telemetry_write_failed",
    "manifest_publication_failed",
    "checkpoint_restore_failed",
    "compatibility_mismatch",
    "configuration_error",
]
MissingEvidenceCode = Literal[
    "checkpoint_artifact_missing",
    "configuration_artifact_missing",
    "dataset_identity_missing",
    "evaluation_summary_missing",
    "generated_output_artifact_missing",
    "model_identity_missing",
    "provenance_artifact_missing",
    "telemetry_artifact_missing",
    "tokenizer_identity_missing",
    "training_budget_missing",
]
ArtifactRole = Literal[
    "configuration",
    "provenance",
    "telemetry",
    "checkpoint",
    "generated_output",
    "review_report",
    "resume_checkpoint",
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )


def _require_non_blank(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-blank.")
    return value


class DatasetReference(_FrozenModel):
    """Dataset identity bound to one specification-fingerprint immutable input."""

    dataset_id: str = Field(..., min_length=1)
    split: str = Field(..., min_length=1)
    immutable_input_name: str = Field(..., min_length=1)
    content_digest: str = Field(..., pattern=r"^sha256:[0-9a-f]{64}$")
    is_fixture: bool

    @field_validator("dataset_id", "split", "immutable_input_name")
    @classmethod
    def _validate_text(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "dataset reference"
        return _require_non_blank(value, field_name=field_name)

    @field_validator("content_digest")
    @classmethod
    def _validate_digest(cls, value: str) -> str:
        validate_sha256_digest(value)
        return value


class TokenizerReference(_FrozenModel):
    """Tokenizer identity bound to one specification-fingerprint immutable input."""

    tokenizer_id: str = Field(..., min_length=1)
    vocab_size: int = Field(..., ge=1)
    immutable_input_name: str = Field(..., min_length=1)
    content_digest: str = Field(..., pattern=r"^sha256:[0-9a-f]{64}$")
    is_fixture: bool

    @field_validator("tokenizer_id", "immutable_input_name")
    @classmethod
    def _validate_text(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "tokenizer reference"
        return _require_non_blank(value, field_name=field_name)

    @field_validator("content_digest")
    @classmethod
    def _validate_digest(cls, value: str) -> str:
        validate_sha256_digest(value)
        return value


class ModelIdentity(_FrozenModel):
    architecture: str = Field(..., min_length=1)
    dim: int = Field(..., ge=1)
    n_layers: int = Field(..., ge=1)
    n_heads: int = Field(..., ge=1)
    ffn_dim: int = Field(..., ge=1)
    parameter_count: int = Field(..., ge=0)
    parameter_count_kind: ParameterCountKind

    @field_validator("architecture")
    @classmethod
    def _validate_architecture(cls, value: str) -> str:
        return _require_non_blank(value, field_name="architecture")

    @model_validator(mode="after")
    def _validate_attention_shape(self) -> ModelIdentity:
        if self.dim % self.n_heads != 0:
            raise ValueError("model dim must be divisible by n_heads.")
        return self


class TrainingBudget(_FrozenModel):
    token_budget: int = Field(..., ge=1)
    batch_size: int = Field(..., ge=1)
    seq_len: int = Field(..., ge=1)
    learning_rate: float = Field(..., gt=0)
    seed: int = Field(..., ge=0)
    accumulation_factor: int | None = Field(default=None, ge=1)

    @field_validator("learning_rate")
    @classmethod
    def _validate_learning_rate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("learning_rate must be finite.")
        return value


class EvaluationSummary(_FrozenModel):
    eval_interval_tokens: int = Field(..., ge=1)
    final_loss: float | None = Field(default=None, ge=0)
    final_perplexity: float | None = Field(default=None, ge=0)
    metrics_recorded: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("final_loss", "final_perplexity")
    @classmethod
    def _validate_finite_optional(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("evaluation values must be finite.")
        return value

    @field_validator("metrics_recorded")
    @classmethod
    def _validate_metrics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("metrics_recorded entries must be non-blank.")
        if tuple(sorted(set(value))) != value:
            raise ValueError("metrics_recorded must be sorted and unique.")
        return value


class EvidenceCompleteness(_FrozenModel):
    status: Literal["complete", "partial"]
    missing: tuple[MissingEvidenceCode, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_status(self) -> EvidenceCompleteness:
        if tuple(sorted(set(self.missing))) != self.missing:
            raise ValueError("missing evidence codes must be sorted and unique.")
        if self.status == "complete" and self.missing:
            raise ValueError("complete evidence cannot declare missing evidence.")
        if self.status == "partial" and not self.missing:
            raise ValueError("partial evidence requires at least one missing-evidence code.")
        return self


def _record_identity(record: ArtifactRecord) -> tuple[str, str, str]:
    return (record.run_id, record.attempt_id, record.specification_fingerprint)


def _immutable_record_identity(record: ArtifactRecord) -> tuple[object, ...]:
    """Descriptor-bound fields; excludes operational retention/storage transitions."""
    parent = None if record.parent is None else record.parent.model_dump(mode="json")
    return (
        record.artifact_id,
        record.category,
        record.format,
        record.format_version,
        record.byte_size,
        record.content_digest,
        record.producing_component,
        record.run_id,
        record.attempt_id,
        record.specification_fingerprint,
        record.relative_path,
        parent,
    )


class ExperimentManifest(_FrozenModel):
    """Canonical immutable manifest for one terminal experiment attempt."""

    schema_name: Literal["expertforge.experiment-manifest"] = Field(
        default=EXPERIMENT_MANIFEST_SCHEMA, alias="schema"
    )
    schema_version: Literal[1] = 1
    format_version: Literal[1] = 1

    identity: AttemptIdentityRecord
    finalized_at_utc: datetime
    issue_number: int | None = Field(default=None, ge=1)
    pull_request_number: int | None = Field(default=None, ge=1)
    maturity_stage: MaturityStage
    research_family: ResearchFamily
    classification: ExperimentClassification
    status: AttemptStatus
    outcome_diagnostic: ManifestDiagnosticCode | None = None
    known_limitations: tuple[str, ...] = Field(default_factory=tuple)
    evidence: EvidenceCompleteness

    configuration_artifact: ArtifactRecord | None = None
    provenance_artifact: ArtifactRecord | None = None
    dataset_identity: DatasetReference | None = None
    tokenizer_identity: TokenizerReference | None = None
    model_descriptor: ModelIdentity | None = None
    training_budget: TrainingBudget | None = None
    evaluation_summary: EvaluationSummary | None = None

    smoke_objective: str | None = None
    smoke_acceptance_criteria: str | None = None
    hypothesis: str | None = None
    control: str | None = None
    fixed_constraints: tuple[str, ...] = Field(default_factory=tuple)
    independent_variable: str | None = None
    dependent_variables: tuple[str, ...] = Field(default_factory=tuple)
    minimum_useful_effect: str | None = None
    failure_threshold: str | None = None
    kill_criterion: str | None = None

    checkpoint_evidence_required: bool = False
    generated_output_evidence_required: bool = False
    telemetry_artifacts: tuple[ArtifactRecord, ...] = Field(default_factory=tuple)
    checkpoint_artifacts: tuple[ArtifactRecord, ...] = Field(default_factory=tuple)
    generated_output_artifacts: tuple[ArtifactRecord, ...] = Field(default_factory=tuple)
    review_report_artifacts: tuple[ArtifactRecord, ...] = Field(default_factory=tuple)
    resume_checkpoint: ArtifactRecord | None = None

    result: str | None = None
    decision: str | None = None

    @field_validator("finalized_at_utc")
    @classmethod
    def _validate_finalized_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("finalized_at_utc must be timezone-aware UTC.")
        if value.tzinfo.utcoffset(value) != timedelta(0):
            raise ValueError("finalized_at_utc must have UTC offset 0.")
        return value

    @field_validator("known_limitations", "fixed_constraints", "dependent_variables")
    @classmethod
    def _validate_sorted_text(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("text tuple entries must be non-blank.")
        if tuple(sorted(set(value))) != value:
            raise ValueError("text tuples must be sorted and unique.")
        return value

    @field_validator(
        "smoke_objective",
        "smoke_acceptance_criteria",
        "hypothesis",
        "control",
        "independent_variable",
        "minimum_useful_effect",
        "failure_threshold",
        "kill_criterion",
        "result",
        "decision",
    )
    @classmethod
    def _validate_optional_text(
        cls, value: str | None, info: ValidationInfo
    ) -> str | None:
        if value is None:
            return None
        field_name = info.field_name or "text"
        return _require_non_blank(value, field_name=field_name)

    @field_validator(
        "telemetry_artifacts",
        "checkpoint_artifacts",
        "generated_output_artifacts",
        "review_report_artifacts",
    )
    @classmethod
    def _validate_sorted_records(
        cls, value: tuple[ArtifactRecord, ...]
    ) -> tuple[ArtifactRecord, ...]:
        ids = tuple(record.artifact_id for record in value)
        if tuple(sorted(set(ids))) != ids:
            raise ValueError("artifact-record tuples must be sorted by artifact_id and unique.")
        return value

    @model_validator(mode="after")
    def _validate_contract(self) -> ExperimentManifest:
        if self.finalized_at_utc < self.identity.created_at_utc:
            raise ValueError("finalized_at_utc cannot precede identity.created_at_utc.")
        self._validate_classification()
        self._validate_lifecycle()
        self._validate_fixture_binding()
        self._validate_artifact_roles()
        self._validate_resume_binding()
        self._validate_evidence()
        self._validate_review_snapshot()
        return self

    def _validate_classification(self) -> None:
        formal_required = {
            "hypothesis": self.hypothesis,
            "control": self.control,
            "independent_variable": self.independent_variable,
            "minimum_useful_effect": self.minimum_useful_effect,
            "failure_threshold": self.failure_threshold,
            "kill_criterion": self.kill_criterion,
        }
        if self.classification == "formal_experiment":
            missing = [name for name, value in formal_required.items() if not value]
            if not self.fixed_constraints:
                missing.append("fixed_constraints")
            if not self.dependent_variables:
                missing.append("dependent_variables")
            if missing:
                raise ValueError(
                    "formal_experiment requires non-empty experiment-contract fields: "
                    + ", ".join(sorted(missing))
                )
            if self.smoke_objective is not None or self.smoke_acceptance_criteria is not None:
                raise ValueError("formal_experiment forbids smoke-test contract fields.")
            return

        if not self.smoke_objective or not self.smoke_acceptance_criteria:
            raise ValueError(
                "smoke_test requires smoke_objective and smoke_acceptance_criteria."
            )
        if any(value is not None for value in formal_required.values()):
            raise ValueError("smoke_test forbids formal experiment scalar fields.")
        if self.fixed_constraints or self.dependent_variables:
            raise ValueError("smoke_test forbids fixed_constraints/dependent_variables.")

    def _validate_lifecycle(self) -> None:
        if self.status in ("failed", "interrupted"):
            if self.outcome_diagnostic is None:
                raise ValueError(f"status={self.status!r} requires outcome_diagnostic.")
        elif self.outcome_diagnostic is not None:
            raise ValueError("completed status forbids outcome_diagnostic.")

    def _validate_fixture_binding(self) -> None:
        if (
            self.dataset_identity is not None
            and self.tokenizer_identity is not None
            and self.dataset_identity.is_fixture != self.tokenizer_identity.is_fixture
        ):
            raise ValueError("dataset and tokenizer fixture status must agree.")
        fingerprint_inputs = {
            item.name: item for item in self.identity.specification_fingerprint.immutable_inputs
        }
        for label, ref in (
            ("dataset_identity", self.dataset_identity),
            ("tokenizer_identity", self.tokenizer_identity),
        ):
            if ref is None:
                continue
            item = fingerprint_inputs.get(ref.immutable_input_name)
            if item is None:
                raise ValueError(
                    f"{label}.immutable_input_name {ref.immutable_input_name!r} "
                    "is absent from the specification fingerprint."
                )
            expected = f"{item.algorithm}:{item.digest}"
            if ref.content_digest != expected:
                raise ValueError(
                    f"{label}.content_digest does not match its fingerprint immutable input."
                )

    def _validate_artifact_roles(self) -> None:
        own_identity = (
            self.identity.run_id,
            self.identity.attempt_id,
            self.identity.fingerprint_digest_str(),
        )
        role_records: list[tuple[ArtifactRole, ArtifactRecord]] = []
        if self.configuration_artifact is not None:
            role_records.append(("configuration", self.configuration_artifact))
        if self.provenance_artifact is not None:
            role_records.append(("provenance", self.provenance_artifact))
        role_records.extend(("telemetry", record) for record in self.telemetry_artifacts)
        role_records.extend(("checkpoint", record) for record in self.checkpoint_artifacts)
        role_records.extend(
            ("generated_output", record) for record in self.generated_output_artifacts
        )
        role_records.extend(
            ("review_report", record) for record in self.review_report_artifacts
        )
        expected_role: dict[ArtifactRole, tuple[str, frozenset[str]]] = {
            "configuration": ("resolved_configuration", frozenset({"json"})),
            "provenance": ("provenance", frozenset({"json"})),
            "telemetry": ("telemetry", frozenset({"jsonl"})),
            "checkpoint": ("checkpoint", frozenset({"tar"})),
            "generated_output": (
                "generated_sample",
                frozenset({"json", "jsonl", "text", "binary", "tar"}),
            ),
            "review_report": ("report", frozenset({"json", "text"})),
            "resume_checkpoint": ("checkpoint", frozenset({"tar"})),
        }
        ids: list[str] = []
        for role, record in role_records:
            if _record_identity(record) != own_identity:
                raise ValueError(f"{role} artifact is not bound to the manifest attempt identity.")
            category, formats = expected_role[role]
            if record.category != category or record.format not in formats:
                raise ValueError(
                    f"{role} artifact requires category={category!r} and format in "
                    f"{sorted(formats)!r}."
                )
            if record.category == "experiment_manifest":
                raise ValueError("a manifest cannot reference itself or another manifest artifact.")
            ids.append(record.artifact_id)
        if self.resume_checkpoint is not None:
            ids.append(self.resume_checkpoint.artifact_id)
        if len(set(ids)) != len(ids):
            raise ValueError("artifact_id values must be globally unique across manifest roles.")

    def _validate_resume_binding(self) -> None:
        lineage = self.identity.lineage
        if lineage is None:
            if self.resume_checkpoint is not None:
                raise ValueError("resume_checkpoint is forbidden when identity.lineage is None.")
            return
        if lineage.parent_attempt_id is None:
            raise ValueError("experiment manifests require fully-qualified native resume lineage.")
        if self.resume_checkpoint is None:
            raise ValueError("resumed attempts require resume_checkpoint.")
        record = self.resume_checkpoint
        if record.category != "checkpoint" or record.format != "tar":
            raise ValueError("resume_checkpoint must be a canonical checkpoint tar artifact.")
        if (
            record.run_id != lineage.parent_run_id
            or record.attempt_id != lineage.parent_attempt_id
            or record.artifact_id != lineage.parent_checkpoint_id
        ):
            raise ValueError("resume_checkpoint does not match identity.lineage.")

    def _validate_evidence(self) -> None:
        missing = self.derived_missing_evidence()
        if self.evidence.missing != missing:
            raise ValueError(
                f"evidence.missing must exactly match derived missing evidence {missing!r}."
            )
        expected_status = "complete" if not missing else "partial"
        if self.evidence.status != expected_status:
            raise ValueError(
                f"evidence.status must be {expected_status!r} for missing={missing!r}."
            )
        if self.status == "completed" and missing:
            raise ValueError("completed attempts require complete core evidence.")

    def _validate_review_snapshot(self) -> None:
        if self.status != "completed" and (self.result is not None or self.decision is not None):
            raise ValueError("result/decision snapshots are permitted only for completed attempts.")

    def derived_missing_evidence(self) -> tuple[MissingEvidenceCode, ...]:
        missing: list[MissingEvidenceCode] = []
        if self.configuration_artifact is None:
            missing.append("configuration_artifact_missing")
        if self.provenance_artifact is None:
            missing.append("provenance_artifact_missing")
        if not self.telemetry_artifacts:
            missing.append("telemetry_artifact_missing")
        if self.dataset_identity is None:
            missing.append("dataset_identity_missing")
        if self.tokenizer_identity is None:
            missing.append("tokenizer_identity_missing")
        if self.model_descriptor is None:
            missing.append("model_identity_missing")
        if self.training_budget is None:
            missing.append("training_budget_missing")
        if self.evaluation_summary is None:
            missing.append("evaluation_summary_missing")
        if self.checkpoint_evidence_required and not self.checkpoint_artifacts:
            missing.append("checkpoint_artifact_missing")
        if self.generated_output_evidence_required and not self.generated_output_artifacts:
            missing.append("generated_output_artifact_missing")
        return tuple(sorted(missing))

    @property
    def fixture(self) -> bool | None:
        """Derived fixture status; absent when neither identity reference exists."""
        if self.dataset_identity is not None:
            return self.dataset_identity.is_fixture
        if self.tokenizer_identity is not None:
            return self.tokenizer_identity.is_fixture
        return None

    def referenced_attempt_artifacts(self) -> tuple[ArtifactRecord, ...]:
        """All same-attempt records, deterministically sorted by artifact_id."""
        records: list[ArtifactRecord] = []
        if self.configuration_artifact is not None:
            records.append(self.configuration_artifact)
        if self.provenance_artifact is not None:
            records.append(self.provenance_artifact)
        records.extend(self.telemetry_artifacts)
        records.extend(self.checkpoint_artifacts)
        records.extend(self.generated_output_artifacts)
        records.extend(self.review_report_artifacts)
        return tuple(sorted(records, key=lambda record: record.artifact_id))

    @staticmethod
    def immutable_artifact_identity(record: ArtifactRecord) -> tuple[object, ...]:
        return _immutable_record_identity(record)
