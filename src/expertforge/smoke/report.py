"""Frozen, versioned smoke comparison report (Issue #14, amendment K + correction #7).

The machine-readable ``ComparisonReport`` is published ONLY as an R1 report
artifact, after the U0@N vs R1@N comparison and before R1's terminal manifest.
It carries both operands' identities (run/attempt IDs + specification
fingerprint) for ATTRIBUTION ONLY — they do not participate in the equality
decision. A failed comparison preserves both observed computational digests.

The report is frozen, versioned, canonically serialized (sorted-key compact
JSON), and uses a closed diagnostic domain. It CANNOT reference R1's future
manifest ID (the manifest is published after the report). The committed
``reports/milestone-0-smoke-gate.md`` (produced from the accepted evidence,
never rewritten by tests/CLI) may include all final artifact and manifest IDs.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "COMPARISON_MISMATCH_DOMAIN",
    "COMPARISON_REPORT_SCHEMA",
    "COMPARISON_REPORT_SCHEMA_VERSION",
    "ComparisonDecision",
    "ComparisonEnvironment",
    "ComparisonMismatchField",
    "ComparisonReport",
    "canonical_report_bytes",
    "parse_report_bytes",
]

COMPARISON_REPORT_SCHEMA: Literal["expertforge.smoke-comparison-report"] = (
    "expertforge.smoke-comparison-report"
)
COMPARISON_REPORT_SCHEMA_VERSION: int = 1

ComparisonDecision = Literal["pass", "fail"]

# Closed diagnostic domain for fields that may mismatch in a comparison.
ComparisonMismatchField = Literal[
    "parameters_digest",
    "optimizer_slots_digest",
    "optimizer_scalar_digest",
    "scheduler_digest",
    "counters_digest",
    "cursor_digest",
    "next_item_probe",
    "next_random_probe",
    "validation_loss",
    "generated_sample",
    "computational_digest",
]

# The closed diagnostic domain as a runtime frozenset, for canonicalizing
# observed mismatch lists before constructing a ComparisonReport (finding F3).
COMPARISON_MISMATCH_DOMAIN: frozenset[str] = frozenset(
    {
        "parameters_digest",
        "optimizer_slots_digest",
        "optimizer_scalar_digest",
        "scheduler_digest",
        "counters_digest",
        "cursor_digest",
        "next_item_probe",
        "next_random_probe",
        "validation_loss",
        "generated_sample",
        "computational_digest",
    }
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
        populate_by_name=True,
    )


class ComparisonEnvironment(_FrozenModel):
    """The deterministic environment the exact-byte claim is bounded to."""

    python_version: str = Field(..., min_length=1)
    numpy_version: str = Field(..., min_length=1)
    platform: str = Field(..., min_length=1)
    machine: str = Field(..., min_length=1)
    processor: str = Field(default="")
    determinism_note: str = Field(default="")


class ComparisonReport(_FrozenModel):
    """The machine-readable U0@N vs R1@N comparison report."""

    schema_name: Literal["expertforge.smoke-comparison-report"] = Field(
        default=COMPARISON_REPORT_SCHEMA, alias="schema"
    )
    schema_version: Literal[1] = COMPARISON_REPORT_SCHEMA_VERSION  # type: ignore[assignment]

    # Operand identities — for attribution only, never part of the equality
    # decision.
    u0_run_id: str = Field(..., min_length=1)
    u0_attempt_id: str = Field(..., min_length=1)
    r1_run_id: str = Field(..., min_length=1)
    r1_attempt_id: str = Field(..., min_length=1)
    specification_fingerprint: str = Field(..., min_length=1)

    # Update-boundary operands.
    n_updates: int = Field(..., ge=2)
    k_updates: int = Field(..., ge=1)
    tokens_per_update: int = Field(..., ge=1)

    # BOTH computational digests, recorded separately (correction #7).
    u0_computational_digest: str = Field(..., min_length=1)
    r1_computational_digest: str = Field(..., min_length=1)

    # Observed validation losses and generated-sample digest.
    u0_validation_loss: float
    r1_validation_loss: float
    generated_sample_digest: str = Field(..., min_length=1)

    decision: ComparisonDecision
    mismatches: tuple[ComparisonMismatchField, ...] = Field(default_factory=tuple)
    environment: ComparisonEnvironment

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, v: int) -> int:
        if v != COMPARISON_REPORT_SCHEMA_VERSION:
            raise ValueError(f"unsupported comparison-report schema_version {v!r}.")
        return v

    @field_validator("mismatches")
    @classmethod
    def _check_mismatches(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if list(v) != sorted(v):
            raise ValueError("mismatches must be sorted.")
        if len(set(v)) != len(v):
            raise ValueError("mismatches must be unique.")
        return v

    @model_validator(mode="after")
    def _check_decision(self) -> ComparisonReport:
        if self.decision == "pass" and self.mismatches:
            raise ValueError("decision='pass' forbids mismatches.")
        if self.decision == "fail" and not self.mismatches:
            raise ValueError("decision='fail' requires at least one mismatch.")
        if self.u0_computational_digest != self.r1_computational_digest and self.decision != "fail":
            raise ValueError("unequal computational digests require decision='fail'.")
        if (
            self.u0_computational_digest == self.r1_computational_digest
            and "computational_digest" in self.mismatches
        ):
            raise ValueError(
                "equal computational digests cannot list 'computational_digest' as a mismatch."
            )
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_report_bytes(self)


def canonical_report_bytes(report: ComparisonReport) -> bytes:
    """Canonical sorted-key compact JSON serialization."""
    return json.dumps(
        report.model_dump(mode="json", by_alias=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def parse_report_bytes(payload: bytes) -> ComparisonReport:
    """Parse and validate a comparison report from canonical bytes."""
    data = json.loads(payload)
    return ComparisonReport.model_validate(data, strict=False)
