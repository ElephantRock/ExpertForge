"""Core frozen telemetry models (Issue #9 design comment 5136093570).

Three independently versioned contracts:

- ``TELEMETRY_STREAM_FORMAT_VERSION = 1`` — JSONL stream framing/lifecycle;
- ``EVENT_SCHEMA_VERSION = 1`` (schema ``expertforge.telemetry-event``);
- ``METRIC_SCHEMA_VERSION = 1`` (schema ``expertforge.metric-record``).

All persisted models are frozen, deeply immutable, ``extra="forbid"``, strict
Pydantic v2 models. Mutable mappings/lists are never persisted; tuples of frozen
nested models are used instead. NaN/Infinity are never serialized as JSON numbers
(see :class:`MetricValueStatus` and :meth:`EventField` value validation).

This module has NO import-time side effects: no handlers, no file access, no
environment reads, no root-logger mutation, no output. It depends only on the
standard library and the existing Pydantic runtime, plus the frozen Issue #6
identity types (which it references only by string, not import, to keep the
boundary one-directional at runtime).
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "DIAGNOSTIC_CODES",
    "DiagnosticCode",
    "EVENT_SCHEMA",
    "EVENT_SCHEMA_VERSION",
    "MAX_EVENT_FIELDS",
    "MAX_METRIC_OBSERVATIONS",
    "MAX_PERSISTED_STRING",
    "MAX_RECORD_BYTES",
    "METRIC_AGGREGATIONS",
    "METRIC_SCHEMA",
    "METRIC_SCHEMA_VERSION",
    "METRIC_UNITS",
    "METRIC_VALUE_REASONS",
    "METRIC_VALUE_STATUSES",
    "METRIC_WINDOWS",
    "STREAM_FORMAT_VERSION",
    "EventField",
    "EventRecord",
    "MetricObservation",
    "MetricRecord",
    "MetricValueStatus",
    "ProcessContext",
    "ProgressPosition",
    "StreamOutcome",
    "StreamStatus",
    "WriterStats",
    "redacted",
    "sanitize_operator_message",
]

# --- independent versions and schema names ---------------------------------

STREAM_FORMAT_VERSION: int = 1
EVENT_SCHEMA_VERSION: int = 1
METRIC_SCHEMA_VERSION: int = 1
EVENT_SCHEMA: str = "expertforge.telemetry-event"
METRIC_SCHEMA: str = "expertforge.metric-record"

# --- bounded overhead limits (design §13) ----------------------------------

MAX_RECORD_BYTES: int = 64 * 1024  # 64 KiB including the trailing newline.
MAX_EVENT_FIELDS: int = 64
MAX_METRIC_OBSERVATIONS: int = 128
MAX_PERSISTED_STRING: int = 4096

# --- closed Literal domains (design §6, §7) --------------------------------

# Event diagnostic codes. Raw exception text, tracebacks, shell output, env
# dumps and arbitrary provider stderr are NOT diagnostic codes and must never
# be persisted here.
DiagnosticCode = Literal[
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
    "console_write_failed",
    "telemetry_write_failed",
    "telemetry_flush_failed",
    "redacted_sensitive_value",
]
DIAGNOSTIC_CODES: frozenset[str] = frozenset(
    {
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
        "console_write_failed",
        "telemetry_write_failed",
        "telemetry_flush_failed",
        "redacted_sensitive_value",
    }
)

# Metric structural value states. NaN and infinities are represented by their
# status, never as JSON NaN/Infinity.
METRIC_VALUE_STATUSES: frozenset[str] = frozenset(
    {
        "finite",
        "nan",
        "positive_infinity",
        "negative_infinity",
        "unavailable",
        "error",
        "redacted",
    }
)

# Metric-value reasons (required for unavailable/error/redacted).
METRIC_VALUE_REASONS: frozenset[str] = frozenset(
    {
        "not_collected",
        "not_applicable",
        "provider_unavailable",
        "input_non_finite",
        "division_by_zero",
        "overflow",
        "computation_failed",
        "sensitive_value_redacted",
    }
)

METRIC_UNITS: frozenset[str] = frozenset(
    {
        "dimensionless",
        "count",
        "ratio",
        "percent",
        "tokens",
        "tokens_per_second",
        "seconds",
        "milliseconds",
        "bytes",
        "bytes_per_second",
        "samples",
        "sequences",
        "steps",
        "updates",
    }
)

METRIC_AGGREGATIONS: frozenset[str] = frozenset(
    {
        "gauge",
        "counter",
        "delta",
        "sum",
        "mean",
        "minimum",
        "maximum",
        "rate",
    }
)

METRIC_WINDOWS: frozenset[str] = frozenset(
    {
        "point",
        "since_last_emit",
        "attempt",
        "run",
        "evaluation",
    }
)

StreamStatus = Literal["complete", "incomplete", "corrupt"]
StreamOutcome = Literal["normal", "interrupted", "failed"]
MetricValueStatus = Literal[
    "finite",
    "nan",
    "positive_infinity",
    "negative_infinity",
    "unavailable",
    "error",
    "redacted",
]

# Stable lowercase identifier: letters, digits, dot, underscore, dash; must
# start with a lowercase letter or digit. Used for component, event_name,
# metric namespace/name, and event-field names.
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
# Namespaced event/metric names allow a dotted path.
_NAMESPACED_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

# Rate-style units encode a denominator (per_second). Required for aggregation
# ``rate`` (and a structural hint, not the sole enforcement).
_RATE_UNITS: frozenset[str] = frozenset({"tokens_per_second", "bytes_per_second"})


def _require_utc_aware(value: datetime, field_name: str) -> datetime:
    """Reject naive and non-UTC datetimes so timestamps are never
    host-timezone-interpreted and never silently normalized."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC; got naive {value!r}.")
    if value.tzinfo.utcoffset(value) != timedelta(0):
        raise ValueError(
            f"{field_name} must be UTC (offset 0); got offset {value.tzinfo.utcoffset(value)!r}."
        )
    return value


# ---------------------------------------------------------------------------
# Structural redaction (design §12)
# ---------------------------------------------------------------------------

# Defense-in-depth regexes. Structural omission is the primary mechanism; these
# sanitize the operator message string before it is persisted.
_RE_URL_CREDENTIALS = re.compile(r"(://)([^@/\s]+@)+")
_RE_QUERY_FRAGMENT = re.compile(r"([?&#][^?&#\s]*)")
_RE_SIGNED_URL = re.compile(r"(sig|signature|token|se)=[A-Za-z0-9%+/=_-]{8,}", re.IGNORECASE)
_RE_KV_SECRET = re.compile(
    r"(?i)\b(token|password|passwd|secret|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|credential|auth|bearer)\b\s*[:=]\s*\S+"
)
_RE_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?\b")
_RE_IPV6 = re.compile(r"\b(?:[A-Fa-f0-9:]{2,})+(?::\d{1,5})?\b")
_RE_HOME_PATH = re.compile(r"(?i)(/[A-Za-z]\:?|[/\\])(?:users|home|root)[/\\][^/\s\\]+")
_RE_WIN_PATH = re.compile(r"\b[A-Za-z]:[\\/](?:[^\s\"'<>|]+)")


def redacted(text: str) -> str:
    """Replace matched sensitive substrings with ``[redacted]`` (defense in depth)."""
    out = _RE_URL_CREDENTIALS.sub(r"\1[redacted]@", text)
    out = _RE_SIGNED_URL.sub("sig=[redacted]", out)
    out = _RE_KV_SECRET.sub(
        lambda m: (
            m.group(0).split("=", 1)[0].split(":", 1)[0] + "=[redacted]"
            if "=" in m.group(0)
            else m.group(0).split(":", 1)[0] + ":[redacted]"
        ),
        out,
    )
    out = _RE_IPV4.sub("[redacted]", out)
    out = _RE_HOME_PATH.sub("[redacted]", out)
    out = _RE_WIN_PATH.sub("[redacted]", out)
    # Remove query strings / fragments after URL credentials handling.
    out = _RE_QUERY_FRAGMENT.sub("[redacted]", out)
    return out


def sanitize_operator_message(message: str) -> str:
    """Structurally sanitize an operator message before persistence.

    Forbids control characters, applies defense-in-depth redaction, and truncates
    to :data:`MAX_PERSISTED_STRING`. Raises ``ValueError`` on control characters
    (callers surface this as a validation error).
    """
    if any(ord(ch) < 0x20 and ch not in ("\t",) for ch in message):
        raise ValueError("operator_message contains forbidden control characters.")
    out = redacted(message)
    if len(out) > MAX_PERSISTED_STRING:
        out = out[:MAX_PERSISTED_STRING]
    return out


# ---------------------------------------------------------------------------
# ProcessContext and ProgressPosition
# ---------------------------------------------------------------------------


class ProcessContext(BaseModel):
    """Process topology coordinates. Callers pass these explicitly."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    rank: int = Field(..., ge=0)
    world_size: int = Field(..., ge=1)
    local_rank: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _enforce_topology(self) -> ProcessContext:
        if self.rank >= self.world_size:
            raise ValueError(f"rank ({self.rank}) must be < world_size ({self.world_size}).")
        if self.local_rank is not None and self.local_rank >= self.world_size:
            raise ValueError(
                f"local_rank ({self.local_rank}) must be < world_size ({self.world_size}) when set."
            )
        return self


class ProgressPosition(BaseModel):
    """Optional committed-progress position at the instant of emission.

    All counters are non-negative. Omitted fields make no progress claim and do
    not reset prior values; counters are non-decreasing within a stream.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    step: int | None = Field(default=None, ge=0)
    update: int | None = Field(default=None, ge=0)
    processed_tokens: int | None = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# EventField and EventRecord
# ---------------------------------------------------------------------------


class EventField(BaseModel):
    """One stable, uniquely-named typed event field.

    Values are strict strings, integers, booleans, or FINITE floats only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    name: str = Field(..., min_length=1)
    value: str | int | float | bool

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(v):
            raise ValueError(
                f"event-field name {v!r} must be a stable lowercase identifier "
                f"({_IDENTIFIER_RE.pattern})."
            )
        return v

    @field_validator("value")
    @classmethod
    def _validate_value(cls, v: str | int | float | bool) -> str | int | float | bool:
        # Booleans are a subclass of int in Python; strict mode keeps them
        # distinct. Reject non-finite floats explicitly.
        if isinstance(v, bool):
            return v
        if isinstance(v, float):
            if not math.isfinite(v):
                raise ValueError(
                    f"event-field value {v!r} is not finite; NaN/Infinity are not persisted."
                )
            return v
        if isinstance(v, (int, str)):
            return v
        # strict=True already rejects other types, but guard defensively.
        raise ValueError(
            f"event-field value has unsupported type {type(v).__name__}."
        )  # pragma: no cover


class _RecordBase(BaseModel):
    """Shared identity/context fields for both event and metric records."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
        populate_by_name=True,
    )

    stream_format_version: int = Field(default=STREAM_FORMAT_VERSION)
    run_id: str = Field(..., min_length=1)
    attempt_id: str = Field(..., min_length=1)
    specification_fingerprint: str = Field(..., min_length=1)
    component: str = Field(..., min_length=1)
    rank: int = Field(..., ge=0)
    local_rank: int | None = Field(default=None, ge=0)
    world_size: int = Field(..., ge=1)
    sequence: int = Field(..., ge=0)
    timestamp_utc: datetime
    elapsed_ns: int = Field(..., ge=0)
    progress: ProgressPosition | None = Field(default=None)

    @field_validator("stream_format_version")
    @classmethod
    def _validate_stream_format_version(cls, v: int) -> int:
        if v != STREAM_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported stream_format_version {v}; this version understands "
                f"{STREAM_FORMAT_VERSION}."
            )
        return v

    @field_validator("specification_fingerprint")
    @classmethod
    def _validate_fingerprint(cls, v: str) -> str:
        if not re.fullmatch(r"spec-v1-sha256-[0-9a-f]{64}", v):
            raise ValueError(
                f"specification_fingerprint must be 'spec-v1-sha256-<64 hex>'; got {v!r}."
            )
        return v

    @field_validator("component")
    @classmethod
    def _validate_component(cls, v: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(v):
            raise ValueError(
                f"component {v!r} must be a stable lowercase identifier ({_IDENTIFIER_RE.pattern})."
            )
        return v

    @field_validator("timestamp_utc")
    @classmethod
    def _validate_timestamp(cls, v: datetime) -> datetime:
        return _require_utc_aware(v, "timestamp_utc")

    @model_validator(mode="after")
    def _enforce_topology(self) -> _RecordBase:
        if self.rank >= self.world_size:
            raise ValueError(f"rank ({self.rank}) must be < world_size ({self.world_size}).")
        if self.local_rank is not None and self.local_rank >= self.world_size:
            raise ValueError(
                f"local_rank ({self.local_rank}) must be < world_size ({self.world_size}) when set."
            )
        return self

    def _deterministic_json(self, payload: dict[str, object]) -> bytes:
        """Compact, sorted-key, UTF-8, allow_nan=False JSON of ``payload``."""
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


class EventRecord(_RecordBase):
    """One structured event record (schema ``expertforge.telemetry-event``)."""

    schema_name: Literal["expertforge.telemetry-event"] = Field(
        default="expertforge.telemetry-event", alias="schema"
    )
    schema_version: int = Field(default=EVENT_SCHEMA_VERSION)
    severity: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    event_name: str = Field(..., min_length=1)
    diagnostic_code: DiagnosticCode | None = Field(default=None)
    fields: tuple[EventField, ...] = Field(default_factory=tuple)
    operator_message: str | None = Field(default=None)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: int) -> int:
        if v != EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported event schema_version {v}; this version understands "
                f"{EVENT_SCHEMA_VERSION}."
            )
        return v

    @field_validator("event_name")
    @classmethod
    def _validate_event_name(cls, v: str) -> str:
        if not _NAMESPACED_RE.fullmatch(v):
            raise ValueError(
                f"event_name {v!r} must be a lowercase namespaced identifier "
                f"({_NAMESPACED_RE.pattern})."
            )
        return v

    @field_validator("operator_message")
    @classmethod
    def _validate_operator_message(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if len(v) > MAX_PERSISTED_STRING:
            raise ValueError(
                f"operator_message length {len(v)} exceeds max {MAX_PERSISTED_STRING}."
            )
        if any(ord(ch) < 0x20 and ch not in ("\t",) for ch in v):
            raise ValueError("operator_message contains forbidden control characters.")
        return v

    @model_validator(mode="after")
    def _enforce_fields(self) -> EventRecord:
        if len(self.fields) > MAX_EVENT_FIELDS:
            raise ValueError(
                f"event fields count {len(self.fields)} exceeds max {MAX_EVENT_FIELDS}."
            )
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"event-field names must be unique; got duplicates {names!r}.")
        if names != sorted(names):
            raise ValueError(f"event-field names must be sorted; got {names!r}.")
        return self

    def to_deterministic_json(self) -> bytes:
        """Compact sorted-key UTF-8 JSON (no NaN/Infinity) of this record."""
        return self._deterministic_json(self.model_dump(mode="json", by_alias=True))


# ---------------------------------------------------------------------------
# MetricObservation and MetricRecord
# ---------------------------------------------------------------------------


class MetricObservation(BaseModel):
    """One metric observation, identified by (namespace, name)."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    namespace: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    unit: Literal[
        "dimensionless",
        "count",
        "ratio",
        "percent",
        "tokens",
        "tokens_per_second",
        "seconds",
        "milliseconds",
        "bytes",
        "bytes_per_second",
        "samples",
        "sequences",
        "steps",
        "updates",
    ]
    aggregation: Literal[
        "gauge",
        "counter",
        "delta",
        "sum",
        "mean",
        "minimum",
        "maximum",
        "rate",
    ]
    window: Literal[
        "point",
        "since_last_emit",
        "attempt",
        "run",
        "evaluation",
    ]
    value_status: MetricValueStatus
    value: int | float | None = Field(default=None)
    sample_count: int | None = Field(default=None, ge=1)
    reason: (
        Literal[
            "not_collected",
            "not_applicable",
            "provider_unavailable",
            "input_non_finite",
            "division_by_zero",
            "overflow",
            "computation_failed",
            "sensitive_value_redacted",
        ]
        | None
    ) = Field(default=None)

    @field_validator("namespace", "name")
    @classmethod
    def _validate_ns_name(cls, v: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(v):
            raise ValueError(
                f"metric namespace/name {v!r} must be a stable lowercase identifier "
                f"({_IDENTIFIER_RE.pattern})."
            )
        return v

    @field_validator("value")
    @classmethod
    def _validate_value(cls, v: int | float | None) -> int | float | None:
        if v is not None and isinstance(v, float) and not math.isfinite(v):
            raise ValueError(
                f"metric value {v!r} is not finite; use a non-finite value_status instead."
            )
        return v

    @model_validator(mode="after")
    def _enforce_relationships(self) -> MetricObservation:
        # finite requires a finite value and forbids a reason.
        if self.value_status == "finite":
            if self.value is None:
                raise ValueError("value_status 'finite' requires a value.")
            if self.reason is not None:
                raise ValueError("value_status 'finite' forbids a reason.")
        else:
            # Every non-finite status forbids a numeric value.
            if self.value is not None:
                raise ValueError(f"value_status {self.value_status!r} forbids a numeric value.")
            if self.value_status in {"unavailable", "error", "redacted"} and self.reason is None:
                raise ValueError(f"value_status {self.value_status!r} requires a reason.")
            if (
                self.value_status in {"nan", "positive_infinity", "negative_infinity"}
                and self.reason is not None
            ):
                raise ValueError(f"value_status {self.value_status!r} forbids a reason.")

        # aggregation / window relationships (design §7).
        agg = self.aggregation
        win = self.window
        if agg == "gauge" and win != "point":
            raise ValueError("aggregation 'gauge' requires window 'point'.")
        if agg == "counter" and win not in {"attempt", "run"}:
            raise ValueError("aggregation 'counter' requires window 'attempt' or 'run'.")
        if agg in {"delta", "rate"} and win != "since_last_emit":
            raise ValueError(f"aggregation {agg!r} requires window 'since_last_emit'.")
        if agg in {"sum", "mean", "minimum", "maximum"}:
            if win == "point":
                raise ValueError(f"aggregation {agg!r} requires a non-point window.")
            if self.sample_count is None:
                raise ValueError(f"aggregation {agg!r} requires a positive sample_count.")
        # rate units must encode a denominator.
        if agg == "rate" and self.unit not in _RATE_UNITS:
            raise ValueError(f"aggregation 'rate' requires a per-second unit; got {self.unit!r}.")
        return self

    def to_deterministic_json(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


class MetricRecord(_RecordBase):
    """One metric record (schema ``expertforge.metric-record``).

    Contains one or more :class:`MetricObservation` values, sorted and unique by
    (namespace, name). Identity/context is inherited from this record.
    """

    schema_name: Literal["expertforge.metric-record"] = Field(
        default="expertforge.metric-record", alias="schema"
    )
    schema_version: int = Field(default=METRIC_SCHEMA_VERSION)
    observations: tuple[MetricObservation, ...] = Field(..., min_length=1)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: int) -> int:
        if v != METRIC_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported metric schema_version {v}; this version understands "
                f"{METRIC_SCHEMA_VERSION}."
            )
        return v

    @model_validator(mode="after")
    def _enforce_observations(self) -> MetricRecord:
        if len(self.observations) > MAX_METRIC_OBSERVATIONS:
            raise ValueError(
                f"metric observations count {len(self.observations)} exceeds max "
                f"{MAX_METRIC_OBSERVATIONS}."
            )
        keys = [(o.namespace, o.name) for o in self.observations]
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"metric observations must be unique by (namespace, name); got {keys!r}."
            )
        if keys != sorted(keys):
            raise ValueError(
                f"metric observations must be sorted by (namespace, name); got {keys!r}."
            )
        return self

    def to_deterministic_json(self) -> bytes:
        return self._deterministic_json(self.model_dump(mode="json", by_alias=True))


# ---------------------------------------------------------------------------
# WriterStats
# ---------------------------------------------------------------------------


class WriterStats(BaseModel):
    """Immutable writer statistics (deterministic overhead evidence)."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True, strict=True)

    records_written: int = Field(..., ge=0)
    bytes_written: int = Field(..., ge=0)
    fsync_count: int = Field(..., ge=0)
    console_failures: int = Field(..., ge=0)
    first_sequence: int | None = Field(default=None, ge=0)
    last_sequence: int | None = Field(default=None, ge=0)
