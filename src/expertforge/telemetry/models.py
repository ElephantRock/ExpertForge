"""Frozen telemetry records and closed domains for Issue #9."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

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
    "Severity",
    "StreamOutcome",
    "StreamStatus",
    "WriterStats",
    "metric_semantic_key",
    "redacted",
    "sanitize_operator_message",
    "sanitize_persisted_string",
]

STREAM_FORMAT_VERSION = 1
EVENT_SCHEMA_VERSION = 1
METRIC_SCHEMA_VERSION = 1
EVENT_SCHEMA = "expertforge.telemetry-event"
METRIC_SCHEMA = "expertforge.metric-record"

MAX_RECORD_BYTES = 64 * 1024
MAX_EVENT_FIELDS = 64
MAX_METRIC_OBSERVATIONS = 128
MAX_PERSISTED_STRING = 4096

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

MetricValueStatus = Literal[
    "finite",
    "nan",
    "positive_infinity",
    "negative_infinity",
    "unavailable",
    "error",
    "redacted",
]
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

MetricValueReason = Literal[
    "not_collected",
    "not_applicable",
    "provider_unavailable",
    "input_non_finite",
    "division_by_zero",
    "overflow",
    "computation_failed",
    "sensitive_value_redacted",
]
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

MetricUnit = Literal[
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

MetricAggregation = Literal[
    "gauge",
    "counter",
    "delta",
    "sum",
    "mean",
    "minimum",
    "maximum",
    "rate",
]
METRIC_AGGREGATIONS: frozenset[str] = frozenset(
    {"gauge", "counter", "delta", "sum", "mean", "minimum", "maximum", "rate"}
)

MetricWindow = Literal["point", "since_last_emit", "attempt", "run", "evaluation"]
METRIC_WINDOWS: frozenset[str] = frozenset(
    {"point", "since_last_emit", "attempt", "run", "evaluation"}
)

Severity = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
StreamStatus = Literal["complete", "incomplete", "corrupt"]
StreamOutcome = Literal["normal", "interrupted", "failed"]

_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_FINGERPRINT_RE = re.compile(r"^spec-v1-sha256-[0-9a-f]{64}$")
_RATE_UNITS = frozenset({"tokens_per_second", "bytes_per_second"})

_RE_URL = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_RE_KV_SECRET = re.compile(
    r"(?i)\b(token|password|passwd|secret|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|credential|auth|bearer)\b\s*[:=]\s*\S+"
)
_RE_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b")
_RE_IPV6 = re.compile(
    r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,}[0-9A-Fa-f]{0,4}"
    r"(?::\d{1,5})?(?![0-9A-Fa-f:])"
)
_RE_HOME_PATH = re.compile(r"(?i)(?:~|/(?:users|home|root))/[^\s\"']+")
_RE_WIN_PATH = re.compile(r"\b[A-Za-z]:[\\/][^\s\"'<>|]+")
_RE_UNIX_ABS_PATH = re.compile(r"(?<!:)(?<![A-Za-z0-9])/(?!/)[^\s\"']+")
_RE_DEVICE_UUID = re.compile(
    r"(?i)\b(?:gpu-|device[_-]?uuid\s*[:=]\s*)?[0-9a-f]{8}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)
_RE_HOST_ASSIGNMENT = re.compile(
    r"(?i)\b(?:host|hostname|user|username|serial|device[_-]?serial)\b\s*[:=]\s*\S+"
)


def _contains_control(value: str) -> bool:
    return any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)


def _sanitize_url(match: re.Match[str]) -> str:
    raw = match.group(0)
    try:
        parsed = urlsplit(raw)
        hostname = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        return "[redacted]"
    sensitive = (
        parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
        or _RE_IPV4.fullmatch(hostname) is not None
        or ":" in hostname
    )
    if sensitive:
        return "[redacted]"
    return urlunsplit((parsed.scheme, f"{hostname}{port}", parsed.path, "", ""))


def sanitize_persisted_string(value: str, *, truncate: bool = False) -> tuple[str, bool]:
    """Sanitize one persisted string and report whether it changed."""
    if _contains_control(value):
        raise ValueError("persisted string contains forbidden control characters.")
    original = value
    sanitized = _RE_URL.sub(_sanitize_url, value)
    sanitized = _RE_KV_SECRET.sub(
        lambda match: f"{match.group(1)}=[redacted]", sanitized
    )
    sanitized = _RE_IPV4.sub("[redacted]", sanitized)
    sanitized = _RE_IPV6.sub("[redacted]", sanitized)
    sanitized = _RE_HOME_PATH.sub("[redacted]", sanitized)
    sanitized = _RE_WIN_PATH.sub("[redacted]", sanitized)
    sanitized = _RE_UNIX_ABS_PATH.sub("[redacted]", sanitized)
    sanitized = _RE_DEVICE_UUID.sub("[redacted]", sanitized)
    sanitized = _RE_HOST_ASSIGNMENT.sub(
        lambda match: match.group(0).split("=", 1)[0].split(":", 1)[0]
        + "=[redacted]",
        sanitized,
    )
    if truncate and len(sanitized) > MAX_PERSISTED_STRING:
        sanitized = sanitized[:MAX_PERSISTED_STRING]
    elif len(sanitized) > MAX_PERSISTED_STRING:
        raise ValueError(
            f"persisted string length {len(sanitized)} exceeds max {MAX_PERSISTED_STRING}."
        )
    return sanitized, sanitized != original


def redacted(text: str) -> str:
    """Return the defense-in-depth sanitized representation."""
    return sanitize_persisted_string(text, truncate=True)[0]


def sanitize_operator_message(message: str) -> str:
    """Sanitize and truncate an operator message for writer use."""
    return sanitize_persisted_string(message, truncate=True)[0]


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("timestamp_utc must be timezone-aware UTC.")
    if value.tzinfo.utcoffset(value) != timedelta(0):
        raise ValueError("timestamp_utc must use UTC offset zero.")
    return value


def _format_utc(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond:06d}Z"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
        populate_by_name=True,
    )


class ProcessContext(_FrozenModel):
    rank: int = Field(..., ge=0)
    world_size: int = Field(..., ge=1)
    local_rank: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _topology(self) -> ProcessContext:
        if self.rank >= self.world_size:
            raise ValueError("rank must be less than world_size.")
        if self.local_rank is not None and self.local_rank >= self.world_size:
            raise ValueError("local_rank must be less than world_size.")
        return self


class ProgressPosition(_FrozenModel):
    step: int | None = Field(default=None, ge=0)
    update: int | None = Field(default=None, ge=0)
    processed_tokens: int | None = Field(default=None, ge=0)


class EventField(_FrozenModel):
    name: str = Field(..., min_length=1, max_length=MAX_PERSISTED_STRING)
    value: str | int | float | bool

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError("event field name must be a lowercase stable identifier.")
        return value

    @field_validator("value")
    @classmethod
    def _value(cls, value: str | int | float | bool) -> str | int | float | bool:
        if isinstance(value, str):
            return sanitize_persisted_string(value)[0]
        if isinstance(value, bool):
            return value
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("event float values must be finite.")
        if isinstance(value, (int, float)):
            return value
        raise ValueError("unsupported event field value type.")


class _RecordBase(_FrozenModel):
    stream_format_version: int = Field(default=STREAM_FORMAT_VERSION)
    run_id: str = Field(..., min_length=1, max_length=MAX_PERSISTED_STRING)
    attempt_id: str = Field(..., min_length=1, max_length=MAX_PERSISTED_STRING)
    specification_fingerprint: str = Field(..., min_length=1)
    component: str = Field(..., min_length=1, max_length=MAX_PERSISTED_STRING)
    rank: int = Field(..., ge=0)
    local_rank: int | None = Field(default=None, ge=0)
    world_size: int = Field(..., ge=1)
    sequence: int = Field(..., ge=0)
    timestamp_utc: datetime
    elapsed_ns: int = Field(..., ge=0)
    progress: ProgressPosition | None = None

    @field_validator("stream_format_version")
    @classmethod
    def _stream_version(cls, value: int) -> int:
        if value != STREAM_FORMAT_VERSION:
            raise ValueError(f"unsupported stream format version {value}.")
        return value

    @field_validator("specification_fingerprint")
    @classmethod
    def _fingerprint(cls, value: str) -> str:
        if not _FINGERPRINT_RE.fullmatch(value):
            raise ValueError("invalid specification fingerprint.")
        return value

    @field_validator("component")
    @classmethod
    def _component(cls, value: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError("component must be a lowercase stable identifier.")
        return value

    @field_validator("timestamp_utc")
    @classmethod
    def _timestamp(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @field_serializer("timestamp_utc")
    def _serialize_timestamp(self, value: datetime) -> str:
        return _format_utc(value)

    @model_validator(mode="after")
    def _topology(self) -> _RecordBase:
        if self.rank >= self.world_size:
            raise ValueError("rank must be less than world_size.")
        if self.local_rank is not None and self.local_rank >= self.world_size:
            raise ValueError("local_rank must be less than world_size.")
        return self

    def _json_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


class EventRecord(_RecordBase):
    schema_name: Literal["expertforge.telemetry-event"] = Field(
        default=EVENT_SCHEMA, alias="schema"
    )
    schema_version: int = Field(default=EVENT_SCHEMA_VERSION)
    severity: Severity
    event_name: str = Field(..., min_length=1, max_length=MAX_PERSISTED_STRING)
    diagnostic_code: DiagnosticCode | None = None
    fields: tuple[EventField, ...] = Field(default_factory=tuple)
    operator_message: str | None = Field(default=None, max_length=MAX_PERSISTED_STRING)

    @field_validator("schema_version")
    @classmethod
    def _event_version(cls, value: int) -> int:
        if value != EVENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported event schema version {value}.")
        return value

    @field_validator("event_name")
    @classmethod
    def _event_name(cls, value: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError("event_name must be a lowercase namespaced identifier.")
        return value

    @field_validator("operator_message")
    @classmethod
    def _message(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return sanitize_persisted_string(value)[0]

    @model_validator(mode="after")
    def _event_invariants(self) -> EventRecord:
        if len(self.fields) > MAX_EVENT_FIELDS:
            raise ValueError(f"event fields exceed max {MAX_EVENT_FIELDS}.")
        names = tuple(field.name for field in self.fields)
        if names != tuple(sorted(names)):
            raise ValueError("event fields must be sorted by name.")
        if len(set(names)) != len(names):
            raise ValueError("event fields must be unique by name.")

        replacement_present = (
            self.operator_message is not None and "[redacted]" in self.operator_message
        ) or any(
            isinstance(field.value, str) and "[redacted]" in field.value
            for field in self.fields
        )
        if replacement_present and self.diagnostic_code != "redacted_sensitive_value":
            raise ValueError(
                "redacted persisted strings require diagnostic_code "
                "'redacted_sensitive_value'."
            )

        if self.event_name == "logging.stream_opened":
            if (
                self.sequence != 0
                or self.elapsed_ns != 0
                or self.component != "logging"
                or self.severity != "INFO"
                or self.diagnostic_code is not None
                or self.fields
                or self.operator_message is not None
                or self.progress is not None
            ):
                raise ValueError("logging.stream_opened has a fixed lifecycle shape.")

        if self.event_name == "logging.stream_closed":
            if (
                self.component != "logging"
                or self.severity != "INFO"
                or self.operator_message is not None
                or self.progress is not None
                or len(self.fields) != 1
                or self.fields[0].name != "outcome"
                or not isinstance(self.fields[0].value, str)
                or self.fields[0].value not in {"normal", "interrupted", "failed"}
            ):
                raise ValueError("logging.stream_closed has an invalid lifecycle shape.")
            outcome = self.fields[0].value
            if outcome == "normal" and self.diagnostic_code is not None:
                raise ValueError("normal close forbids a diagnostic code.")
            if outcome == "interrupted" and self.diagnostic_code != "handled_interruption":
                raise ValueError(
                    "interrupted close requires diagnostic_code 'handled_interruption'."
                )
            if outcome == "failed" and self.diagnostic_code in {
                None,
                "handled_interruption",
            }:
                raise ValueError("failed close requires a failure diagnostic code.")
        return self

    def to_deterministic_json(self) -> bytes:
        return self._json_bytes()


class MetricObservation(_FrozenModel):
    namespace: str = Field(..., min_length=1, max_length=MAX_PERSISTED_STRING)
    name: str = Field(..., min_length=1, max_length=MAX_PERSISTED_STRING)
    unit: MetricUnit
    aggregation: MetricAggregation
    window: MetricWindow
    value_status: MetricValueStatus
    value: int | float | None = None
    sample_count: int | None = Field(default=None, ge=1)
    reason: MetricValueReason | None = None

    @field_validator("namespace", "name")
    @classmethod
    def _metric_name(cls, value: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError("metric namespace/name must be lowercase identifiers.")
        return value

    @field_validator("value")
    @classmethod
    def _metric_value(cls, value: int | float | None) -> int | float | None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("metric numeric values must be finite.")
        return value

    @model_validator(mode="after")
    def _relationships(self) -> MetricObservation:
        if self.value_status == "finite":
            if self.value is None or self.reason is not None:
                raise ValueError("finite status requires a value and forbids a reason.")
        else:
            if self.value is not None:
                raise ValueError("non-finite statuses forbid numeric values.")
            if self.value_status in {"unavailable", "error", "redacted"}:
                if self.reason is None:
                    raise ValueError(f"{self.value_status} status requires a reason.")
            elif self.reason is not None:
                raise ValueError(f"{self.value_status} status forbids a reason.")

        if self.aggregation == "gauge" and self.window != "point":
            raise ValueError("gauge requires point window.")
        if self.aggregation == "counter" and self.window not in {"attempt", "run"}:
            raise ValueError("counter requires attempt or run window.")
        if self.aggregation in {"delta", "rate"} and self.window != "since_last_emit":
            raise ValueError("delta/rate require since_last_emit window.")
        if self.aggregation in {"sum", "mean", "minimum", "maximum"}:
            if self.window == "point" or self.sample_count is None:
                raise ValueError(
                    "sum/mean/minimum/maximum require non-point window and sample_count."
                )
        if self.aggregation == "rate" and self.unit not in _RATE_UNITS:
            raise ValueError("rate requires a per-second unit.")
        return self

    def to_deterministic_json(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


def metric_semantic_key(
    observation: MetricObservation,
) -> tuple[str, str, str, str, str]:
    return (
        observation.namespace,
        observation.name,
        observation.unit,
        observation.aggregation,
        observation.window,
    )


class MetricRecord(_RecordBase):
    schema_name: Literal["expertforge.metric-record"] = Field(
        default=METRIC_SCHEMA, alias="schema"
    )
    schema_version: int = Field(default=METRIC_SCHEMA_VERSION)
    observations: tuple[MetricObservation, ...] = Field(..., min_length=1)

    @field_validator("schema_version")
    @classmethod
    def _metric_version(cls, value: int) -> int:
        if value != METRIC_SCHEMA_VERSION:
            raise ValueError(f"unsupported metric schema version {value}.")
        return value

    @model_validator(mode="after")
    def _observation_invariants(self) -> MetricRecord:
        if len(self.observations) > MAX_METRIC_OBSERVATIONS:
            raise ValueError(
                f"metric observations exceed max {MAX_METRIC_OBSERVATIONS}."
            )
        keys = tuple(metric_semantic_key(observation) for observation in self.observations)
        if keys != tuple(sorted(keys)):
            raise ValueError("metric observations must use canonical semantic order.")
        if len(set(keys)) != len(keys):
            raise ValueError("metric observations must be unique by semantic key.")
        return self

    def to_deterministic_json(self) -> bytes:
        return self._json_bytes()


class WriterStats(_FrozenModel):
    records_written: int = Field(..., ge=0)
    bytes_written: int = Field(..., ge=0)
    fsync_count: int = Field(..., ge=0)
    console_failures: int = Field(..., ge=0)
    first_sequence: int | None = Field(default=None, ge=0)
    last_sequence: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _coherence(self) -> WriterStats:
        if self.records_written == 0:
            if self.first_sequence is not None or self.last_sequence is not None:
                raise ValueError("empty writer stats cannot contain sequence bounds.")
        elif self.first_sequence is None or self.last_sequence is None:
            raise ValueError("non-empty writer stats require sequence bounds.")
        elif self.first_sequence > self.last_sequence:
            raise ValueError("first_sequence cannot exceed last_sequence.")
        return self
