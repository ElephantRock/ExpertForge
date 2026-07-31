"""Tests for the Issue #9 telemetry core models (design comment 5136093570).

Covers: immutability, frozen/extra=forbid/strict semantics, closed Literal domains
(diagnostic codes, metric units/aggregations/windows, value statuses, reasons),
cross-field validators (process context, metric aggregation/window/value
relationships, finite/non-finite handling), bounded size limits, non-finite float
rejection, event-field ordering/uniqueness, and structural redaction.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from expertforge.telemetry.models import (
    DIAGNOSTIC_CODES,
    EVENT_SCHEMA,
    EVENT_SCHEMA_VERSION,
    MAX_EVENT_FIELDS,
    MAX_METRIC_OBSERVATIONS,
    MAX_PERSISTED_STRING,
    METRIC_AGGREGATIONS,
    METRIC_SCHEMA,
    METRIC_SCHEMA_VERSION,
    METRIC_UNITS,
    METRIC_VALUE_REASONS,
    METRIC_VALUE_STATUSES,
    METRIC_WINDOWS,
    STREAM_FORMAT_VERSION,
    EventField,
    EventRecord,
    MetricObservation,
    MetricRecord,
    MetricValueStatus,
    ProcessContext,
    ProgressPosition,
    StreamStatus,
    WriterStats,
    sanitize_operator_message,
)

_VALID_RUN = "run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb"
_VALID_ATTEMPT = "attempt-20260101t000000z-cccccccccccccccccccc"
_VALID_FINGERPRINT = "spec-v1-sha256-" + "0" * 64
_TS = datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(**overrides: object) -> EventRecord:
    base: dict[str, object] = dict(
        schema=EVENT_SCHEMA,
        schema_version=EVENT_SCHEMA_VERSION,
        stream_format_version=STREAM_FORMAT_VERSION,
        run_id=_VALID_RUN,
        attempt_id=_VALID_ATTEMPT,
        specification_fingerprint=_VALID_FINGERPRINT,
        component="training",
        rank=0,
        local_rank=0,
        world_size=1,
        sequence=0,
        timestamp_utc=_TS,
        elapsed_ns=0,
        progress=None,
        severity="INFO",
        event_name="training.update",
        diagnostic_code=None,
        fields=(),
        operator_message=None,
    )
    base.update(overrides)
    return EventRecord.model_validate(base)


def _metric(**overrides: object) -> MetricRecord:
    base: dict[str, object] = dict(
        schema=METRIC_SCHEMA,
        schema_version=METRIC_SCHEMA_VERSION,
        stream_format_version=STREAM_FORMAT_VERSION,
        run_id=_VALID_RUN,
        attempt_id=_VALID_ATTEMPT,
        specification_fingerprint=_VALID_FINGERPRINT,
        component="training",
        rank=0,
        local_rank=0,
        world_size=1,
        sequence=1,
        timestamp_utc=_TS,
        elapsed_ns=10,
        progress=None,
        observations=(_obs(),),
    )
    base.update(overrides)
    return MetricRecord.model_validate(base)


def _obs(**overrides: object) -> MetricObservation:
    base: dict[str, object] = dict(
        namespace="training",
        name="loss",
        unit="dimensionless",
        aggregation="gauge",
        window="point",
        value_status="finite",
        value=1.0,
        sample_count=None,
        reason=None,
    )
    base.update(overrides)
    return MetricObservation.model_validate(base)


# ---------------------------------------------------------------------------
# ProcessContext
# ---------------------------------------------------------------------------


class TestProcessContext:
    def test_minimal_non_distributed_run(self) -> None:
        ctx = ProcessContext(rank=0, world_size=1, local_rank=0)
        assert ctx.rank == 0 and ctx.world_size == 1 and ctx.local_rank == 0

    def test_local_rank_optional(self) -> None:
        ctx = ProcessContext(rank=0, world_size=1)
        assert ctx.local_rank is None

    def test_rank_lt_world_size_enforced(self) -> None:
        with pytest.raises(ValidationError):
            ProcessContext(rank=1, world_size=1)

    def test_local_rank_lt_world_size_when_present(self) -> None:
        with pytest.raises(ValidationError):
            ProcessContext(rank=0, world_size=2, local_rank=2)

    def test_rank_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            ProcessContext(rank=-1, world_size=2)

    def test_world_size_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            ProcessContext(rank=0, world_size=0)

    def test_local_rank_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            ProcessContext(rank=0, world_size=2, local_rank=-1)

    def test_frozen(self) -> None:
        ctx = ProcessContext(rank=0, world_size=2, local_rank=0)
        with pytest.raises(ValidationError):
            ctx.rank = 1  # type: ignore[misc]

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ProcessContext.model_validate({"rank": 0, "world_size": 1, "extra": 1})


# ---------------------------------------------------------------------------
# ProgressPosition
# ---------------------------------------------------------------------------


class TestProgressPosition:
    def test_all_optional(self) -> None:
        p = ProgressPosition()
        assert p.step is None and p.update is None and p.processed_tokens is None

    def test_partial(self) -> None:
        p = ProgressPosition(step=5)
        assert p.step == 5 and p.update is None

    def test_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            ProgressPosition(step=-1)

    def test_frozen(self) -> None:
        p = ProgressPosition(step=1)
        with pytest.raises(ValidationError):
            p.step = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EventField
# ---------------------------------------------------------------------------


class TestEventField:
    def test_accepts_str_int_float_bool(self) -> None:
        assert EventField(name="a", value="x").value == "x"
        assert EventField(name="a", value=1).value == 1
        assert EventField(name="a", value=1.5).value == 1.5
        assert EventField(name="a", value=True).value is True

    def test_name_lowercase_identifier(self) -> None:
        with pytest.raises(ValidationError):
            EventField(name="BadName", value=1)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_floats(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            EventField(name="a", value=bad)

    def test_rejects_other_types(self) -> None:
        with pytest.raises(ValidationError):
            EventField(name="a", value=[1, 2])  # type: ignore[arg-type]

    def test_frozen_extra_forbid(self) -> None:
        f = EventField(name="a", value=1)
        with pytest.raises(ValidationError):
            f.value = 2  # type: ignore[misc]
        with pytest.raises(ValidationError):
            EventField.model_validate({"name": "a", "value": 1, "x": 2})


# ---------------------------------------------------------------------------
# EventRecord
# ---------------------------------------------------------------------------


class TestEventRecord:
    def test_default_schema_and_versions(self) -> None:
        e = _event()
        assert e.schema_name == EVENT_SCHEMA == "expertforge.telemetry-event"
        assert e.schema_version == EVENT_SCHEMA_VERSION == 1
        assert e.stream_format_version == STREAM_FORMAT_VERSION == 1

    def test_frozen(self) -> None:
        e = _event()
        with pytest.raises(ValidationError):
            e.sequence = 5  # type: ignore[misc]

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            _event(unknown_field=1)

    def test_sequence_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            _event(sequence=-1)

    def test_elapsed_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            _event(elapsed_ns=-1)

    def test_timestamp_must_be_aware_utc(self) -> None:
        naive = datetime(2026, 1, 1, 0, 0, 0)
        with pytest.raises(ValidationError):
            _event(timestamp_utc=naive)

    def test_event_name_lowercase_namespaced(self) -> None:
        with pytest.raises(ValidationError):
            _event(event_name="Bad Name")

    def test_severity_closed_domain(self) -> None:
        with pytest.raises(ValidationError):
            _event(severity="TRACE")

    def test_diagnostic_code_closed_domain(self) -> None:
        with pytest.raises(ValidationError):
            _event(diagnostic_code="not_a_code")

    def test_diagnostic_code_accepted(self) -> None:
        e = _event(diagnostic_code="non_finite_loss")
        assert e.diagnostic_code == "non_finite_loss"

    def test_unknown_schema_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _event(schema="something.else")

    def test_unknown_schema_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _event(schema_version=999)

    def test_unknown_stream_format_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _event(stream_format_version=999)

    def test_fields_sorted_unique_by_name(self) -> None:
        # Design §6: non-canonical field order is rejected; callers must sort.
        fields = (
            EventField(name="a", value=1),
            EventField(name="b", value=2),
        )
        e = _event(fields=fields)
        assert [f.name for f in e.fields] == ["a", "b"]

    def test_fields_unsorted_order_rejected(self) -> None:
        fields = (
            EventField(name="b", value=2),
            EventField(name="a", value=1),
        )
        with pytest.raises(ValidationError):
            _event(fields=fields)

    def test_fields_duplicate_name_rejected(self) -> None:
        fields = (
            EventField(name="a", value=1),
            EventField(name="a", value=2),
        )
        with pytest.raises(ValidationError):
            _event(fields=fields)

    def test_fields_max_64(self) -> None:
        fields = tuple(EventField(name=f"f{i:03d}", value=i) for i in range(MAX_EVENT_FIELDS))
        e = _event(fields=fields)
        assert len(e.fields) == MAX_EVENT_FIELDS
        too_many = fields + (EventField(name="z", value=0),)
        with pytest.raises(ValidationError):
            _event(fields=too_many)

    def test_operator_message_max_length(self) -> None:
        with pytest.raises(ValidationError):
            _event(operator_message="x" * (MAX_PERSISTED_STRING + 1))

    def test_operator_message_control_chars_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _event(operator_message="line\nbreak")

    def test_serialization_is_compact_sorted_no_nan(self) -> None:
        e = _event(fields=(EventField(name="a", value=1), EventField(name="b", value=2)))
        blob = e.to_deterministic_json()
        assert b", " not in blob
        assert b": " not in blob
        # sorted keys: "a" field appears before "b" field
        assert blob.find(b'"name":"a"') < blob.find(b'"name":"b"')
        # the `schema` alias is emitted (not `schema_name`)
        assert b'"schema":"expertforge.telemetry-event"' in blob

    def test_json_round_trip(self) -> None:
        e = _event(
            fields=(EventField(name="a", value=1),),
            progress=ProgressPosition(step=3, update=3, processed_tokens=1024),
            operator_message="hello",
            diagnostic_code="optimizer_failure",
        )
        restored = EventRecord.model_validate_json(e.to_deterministic_json())
        assert restored == e

    def test_progress_round_trips(self) -> None:
        e = _event(progress=ProgressPosition(step=2))
        assert e.progress is not None and e.progress.step == 2


# ---------------------------------------------------------------------------
# MetricObservation
# ---------------------------------------------------------------------------


class TestMetricObservation:
    def test_finite_requires_value_forbids_reason(self) -> None:
        o = _obs(value_status="finite", value=2.0, reason=None)
        assert o.value == 2.0
        with pytest.raises(ValidationError):
            _obs(value_status="finite", value=None)
        with pytest.raises(ValidationError):
            _obs(value_status="finite", value=2.0, reason="not_collected")

    def test_non_finite_status_forbids_value(self) -> None:
        for status in ("nan", "positive_infinity", "negative_infinity"):
            with pytest.raises(ValidationError):
                _obs(value_status=status, value=1.0)
            o = _obs(value_status=status, value=None, reason=None)
            assert o.value is None

    @pytest.mark.parametrize("status", ["unavailable", "error", "redacted"])
    def test_status_requires_reason(self, status: str) -> None:
        with pytest.raises(ValidationError):
            _obs(value_status=status, value=None, reason=None)
        o = _obs(value_status=status, value=None, reason="not_collected")
        assert o.reason == "not_collected"

    def test_reason_closed_domain(self) -> None:
        with pytest.raises(ValidationError):
            _obs(value_status="error", value=None, reason="bogus")

    def test_value_status_closed_domain(self) -> None:
        with pytest.raises(ValidationError):
            _obs(value_status="nope")

    def test_aggregation_gauge_requires_point(self) -> None:
        with pytest.raises(ValidationError):
            _obs(aggregation="gauge", window="attempt")

    def test_aggregation_counter_requires_attempt_or_run(self) -> None:
        o = _obs(
            aggregation="counter", window="attempt", value_status="finite", value=1, sample_count=1
        )
        assert o.window == "attempt"
        with pytest.raises(ValidationError):
            _obs(
                aggregation="counter",
                window="point",
                value_status="finite",
                value=1,
                sample_count=1,
            )

    def test_delta_requires_since_last_emit(self) -> None:
        o = _obs(
            aggregation="delta",
            window="since_last_emit",
            value_status="finite",
            value=1,
            sample_count=1,
        )
        assert o.window == "since_last_emit"
        with pytest.raises(ValidationError):
            _obs(
                aggregation="delta",
                window="attempt",
                value_status="finite",
                value=1,
                sample_count=1,
            )

    def test_rate_requires_since_last_emit(self) -> None:
        o = _obs(
            aggregation="rate",
            window="since_last_emit",
            unit="tokens_per_second",
            value_status="finite",
            value=100.0,
            sample_count=1,
        )
        assert o.window == "since_last_emit"
        with pytest.raises(ValidationError):
            _obs(
                aggregation="rate",
                window="attempt",
                unit="tokens_per_second",
                value_status="finite",
                value=100.0,
                sample_count=1,
            )

    @pytest.mark.parametrize("agg", ["sum", "mean", "minimum", "maximum"])
    def test_aggregations_require_sample_count_and_non_point_window(self, agg: str) -> None:
        # missing sample_count
        with pytest.raises(ValidationError):
            _obs(
                aggregation=agg,
                window="attempt",
                value_status="finite",
                value=1.0,
                sample_count=None,
            )
        # point window
        with pytest.raises(ValidationError):
            _obs(
                aggregation=agg,
                window="point",
                value_status="finite",
                value=1.0,
                sample_count=2,
            )
        # valid
        o = _obs(
            aggregation=agg,
            window="attempt",
            value_status="finite",
            value=1.0,
            sample_count=2,
        )
        assert o.sample_count == 2

    def test_sample_count_positive(self) -> None:
        with pytest.raises(ValidationError):
            _obs(
                aggregation="mean",
                window="attempt",
                value_status="finite",
                value=1.0,
                sample_count=0,
            )

    def test_finite_integer_value_accepted(self) -> None:
        o = _obs(value_status="finite", value=5)
        assert o.value == 5 and isinstance(o.value, int)

    def test_finite_nan_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _obs(value_status="finite", value=math.nan)

    def test_frozen_extra_forbid(self) -> None:
        o = _obs()
        with pytest.raises(ValidationError):
            o.value = 9.0  # type: ignore[misc]

    def test_unit_closed_domain(self) -> None:
        with pytest.raises(ValidationError):
            _obs(unit="widgets")

    def test_aggregation_closed_domain(self) -> None:
        with pytest.raises(ValidationError):
            _obs(aggregation="median")

    def test_window_closed_domain(self) -> None:
        with pytest.raises(ValidationError):
            _obs(window="epoch")

    def test_round_trip(self) -> None:
        o = _obs()
        restored = MetricObservation.model_validate_json(o.to_deterministic_json())
        assert restored == o


# ---------------------------------------------------------------------------
# MetricRecord
# ---------------------------------------------------------------------------


class TestMetricRecord:
    def test_default_schema_and_versions(self) -> None:
        m = _metric()
        assert m.schema_name == METRIC_SCHEMA == "expertforge.metric-record"
        assert m.schema_version == METRIC_SCHEMA_VERSION == 1

    def test_observations_sorted_unique_by_namespace_name(self) -> None:
        # Design §7: observations must be sorted+unique by (namespace, name).
        obs = (
            _obs(namespace="a", name="x"),
            _obs(namespace="b", name="x"),
        )
        m = _metric(observations=obs)
        assert [(o.namespace, o.name) for o in m.observations] == [("a", "x"), ("b", "x")]

    def test_observations_unsorted_order_rejected(self) -> None:
        obs = (
            _obs(namespace="b", name="x"),
            _obs(namespace="a", name="x"),
        )
        with pytest.raises(ValidationError):
            _metric(observations=obs)

    def test_observations_duplicate_key_rejected(self) -> None:
        obs = (
            _obs(namespace="a", name="x"),
            _obs(namespace="a", name="x"),
        )
        with pytest.raises(ValidationError):
            _metric(observations=obs)

    def test_observations_max_128(self) -> None:
        obs = tuple(_obs(namespace=f"ns{i:04d}", name="x") for i in range(MAX_METRIC_OBSERVATIONS))
        m = _metric(observations=obs)
        assert len(m.observations) == MAX_METRIC_OBSERVATIONS
        too_many = obs + (_obs(namespace="zzzz", name="x"),)
        with pytest.raises(ValidationError):
            _metric(observations=too_many)

    def test_frozen(self) -> None:
        m = _metric()
        with pytest.raises(ValidationError):
            m.sequence = 9  # type: ignore[misc]

    def test_round_trip(self) -> None:
        m = _metric(
            observations=(
                _obs(namespace="training", name="loss"),
                _obs(namespace="training", name="ppl", value=10.0),
            ),
            progress=ProgressPosition(step=1),
        )
        restored = MetricRecord.model_validate_json(m.to_deterministic_json())
        assert restored == m

    def test_no_json_nan_on_non_finite(self) -> None:
        m = _metric(
            observations=(
                _obs(
                    namespace="training", name="loss", value_status="nan", value=None, reason=None
                ),
            )
        )
        blob = m.to_deterministic_json()
        assert b"NaN" not in blob
        assert b"Infinity" not in blob
        assert b'"value_status":"nan"' in blob


# ---------------------------------------------------------------------------
# WriterStats / StreamStatus / closed-domain sets
# ---------------------------------------------------------------------------


class TestWriterStatsAndDomains:
    def test_writer_stats_frozen(self) -> None:
        s = WriterStats(
            records_written=1,
            bytes_written=10,
            fsync_count=1,
            console_failures=0,
            first_sequence=0,
            last_sequence=0,
        )
        with pytest.raises(ValidationError):
            s.records_written = 2  # type: ignore[misc]

    def test_stream_status_domain(self) -> None:
        assert set(StreamStatus.__args__) == {"complete", "incomplete", "corrupt"}  # type: ignore[attr-defined]

    def test_metric_value_status_domain(self) -> None:
        assert set(MetricValueStatus.__args__) == METRIC_VALUE_STATUSES  # type: ignore[attr-defined]

    def test_diagnostic_codes_present(self) -> None:
        for code in (
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
        ):
            assert code in DIAGNOSTIC_CODES

    def test_unit_aggregation_window_domains_nonempty(self) -> None:
        for u in ("tokens_per_second", "bytes_per_second", "dimensionless", "steps"):
            assert u in METRIC_UNITS
        for a in ("gauge", "counter", "delta", "sum", "mean", "minimum", "maximum", "rate"):
            assert a in METRIC_AGGREGATIONS
        for w in ("point", "since_last_emit", "attempt", "run", "evaluation"):
            assert w in METRIC_WINDOWS

    def test_metric_value_reasons_domain(self) -> None:
        for r in (
            "not_collected",
            "not_applicable",
            "provider_unavailable",
            "input_non_finite",
            "division_by_zero",
            "overflow",
            "computation_failed",
            "sensitive_value_redacted",
        ):
            assert r in METRIC_VALUE_REASONS


# ---------------------------------------------------------------------------
# Structural redaction (operator message sanitizer)
# ---------------------------------------------------------------------------


class TestRedaction:
    @pytest.mark.parametrize(
        "raw,redacted",
        [
            ("see https://user:secrets@example.com/path", True),
            ("token=abcdef1234567890", True),
            ("password=hunter2", True),
            ("secret=s3cr3tvalue", True),
            ("my home is /Users/alice", True),
            ("endpoint http://10.0.0.1:8080 here", True),
        ],
    )
    def test_sensitive_patterns_redacted(self, raw: str, redacted: bool) -> None:
        out = sanitize_operator_message(raw)
        # the original literal must not survive
        for needle in ("secrets@example", "hunter2", "s3cr3tvalue", "abcdef1234567890"):
            if needle in raw:
                assert needle not in out, f"{needle!r} survived sanitization"
        # redaction diagnostic is recorded structurally
        if redacted:
            assert "[redacted]" in out

    def test_sanitizer_truncates_to_max(self) -> None:
        out = sanitize_operator_message("x" * (MAX_PERSISTED_STRING + 100))
        assert len(out) <= MAX_PERSISTED_STRING

    def test_sanitizer_rejects_control_chars(self) -> None:
        with pytest.raises(ValueError):
            sanitize_operator_message("bad\nnewline")
