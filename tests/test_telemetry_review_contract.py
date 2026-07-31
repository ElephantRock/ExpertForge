"""Regressions for the final Issue #9 review contract."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from expertforge.identity.fingerprint import specification_fingerprint
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.telemetry.loader import (
    TelemetryLoadError,
    load_telemetry_stream,
    scan_telemetry_stream,
)
from expertforge.telemetry.models import (
    MAX_RECORD_BYTES,
    EventField,
    EventRecord,
    MetricObservation,
    MetricRecord,
    ProcessContext,
    ProgressPosition,
    sanitize_persisted_string,
)
from expertforge.telemetry.writer import TelemetryWriter, WriterFailedError

_RUN_ID = "run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb"
_ATTEMPT_ID = "attempt-20260101t000000z-cccccccccccccccccccc"
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _identity() -> AttemptIdentityRecord:
    return AttemptIdentityRecord(
        specification_fingerprint=specification_fingerprint(b'{"telemetry":1}'),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        created_at_utc=_NOW,
    )


def _context() -> ProcessContext:
    return ProcessContext(rank=0, world_size=1, local_rank=0)


class _Clock:
    def __init__(self, *values: int) -> None:
        self._values = iter(values or tuple(range(20)))

    def __call__(self) -> int:
        return next(self._values)


class _Console:
    def __init__(self, fail_on: int, *, short: bool = False) -> None:
        self.fail_on = fail_on
        self.short = short
        self.calls = 0

    def write(self, data: str) -> int:
        self.calls += 1
        if self.calls == self.fail_on:
            if self.short:
                return max(0, len(data) - 1)
            raise ValueError("closed console")
        return len(data)

    def flush(self) -> None:
        return None


def _writer(
    root: Path,
    *,
    monotonic: _Clock | None = None,
    console: _Console | None = None,
    level: str = "INFO",
) -> TelemetryWriter:
    return TelemetryWriter(
        root,
        _identity(),
        _context(),
        console_enabled=console is not None,
        console_stream=console,
        wall_clock=lambda: _NOW,
        monotonic_clock=monotonic or _Clock(),
        level=level,
    )


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _canonical(data: dict[str, Any]) -> bytes:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8") + b"\n"


def test_writer_cannot_append_elapsed_regression(tmp_path: Path) -> None:
    writer = _writer(tmp_path, monotonic=_Clock(0, 10, 5))
    writer.emit_event(component="training", severity="INFO", event_name="training.update")
    before = writer.path.read_bytes()
    with pytest.raises(WriterFailedError, match="regressed"):
        writer.emit_event(component="training", severity="INFO", event_name="training.update")
    assert writer.path.read_bytes() == before
    loaded = load_telemetry_stream(
        writer.path,
        expected_identity=_identity(),
        expected_process_context=_context(),
        require_closed=False,
    )
    assert [record.sequence for record in loaded] == [0, 1]


def test_progress_regression_does_not_allocate_sequence(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.emit_event(
        component="training",
        severity="INFO",
        event_name="training.update",
        progress=ProgressPosition(step=5, update=2, processed_tokens=100),
    )
    with pytest.raises(ValueError, match="regresses"):
        writer.emit_event(
            component="training",
            severity="INFO",
            event_name="training.update",
            progress=ProgressPosition(step=4),
        )
    writer.emit_event(
        component="training",
        severity="INFO",
        event_name="training.update",
        progress=ProgressPosition(step=6, update=3, processed_tokens=120),
    )
    writer.close("normal")
    assert [record["sequence"] for record in _records(writer.path)] == [0, 1, 2, 3]


def test_initial_validation_failure_removes_exclusive_path(tmp_path: Path) -> None:
    with pytest.raises(WriterFailedError, match="initialization"):
        TelemetryWriter(
            tmp_path,
            _identity(),
            _context(),
            console_enabled=False,
            wall_clock=lambda: datetime(2026, 1, 1),
            monotonic_clock=_Clock(),
        )
    path = (
        tmp_path
        / _RUN_ID
        / "attempts"
        / _ATTEMPT_ID
        / "logs"
        / "telemetry-rank-0000000000.jsonl"
    )
    assert not path.exists()
    retry = _writer(tmp_path)
    retry.close("normal")


def test_terminal_console_failure_does_not_append_after_close(tmp_path: Path) -> None:
    console = _Console(2)
    writer = _writer(tmp_path, console=console)
    writer.close("normal")
    loaded = load_telemetry_stream(
        writer.path,
        expected_identity=_identity(),
        expected_process_context=_context(),
    )
    assert [getattr(record, "event_name", None) for record in loaded] == [
        "logging.stream_opened",
        "logging.stream_closed",
    ]
    assert writer.stats.console_failures == 1


def test_console_short_write_degrades_once(tmp_path: Path) -> None:
    console = _Console(1, short=True)
    writer = _writer(tmp_path, console=console)
    writer.emit_event(component="training", severity="INFO", event_name="training.update")
    writer.close("normal")
    diagnostics = [
        record
        for record in _records(writer.path)
        if record.get("diagnostic_code") == "console_write_failed"
    ]
    assert len(diagnostics) == 1
    assert console.calls == 1


def test_context_manager_records_unhandled_exception_as_failed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        with _writer(tmp_path) as writer:
            path = writer.path
            raise RuntimeError("not persisted")
    loaded = load_telemetry_stream(
        path,
        expected_identity=_identity(),
        expected_process_context=_context(),
    )
    close = loaded[-1]
    assert isinstance(close, EventRecord)
    assert close.fields == (EventField(name="outcome", value="failed"),)
    assert close.diagnostic_code == "unhandled_exception"


def test_severity_filter_preserves_metrics_and_sequence(tmp_path: Path) -> None:
    writer = _writer(tmp_path, level="WARNING")
    writer.emit_event(component="training", severity="DEBUG", event_name="training.debug")
    writer.emit_event(component="training", severity="INFO", event_name="training.info")
    writer.emit_metric(
        component="training",
        observations=(
            MetricObservation(
                namespace="training",
                name="loss",
                unit="dimensionless",
                aggregation="gauge",
                window="point",
                value_status="finite",
                value=1.0,
            ),
        ),
    )
    writer.emit_event(
        component="training",
        severity="WARNING",
        event_name="training.warning",
    )
    writer.close("normal")
    loaded = load_telemetry_stream(
        writer.path,
        expected_identity=_identity(),
        expected_process_context=_context(),
    )
    assert [record.sequence for record in loaded] == [0, 1, 2, 3]
    assert isinstance(loaded[1], MetricRecord)


def test_complete_metric_semantic_key_allows_distinct_meanings(tmp_path: Path) -> None:
    observations = (
        MetricObservation(
            namespace="training",
            name="tokens",
            unit="count",
            aggregation="counter",
            window="attempt",
            value_status="finite",
            value=10,
        ),
        MetricObservation(
            namespace="training",
            name="tokens",
            unit="tokens_per_second",
            aggregation="rate",
            window="since_last_emit",
            value_status="finite",
            value=5.0,
        ),
    )
    writer = _writer(tmp_path)
    writer.emit_metric(component="training", observations=tuple(reversed(observations)))
    writer.close("normal")
    loaded = load_telemetry_stream(
        writer.path,
        expected_identity=_identity(),
        expected_process_context=_context(),
    )
    metric = loaded[1]
    assert isinstance(metric, MetricRecord)
    assert metric.observations == observations


def test_every_persisted_event_string_is_sanitized_with_evidence(tmp_path: Path) -> None:
    secret = "pass" + "word=hunter2"
    writer = _writer(tmp_path)
    writer.emit_event(
        component="training",
        severity="WARNING",
        event_name="training.warning",
        fields=(("detail", secret), ("path", "/workspace/private/run.log")),
        operator_message="endpoint=[2001:db8::1]:9000",
    )
    writer.close("normal")
    event = _records(writer.path)[1]
    serialized = json.dumps(event)
    assert "hunter2" not in serialized
    assert "2001:db8" not in serialized
    assert "/workspace/private" not in serialized
    assert event["diagnostic_code"] == "redacted_sensitive_value"


def test_query_bearing_url_uses_explicit_redaction_marker() -> None:
    sanitized, changed = sanitize_persisted_string(
        "https://example.test/model?signature=abcdef123456"
    )
    assert changed is True
    assert sanitized == "[redacted]"


def test_loader_rejects_noncanonical_timestamp_and_oversized_line(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.close("normal")
    raw_lines = writer.path.read_bytes().splitlines()
    opened = json.loads(raw_lines[0])
    opened["timestamp_utc"] = "2026-01-01T00:00:00Z"
    bad_timestamp = tmp_path / "bad-timestamp.jsonl"
    bad_timestamp.write_bytes(_canonical(opened) + raw_lines[1] + b"\n")
    with pytest.raises(TelemetryLoadError, match="timestamp"):
        load_telemetry_stream(
            bad_timestamp,
            expected_identity=_identity(),
            expected_process_context=_context(),
        )
    assert scan_telemetry_stream(bad_timestamp).status == "corrupt"

    oversized = tmp_path / "oversized.jsonl"
    oversized.write_bytes(b" " * MAX_RECORD_BYTES + raw_lines[0] + b"\n")
    with pytest.raises(TelemetryLoadError, match="size"):
        load_telemetry_stream(
            oversized,
            expected_identity=_identity(),
            expected_process_context=_context(),
        )
    assert scan_telemetry_stream(oversized).status == "corrupt"


def test_scan_enforces_open_record_and_constant_internal_binding(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.emit_event(component="training", severity="INFO", event_name="training.update")
    writer.close("normal")
    raw_lines = writer.path.read_bytes().splitlines()

    first = json.loads(raw_lines[0])
    first["component"] = "training"
    first["event_name"] = "training.update"
    no_open = tmp_path / "no-open.jsonl"
    no_open.write_bytes(_canonical(first) + raw_lines[-1] + b"\n")
    assert scan_telemetry_stream(no_open).status == "corrupt"

    switched = json.loads(raw_lines[1])
    switched["attempt_id"] = "attempt-switched"
    switched_path = tmp_path / "switched.jsonl"
    switched_path.write_bytes(
        raw_lines[0] + b"\n" + _canonical(switched) + raw_lines[-1] + b"\n"
    )
    assert scan_telemetry_stream(switched_path).status == "corrupt"


def test_loader_rejects_malformed_close_shape(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.close("normal")
    raw_lines = writer.path.read_bytes().splitlines()
    malformed = json.loads(raw_lines[-1])
    malformed["fields"] = []
    path = tmp_path / "malformed-close.jsonl"
    path.write_bytes(raw_lines[0] + b"\n" + _canonical(malformed))
    with pytest.raises(TelemetryLoadError):
        load_telemetry_stream(
            path,
            expected_identity=_identity(),
            expected_process_context=_context(),
        )
    assert scan_telemetry_stream(path).status == "corrupt"


def test_zero_microseconds_are_serialized_fixed_width(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.close("normal")
    assert b'"timestamp_utc":"2026-01-01T00:00:00.000000Z"' in writer.path.read_bytes()
