"""Portable cross-component checks for the final Issue #9 review contract."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from expertforge.identity.fingerprint import specification_fingerprint
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.telemetry.loader import load_telemetry_stream, scan_telemetry_stream
from expertforge.telemetry.models import MetricObservation, ProcessContext, ProgressPosition
from expertforge.telemetry.writer import TelemetryWriter

pytestmark = pytest.mark.integration


def test_filtered_redacted_stream_round_trips_authoritatively(tmp_path: Path) -> None:
    identity = AttemptIdentityRecord(
        specification_fingerprint=specification_fingerprint(b'{"integration":9}'),
        run_id="run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb",
        attempt_id="attempt-20260101t000000z-cccccccccccccccccccc",
        created_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    process = ProcessContext(rank=0, world_size=1, local_rank=0)
    ticks = iter(range(20))
    writer = TelemetryWriter(
        tmp_path,
        identity,
        process,
        level="WARNING",
        console_enabled=False,
        wall_clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        monotonic_clock=lambda: next(ticks),
    )
    writer.emit_event(
        component="training",
        severity="INFO",
        event_name="training.filtered",
        progress=ProgressPosition(step=1, processed_tokens=16),
    )
    writer.emit_metric(
        component="training",
        progress=ProgressPosition(step=1, processed_tokens=16),
        observations=(
            MetricObservation(
                namespace="throughput",
                name="tokens",
                unit="tokens_per_second",
                aggregation="rate",
                window="since_last_emit",
                value_status="finite",
                value=16.0,
            ),
        ),
    )
    writer.emit_event(
        component="training",
        severity="WARNING",
        event_name="training.warning",
        fields=(("location", "/work/private/checkpoint"),),
        progress=ProgressPosition(step=2, processed_tokens=32),
    )
    writer.close("normal")
    records = load_telemetry_stream(
        writer.path,
        expected_identity=identity,
        expected_process_context=process,
    )
    assert [record.sequence for record in records] == [0, 1, 2, 3]
    assert scan_telemetry_stream(writer.path).status == "complete"
    assert b"/work/private" not in writer.path.read_bytes()


def test_truncated_closed_stream_is_never_reported_complete(tmp_path: Path) -> None:
    identity = AttemptIdentityRecord(
        specification_fingerprint=specification_fingerprint(b'{"integration":10}'),
        run_id="run-20260101t000000z-dddddddddddd-eeeeeeeeeeeeeeeeeeee",
        attempt_id="attempt-20260101t000000z-ffffffffffffffffffff",
        created_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    process = ProcessContext(rank=0, world_size=1, local_rank=0)
    ticks = iter(range(10))
    writer = TelemetryWriter(
        tmp_path,
        identity,
        process,
        console_enabled=False,
        wall_clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        monotonic_clock=lambda: next(ticks),
    )
    writer.close("normal")
    raw = writer.path.read_bytes()
    writer.path.write_bytes(raw[:-7])
    result = scan_telemetry_stream(writer.path)
    assert result.status == "incomplete"
    assert result.accepted_count == 1
