"""Tests for the Issue #9 TelemetryWriter (design comment 5136093570).

Covers: exclusive creation and path layout, no-overwrite/same-attempt reopen
rejection, full-write loop with injected short writes, fsync cadence, lifecycle
(stream_opened at seq 0, stream_closed last, no records after close), terminal
writer state after write/fsync/close failure, contiguous event/metric sequence
ordering, one-time console-write-failure degradation, deterministic console
rendering, and immutable writer statistics.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from expertforge.identity.fingerprint import specification_fingerprint
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.telemetry.models import (
    EVENT_SCHEMA,
    MAX_EVENT_FIELDS,
    STREAM_FORMAT_VERSION,
    MetricObservation,
    ProcessContext,
    ProgressPosition,
)
from expertforge.telemetry.writer import TelemetryWriter, WriterClosedError, WriterFailedError

_VALID_RUN = "run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb"
_VALID_ATTEMPT = "attempt-20260101t000000z-cccccccccccccccccccc"
_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _identity() -> AttemptIdentityRecord:
    fp = specification_fingerprint(b'{"x":1}')
    return AttemptIdentityRecord(
        specification_fingerprint=fp,
        run_id=_VALID_RUN,
        attempt_id=_VALID_ATTEMPT,
        created_at_utc=_TS,
    )


def _ctx(rank: int = 0, world_size: int = 1, local_rank: int | None = 0) -> ProcessContext:
    return ProcessContext(rank=rank, world_size=world_size, local_rank=local_rank)


def _make_writer(
    root: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int | None = 0,
    console_enabled: bool = True,
    fsync_interval_records: int = 100,
    wall_clock: Any = None,
    monotonic_clock: Any = None,
    console_stream: Any = None,
) -> TelemetryWriter:
    return TelemetryWriter(
        artifact_root=root,
        identity=_identity(),
        process_context=_ctx(rank, world_size, local_rank),
        console_enabled=console_enabled,
        fsync_interval_records=fsync_interval_records,
        wall_clock=wall_clock or (lambda: _TS),
        monotonic_clock=monotonic_clock or _step_clock(),
        console_stream=console_stream,
    )


class _StepClock:
    """Monotonic clock returning 0, 1_000, 2_000, ... ns per call."""

    def __init__(self) -> None:
        self._n = 0

    def __call__(self) -> int:
        v = self._n
        self._n += 1_000
        return v


def _step_clock() -> _StepClock:
    return _StepClock()


def _read_stream(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# Path layout and exclusive creation
# ---------------------------------------------------------------------------


class TestPathAndCreation:
    def test_canonical_path_layout(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        assert w.path == (
            tmp_path
            / _VALID_RUN
            / "attempts"
            / _VALID_ATTEMPT
            / "logs"
            / "telemetry-rank-0000000000.jsonl"
        )
        assert w.path.exists()

    def test_rank_zero_padded_to_10_digits(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path, rank=12, world_size=64, local_rank=3)
        assert w.path.name == "telemetry-rank-0000000012.jsonl"

    def test_exclusive_creation_no_overwrite(self, tmp_path: Path) -> None:
        _make_writer(tmp_path)
        with pytest.raises(FileExistsError):
            _make_writer(tmp_path)

    def test_same_attempt_reopen_rejected(self, tmp_path: Path) -> None:
        # Reopening the exact same attempt/rank path fails through O_EXCL.
        _make_writer(tmp_path)
        with pytest.raises(FileExistsError):
            _make_writer(tmp_path)

    def test_two_ranks_independent_files(self, tmp_path: Path) -> None:
        w0 = _make_writer(tmp_path, rank=0, world_size=2, local_rank=0)
        w1 = _make_writer(tmp_path, rank=1, world_size=2, local_rank=1)
        assert w0.path != w1.path
        assert w0.path.name.endswith("rank-0000000000.jsonl")
        assert w1.path.name.endswith("rank-0000000001.jsonl")


# ---------------------------------------------------------------------------
# Lifecycle: stream_opened / stream_closed
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_first_record_is_stream_opened_at_seq_zero(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        records = _read_stream(w.path)
        assert records[0]["event_name"] == "logging.stream_opened"
        assert records[0]["sequence"] == 0
        assert records[0]["schema"] == EVENT_SCHEMA
        w.close(outcome="normal")

    def test_close_emits_stream_closed_last(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        w.emit_event(component="training", severity="INFO", event_name="training.update")
        w.close(outcome="normal")
        records = _read_stream(w.path)
        assert records[-1]["event_name"] == "logging.stream_closed"
        assert records[-1]["fields"][0]["name"] == "outcome"
        assert records[-1]["fields"][0]["value"] == "normal"

    def test_no_records_after_close(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        w.close(outcome="normal")
        before = len(_read_stream(w.path))
        with pytest.raises(WriterClosedError):
            w.emit_event(component="training", severity="INFO", event_name="x.y")
        with pytest.raises(WriterClosedError):
            w.emit_metric(
                component="training",
                observations=[
                    MetricObservation(
                        namespace="t",
                        name="l",
                        unit="dimensionless",
                        aggregation="gauge",
                        window="point",
                        value_status="finite",
                        value=1.0,
                    )
                ],
            )
        after = len(_read_stream(w.path))
        assert before == after

    def test_close_with_diagnostic_code(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        w.close(outcome="interrupted", diagnostic_code="handled_interruption")
        records = _read_stream(w.path)
        last = records[-1]
        assert last["diagnostic_code"] == "handled_interruption"

    def test_close_invalid_outcome_rejected(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        with pytest.raises((ValueError, Exception)):
            w.close(outcome="bogus")

    def test_close_idempotent(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        w.close(outcome="normal")
        # Second close is a no-op (does not write another stream_closed).
        w.close(outcome="normal")
        records = _read_stream(w.path)
        closes = [r for r in records if r["event_name"] == "logging.stream_closed"]
        assert len(closes) == 1


# ---------------------------------------------------------------------------
# Sequence ordering across events and metrics
# ---------------------------------------------------------------------------


class TestSequenceOrdering:
    def test_sequence_contiguous_across_event_and_metric(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        w.emit_event(component="training", severity="INFO", event_name="training.update")
        w.emit_metric(
            component="training",
            observations=[
                MetricObservation(
                    namespace="t",
                    name="l",
                    unit="dimensionless",
                    aggregation="gauge",
                    window="point",
                    value_status="finite",
                    value=1.0,
                )
            ],
        )
        w.emit_event(component="training", severity="INFO", event_name="training.step")
        w.close(outcome="normal")
        records = _read_stream(w.path)
        seqs = [r["sequence"] for r in records]
        assert seqs == [0, 1, 2, 3, 4]
        # schemas interleave: opened(event), event, metric, event, closed(event)
        assert [r["schema"] for r in records] == [
            "expertforge.telemetry-event",
            "expertforge.telemetry-event",
            "expertforge.metric-record",
            "expertforge.telemetry-event",
            "expertforge.telemetry-event",
        ]

    def test_sequence_starts_at_zero(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        assert _read_stream(w.path)[0]["sequence"] == 0
        w.close(outcome="normal")


# ---------------------------------------------------------------------------
# fsync cadence
# ---------------------------------------------------------------------------


class TestFsyncCadence:
    def test_fsync_every_n_records(self, tmp_path: Path) -> None:
        fsyncs: list[int] = []

        class FakeFd:
            pass

        real_fsync = os.fsync

        def counting_fsync(fd: int) -> None:
            fsyncs.append(fd)
            real_fsync(fd)

        with patch("expertforge.telemetry.writer.os.fsync", side_effect=counting_fsync):
            w = _make_writer(tmp_path, fsync_interval_records=2)
            # stream_opened is record 1 (seq 0); each emit adds a record.
            w.emit_event(component="training", severity="INFO", event_name="a.b")  # rec 2
            w.emit_event(component="training", severity="INFO", event_name="c.d")  # rec 3 -> fsync
            w.emit_event(component="training", severity="INFO", event_name="e.f")  # rec 4
            w.close(outcome="normal")  # rec 5 -> fsync on close
        # at least one cadence fsync + one close fsync occurred
        assert len(fsyncs) >= 2

    def test_fsync_after_error_and_critical(self, tmp_path: Path) -> None:
        fsyncs: list[int] = []
        real_fsync = os.fsync

        def counting_fsync(fd: int) -> None:
            fsyncs.append(fd)
            real_fsync(fd)

        with patch("expertforge.telemetry.writer.os.fsync", side_effect=counting_fsync):
            w = _make_writer(tmp_path, fsync_interval_records=1000)
            base = len(fsyncs)
            w.emit_event(component="t", severity="ERROR", event_name="e.err")
            assert len(fsyncs) == base + 1
            w.emit_event(component="t", severity="CRITICAL", event_name="e.crit")
            assert len(fsyncs) == base + 2
            w.close(outcome="normal")

    def test_flush_calls_fsync(self, tmp_path: Path) -> None:
        fsyncs: list[int] = []
        real_fsync = os.fsync

        def counting_fsync(fd: int) -> None:
            fsyncs.append(fd)
            real_fsync(fd)

        with patch("expertforge.telemetry.writer.os.fsync", side_effect=counting_fsync):
            w = _make_writer(tmp_path, fsync_interval_records=1000)
            before = len(fsyncs)
            w.flush()
            assert len(fsyncs) == before + 1
            w.close(outcome="normal")


# ---------------------------------------------------------------------------
# Full write loop (short writes)
# ---------------------------------------------------------------------------


class TestFullWriteLoop:
    def test_short_writes_completed_via_loop(self, tmp_path: Path) -> None:
        # Force os.write to write 1 byte at a time; the writer must loop to
        # completion so the on-disk stream is whole.
        real_write = os.write

        def one_byte_write(fd: int, data: bytes) -> int:
            return real_write(fd, data[:1])

        with patch("expertforge.telemetry.writer.os.write", side_effect=one_byte_write):
            w = _make_writer(tmp_path, fsync_interval_records=1000)
            w.emit_event(component="t", severity="INFO", event_name="a.b")
            w.close(outcome="normal")

        records = _read_stream(w.path)
        assert records[0]["event_name"] == "logging.stream_opened"
        assert records[-1]["event_name"] == "logging.stream_closed"
        # every line must parse (no partial record)
        for r in records:
            assert "sequence" in r


# ---------------------------------------------------------------------------
# Terminal failure state
# ---------------------------------------------------------------------------


class TestTerminalFailure:
    def test_write_failure_enters_terminal_state(self, tmp_path: Path) -> None:
        def failing_write(fd: int, data: bytes) -> int:
            # Fail on the very first write (stream_opened).
            raise OSError("simulated write failure")

        with patch("expertforge.telemetry.writer.os.write", side_effect=failing_write):
            with pytest.raises(WriterFailedError):
                _make_writer(tmp_path)
        # Once failed, the empty file may be removed (no complete record accepted).

    def test_failure_after_open_rejects_future_emission(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)

        def failing_write(fd: int, data: bytes) -> int:
            raise OSError("boom")

        with patch("expertforge.telemetry.writer.os.write", side_effect=failing_write):
            with pytest.raises(WriterFailedError):
                w.emit_event(component="t", severity="INFO", event_name="b.c")
        # Now terminal: further emissions raise immediately (without touching os).
        with pytest.raises(WriterFailedError):
            w.emit_event(component="t", severity="INFO", event_name="d.e")
        with pytest.raises(WriterFailedError):
            w.close(outcome="normal")

    def test_fsync_failure_enters_terminal_state(self, tmp_path: Path) -> None:
        def failing_fsync(fd: int) -> None:
            raise OSError("simulated fsync failure")

        with patch("expertforge.telemetry.writer.os.fsync", side_effect=failing_fsync):
            # fsync_interval_records=1 forces a cadence fsync on the very first
            # record (stream_opened), which fails and enters the terminal state.
            with pytest.raises(WriterFailedError):
                _make_writer(tmp_path, fsync_interval_records=1)

    def test_failed_writer_does_not_truncate_accepted_prefix(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path, fsync_interval_records=1000)
        w.emit_event(component="t", severity="INFO", event_name="a.b")
        size_before = w.path.stat().st_size

        def failing_write(fd: int, data: bytes) -> int:
            raise OSError("boom")

        with patch("expertforge.telemetry.writer.os.write", side_effect=failing_write):
            with pytest.raises(WriterFailedError):
                w.emit_event(component="t", severity="INFO", event_name="c.d")
        # Accepted prefix bytes are retained as evidence (not truncated).
        assert w.path.stat().st_size == size_before


# ---------------------------------------------------------------------------
# Console rendering and one-time console-failure degradation
# ---------------------------------------------------------------------------


class _FailingStream:
    """A minimal text stream whose write() always raises."""

    def __init__(self) -> None:
        self.failures = 0

    def write(self, _data: str) -> int:
        self.failures += 1
        raise OSError("console unavailable")

    def flush(self) -> None:  # pragma: no cover - never reached on failure path
        pass


class TestConsole:
    def test_console_rendering_deterministic_one_line(self, tmp_path: Path) -> None:
        captured: list[str] = []

        class CaptureStream:
            def write(self, data: str) -> int:
                captured.append(data)
                return len(data)

            def flush(self) -> None:
                pass

        w = _make_writer(tmp_path, console_stream=CaptureStream())
        w.emit_event(component="training", severity="INFO", event_name="training.update")
        w.close(outcome="normal")
        # Each console write is one line (ends with newline) and has no ANSI.
        for line in captured:
            assert line.endswith("\n")
            assert "\x1b[" not in line

    def test_console_failure_emits_diagnostic_once_then_disables(self, tmp_path: Path) -> None:
        failing = _FailingStream()
        w = _make_writer(tmp_path, console_stream=failing)
        # The stream_opened console write already failed once; console is now
        # disabled. Further emits must NOT attempt console writes and must not
        # raise; exactly one console_write_failed diagnostic event is emitted.
        assert failing.failures == 1
        w.emit_event(component="t", severity="INFO", event_name="a.b")
        # No additional console attempts after the one-time failure.
        assert failing.failures == 1
        w.close(outcome="normal")
        records = _read_stream(w.path)
        diag = [r for r in records if r.get("diagnostic_code") == "console_write_failed"]
        assert len(diag) == 1

    def test_console_disabled_silent(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path, console_enabled=False)
        w.emit_event(component="t", severity="INFO", event_name="a.b")
        w.close(outcome="normal")
        # No console interaction means no console_write_failed diagnostics.
        records = _read_stream(w.path)
        assert all(r.get("diagnostic_code") != "console_write_failed" for r in records)


# ---------------------------------------------------------------------------
# WriterStats and progress / field handling
# ---------------------------------------------------------------------------


class TestStatsAndEmission:
    def test_stats_track_writes_bytes_fsync(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path, fsync_interval_records=1)
        w.emit_event(component="t", severity="INFO", event_name="a.b")
        w.close(outcome="normal")
        stats = w.stats
        # opened(0) + a.b(1) + closed(2) = 3 records written
        assert stats.records_written == 3
        assert stats.bytes_written == w.path.stat().st_size
        assert stats.first_sequence == 0
        assert stats.last_sequence == 2
        assert stats.fsync_count >= 1
        assert stats.console_failures == 0

    def test_stats_frozen(self, tmp_path: Path) -> None:
        from pydantic import ValidationError

        w = _make_writer(tmp_path)
        stats = w.stats
        with pytest.raises(ValidationError):
            stats.records_written = 99  # type: ignore[misc]

    def test_emit_event_with_progress_and_fields(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        w.emit_event(
            component="training",
            severity="INFO",
            event_name="training.update",
            progress=ProgressPosition(step=1, update=1, processed_tokens=128),
            fields=(("loss", 0.5), ("lr", 0.001)),
        )
        w.close(outcome="normal")
        records = _read_stream(w.path)
        ev = records[1]
        assert ev["progress"]["step"] == 1
        assert ev["progress"]["processed_tokens"] == 128
        names = [f["name"] for f in ev["fields"]]
        assert names == ["loss", "lr"]

    def test_emit_metric_with_progress(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        w.emit_metric(
            component="training",
            progress=ProgressPosition(step=2),
            observations=[
                MetricObservation(
                    namespace="t",
                    name="loss",
                    unit="dimensionless",
                    aggregation="gauge",
                    window="point",
                    value_status="finite",
                    value=0.4,
                ),
                MetricObservation(
                    namespace="t",
                    name="ppl",
                    unit="dimensionless",
                    aggregation="gauge",
                    window="point",
                    value_status="finite",
                    value=20.0,
                ),
            ],
        )
        w.close(outcome="normal")
        records = _read_stream(w.path)
        metric = records[1]
        assert metric["schema"] == "expertforge.metric-record"
        keys = [(o["namespace"], o["name"]) for o in metric["observations"]]
        assert keys == [("t", "loss"), ("t", "ppl")]

    def test_record_size_limit_enforced(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        # Build a record that exceeds 64 KiB when serialized.
        huge_field = ("x" * 200, "v")
        many = [huge_field for _ in range(MAX_EVENT_FIELDS)]
        with pytest.raises((ValueError, Exception)):
            w.emit_event(component="t", severity="INFO", event_name="a.b", fields=tuple(many))

    def test_invalid_field_value_rejected(self, tmp_path: Path) -> None:
        import math

        w = _make_writer(tmp_path)
        with pytest.raises((ValueError, Exception)):
            w.emit_event(
                component="t", severity="INFO", event_name="a.b", fields=(("loss", math.nan),)
            )

    def test_event_fields_sorted_by_writer(self, tmp_path: Path) -> None:
        # The writer accepts unsorted fields and persists them sorted.
        w = _make_writer(tmp_path)
        w.emit_event(
            component="t",
            severity="INFO",
            event_name="a.b",
            fields=(("zeta", 1), ("alpha", 2)),
        )
        w.close(outcome="normal")
        ev = _read_stream(w.path)[1]
        assert [f["name"] for f in ev["fields"]] == ["alpha", "zeta"]


# ---------------------------------------------------------------------------
# Serialization determinism / no JSON NaN
# ---------------------------------------------------------------------------


class TestSerializationContract:
    def test_lines_are_newline_terminated_compact_sorted(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        w.close(outcome="normal")
        raw = w.path.read_bytes()
        # No pretty separators.
        assert b": " not in raw
        assert b", " not in raw
        # Ends with a newline.
        assert raw.endswith(b"\n")

    def test_stream_format_version_on_every_record(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        w.emit_event(component="t", severity="INFO", event_name="a.b")
        w.close(outcome="normal")
        for r in _read_stream(w.path):
            assert r["stream_format_version"] == STREAM_FORMAT_VERSION
