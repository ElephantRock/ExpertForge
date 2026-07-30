"""TelemetryWriter — exclusive-create JSONL stream writer (Issue #9).

One canonical per-process stream::

    <artifact-root>/<run-id>/attempts/<attempt-id>/logs/telemetry-rank-<10-digit-rank>.jsonl

The writer opens the stream with exclusive creation (``O_CREAT | O_EXCL``); a
second writer for the same attempt/rank path fails atomically rather than
silently clobbering prior telemetry. Each record is rendered and validated as
complete compact JSON plus newline before any byte is written; a full
``os.write`` loop handles legal short writes.

Lifecycle: the first record is ``logging.stream_opened`` at sequence 0; the
final record is ``logging.stream_closed`` with a closed outcome. Sequence starts
at 0 and increases contiguously across both event and metric records. After any
write/fsync/close failure the writer enters a terminal failed state and rejects
future emission; it never deletes or truncates already-accepted records (an
empty newly-created file with no accepted record may be removed on the initial
open failure).

Bounded overhead (design §13): synchronous, no background thread, no unbounded
queue; record size ≤ 64 KiB; event fields ≤ 64; metric observations ≤ 128;
persisted strings ≤ 4096.

This module has NO import-time side effects.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from expertforge.identity.record import AttemptIdentityRecord
from expertforge.telemetry.console import render_record
from expertforge.telemetry.models import (
    EVENT_SCHEMA_VERSION,
    MAX_EVENT_FIELDS,
    MAX_RECORD_BYTES,
    STREAM_FORMAT_VERSION,
    DiagnosticCode,
    EventField,
    EventRecord,
    MetricObservation,
    MetricRecord,
    ProcessContext,
    ProgressPosition,
    WriterStats,
    sanitize_operator_message,
)

__all__ = [
    "ConsoleStream",
    "MonotonicClock",
    "TelemetryWriter",
    "WallClock",
    "WriterClosedError",
    "WriterFailedError",
]

WallClock = Callable[[], datetime]
MonotonicClock = Callable[[], int]

_LOGGING_COMPONENT = "logging"
_EVENT_STREAM_OPENED = "logging.stream_opened"
_EVENT_STREAM_CLOSED = "logging.stream_closed"


class ConsoleStream(Protocol):
    """Minimal text-stream protocol for console rendering."""

    def write(self, data: str) -> int: ...

    def flush(self) -> None: ...


class WriterClosedError(Exception):
    """Raised when emitting after :meth:`TelemetryWriter.close`."""


class WriterFailedError(Exception):
    """Raised on write/fsync/close failure, and for any emission after failure."""


class _DeferredWriterStream(ConsoleStream):
    """A console stream wrapper that defers stdout selection until first use.

    Selecting ``sys.stderr`` at construction time (rather than import time) keeps
    the package free of import-time output/handler mutation. The default is only
    resolved if/when a console write is attempted.
    """

    def __init__(self, factory: Callable[[], ConsoleStream]) -> None:
        self._factory = factory
        self._target: ConsoleStream | None = None

    def _resolve(self) -> ConsoleStream:
        if self._target is None:
            self._target = self._factory()
        return self._target

    def write(self, data: str) -> int:
        return self._resolve().write(data)

    def flush(self) -> None:
        self._resolve().flush()


def _default_console_stream() -> ConsoleStream:
    def _stderr() -> ConsoleStream:
        import sys

        return sys.stderr

    return _DeferredWriterStream(_stderr)


class TelemetryWriter:
    """Exclusive-create JSONL telemetry stream writer.

    Args:
        artifact_root: root under which the per-attempt logs directory lives.
        identity: the Issue #6 :class:`AttemptIdentityRecord`; ``run_id``,
            ``attempt_id``, and the full ``spec-v1-sha256-...`` fingerprint are
            copied from it. Independent identity strings are NOT accepted.
        process_context: explicit process topology coordinates.
        console_enabled: whether to render each record to the console stream.
        fsync_interval_records: fsync every N successfully written records (≥1).
        wall_clock: injectable timezone-aware UTC clock for ``timestamp_utc``.
        monotonic_clock: injectable nanosecond clock for ``elapsed_ns``.
        console_stream: injectable console sink (default: stderr, deferred).
    """

    def __init__(
        self,
        artifact_root: Path,
        identity: AttemptIdentityRecord,
        process_context: ProcessContext,
        *,
        console_enabled: bool = True,
        fsync_interval_records: int = 100,
        wall_clock: WallClock | None = None,
        monotonic_clock: MonotonicClock | None = None,
        console_stream: ConsoleStream | None = None,
    ) -> None:
        if fsync_interval_records < 1:
            raise ValueError("fsync_interval_records must be >= 1.")
        self._artifact_root = Path(artifact_root)
        self._identity = identity
        self._ctx = process_context
        self._console_enabled = console_enabled
        self._fsync_interval = fsync_interval_records
        self._wall_clock: WallClock = wall_clock or _default_wall_clock
        self._monotonic_clock: MonotonicClock = monotonic_clock or _default_monotonic_clock
        self._console_stream: ConsoleStream = console_stream or _default_console_stream()

        # Writer state.
        self._closed = False
        self._failed = False
        self._sequence = 0
        self._records_written = 0
        self._bytes_written = 0
        self._fsync_count = 0
        self._console_failures = 0
        self._console_failed_reported = False
        self._first_sequence: int | None = None
        self._last_sequence: int | None = None
        # Monotonic baseline (ns) captured at construction for elapsed_ns.
        self._monotonic_baseline = self._monotonic_clock()
        self._fd: int | None = None

        self._path = self._compute_path()
        # Open exclusively and write the first record (logging.stream_opened).
        self._open_and_emit_opened()

    # --- public API --------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def stats(self) -> WriterStats:
        return WriterStats(
            records_written=self._records_written,
            bytes_written=self._bytes_written,
            fsync_count=self._fsync_count,
            console_failures=self._console_failures,
            first_sequence=self._first_sequence,
            last_sequence=self._last_sequence,
        )

    def emit_event(
        self,
        *,
        component: str,
        severity: str,
        event_name: str,
        progress: ProgressPosition | None = None,
        fields: tuple[tuple[str, Any], ...] = (),
        diagnostic_code: DiagnosticCode | None = None,
        operator_message: str | None = None,
    ) -> None:
        """Emit one structured event record to the stream (and console, if enabled)."""
        self._require_open()
        sanitized_fields = _coerce_event_fields(fields)
        sanitized_msg = sanitize_operator_message(operator_message) if operator_message else None
        record = EventRecord(
            schema_name="expertforge.telemetry-event",
            schema_version=EVENT_SCHEMA_VERSION,
            stream_format_version=STREAM_FORMAT_VERSION,
            run_id=self._identity.run_id,
            attempt_id=self._identity.attempt_id,
            specification_fingerprint=self._identity.fingerprint_digest_str(),
            component=component,
            rank=self._ctx.rank,
            local_rank=self._ctx.local_rank,
            world_size=self._ctx.world_size,
            sequence=self._sequence,
            timestamp_utc=self._wall_clock(),
            elapsed_ns=self._elapsed_ns(),
            progress=progress,
            severity=severity,  # type: ignore[arg-type]
            event_name=event_name,
            diagnostic_code=diagnostic_code,
            fields=sanitized_fields,
            operator_message=sanitized_msg,
        )
        self._write_record(record)
        if severity in {"ERROR", "CRITICAL"}:
            self._fsync_or_fail()
        self._render_console(record)

    def emit_metric(
        self,
        *,
        component: str,
        observations: list[MetricObservation] | tuple[MetricObservation, ...],
        progress: ProgressPosition | None = None,
    ) -> None:
        """Emit one metric record (one or more observations) to the stream."""
        self._require_open()
        sorted_obs = tuple(sorted(observations, key=lambda o: (o.namespace, o.name)))
        record = MetricRecord(
            schema_name="expertforge.metric-record",
            schema_version=1,
            stream_format_version=STREAM_FORMAT_VERSION,
            run_id=self._identity.run_id,
            attempt_id=self._identity.attempt_id,
            specification_fingerprint=self._identity.fingerprint_digest_str(),
            component=component,
            rank=self._ctx.rank,
            local_rank=self._ctx.local_rank,
            world_size=self._ctx.world_size,
            sequence=self._sequence,
            timestamp_utc=self._wall_clock(),
            elapsed_ns=self._elapsed_ns(),
            progress=progress,
            observations=sorted_obs,
        )
        self._write_record(record)
        self._render_console(record)

    def flush(self) -> None:
        """fsync the underlying file immediately."""
        self._require_open()
        self._fsync_or_fail()

    def close(self, outcome: str, diagnostic_code: DiagnosticCode | None = None) -> None:
        """Emit the terminal ``logging.stream_closed`` event and close the file.

        Idempotent: a second call is a no-op. ``outcome`` must be one of
        ``normal | interrupted | failed``.
        """
        if self._closed:
            return
        if self._failed:
            # A failed writer cannot emit a clean close. Mark closed and surface
            # the terminal state so callers cannot mistake the stream for intact.
            self._closed = True
            raise WriterFailedError("telemetry writer is in a terminal failed state; cannot close.")
        if outcome not in ("normal", "interrupted", "failed"):
            raise ValueError(f"close outcome must be normal|interrupted|failed; got {outcome!r}.")
        try:
            record = EventRecord(
                schema_name="expertforge.telemetry-event",
                schema_version=EVENT_SCHEMA_VERSION,
                stream_format_version=STREAM_FORMAT_VERSION,
                run_id=self._identity.run_id,
                attempt_id=self._identity.attempt_id,
                specification_fingerprint=self._identity.fingerprint_digest_str(),
                component=_LOGGING_COMPONENT,
                rank=self._ctx.rank,
                local_rank=self._ctx.local_rank,
                world_size=self._ctx.world_size,
                sequence=self._sequence,
                timestamp_utc=self._wall_clock(),
                elapsed_ns=self._elapsed_ns(),
                progress=None,
                severity="INFO",
                event_name=_EVENT_STREAM_CLOSED,
                diagnostic_code=diagnostic_code,
                fields=(EventField(name="outcome", value=outcome),),
                operator_message=None,
            )
            self._write_record(record)
            self._fsync_or_fail()
            self._render_console(record)
            self._close_fd_or_fail()
            self._closed = True
        except WriterFailedError:
            self._failed = True
            self._closed = True
            raise

    # --- context manager support ------------------------------------------

    def __enter__(self) -> TelemetryWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, _tb: Any) -> None:
        if self._closed:
            return
        if exc is None:
            self.close(outcome="normal")
        else:
            # Generic unhandled-exception close without persisting raw exception text.
            try:
                self.close(outcome="interrupted", diagnostic_code="unhandled_exception")
            except WriterFailedError:
                # If the writer is already failing, a failed close is the outcome.
                self.close(outcome="failed", diagnostic_code="telemetry_write_failed")

    # --- internals ---------------------------------------------------------

    def _compute_path(self) -> Path:
        return (
            self._artifact_root
            / self._identity.run_id
            / "attempts"
            / self._identity.attempt_id
            / "logs"
            / f"telemetry-rank-{self._ctx.rank:010d}.jsonl"
        )

    def _elapsed_ns(self) -> int:
        now = self._monotonic_clock()
        delta = now - self._monotonic_baseline
        if delta < 0:
            # A regressing monotonic clock is a typed hard failure (design §4).
            raise WriterFailedError(
                f"monotonic clock regressed: baseline={self._monotonic_baseline} now={now}."
            )
        return delta

    def _open_and_emit_opened(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # On Windows, os.open/os.write default to text mode and would translate
        # b"\n" -> b"\r\n"; open in binary so the byte stream is exact.
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        binary_flag = getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(self._path, flags | binary_flag)
        except FileExistsError:
            # Re-raise as FileExistsError so callers see the no-overwrite contract.
            self._failed = True
            raise
        except OSError:
            self._failed = True
            raise
        self._fd = fd
        try:
            record = EventRecord(
                schema_name="expertforge.telemetry-event",
                schema_version=EVENT_SCHEMA_VERSION,
                stream_format_version=STREAM_FORMAT_VERSION,
                run_id=self._identity.run_id,
                attempt_id=self._identity.attempt_id,
                specification_fingerprint=self._identity.fingerprint_digest_str(),
                component=_LOGGING_COMPONENT,
                rank=self._ctx.rank,
                local_rank=self._ctx.local_rank,
                world_size=self._ctx.world_size,
                sequence=self._sequence,
                timestamp_utc=self._wall_clock(),
                elapsed_ns=0,
                progress=None,
                severity="INFO",
                event_name=_EVENT_STREAM_OPENED,
                diagnostic_code=None,
                fields=(),
                operator_message=None,
            )
            self._write_record(record)
            self._render_console(record)
        except WriterFailedError:
            # Initial open failed before any complete record was accepted: an
            # empty newly-created file may be removed so O_EXCL does not block
            # retries. (Once a record exists, the prefix is retained.)
            if self._records_written == 0:
                self._safe_unlink()
            raise

    def _write_record(self, record: EventRecord | MetricRecord) -> None:
        if self._fd is None:  # pragma: no cover - defensive
            raise WriterFailedError("writer file descriptor is not open.")
        payload = record.to_deterministic_json() + b"\n"
        if len(payload) > MAX_RECORD_BYTES:
            raise ValueError(
                f"serialized record ({len(payload)} bytes) exceeds max {MAX_RECORD_BYTES}."
            )
        self._write_all(payload)
        self._bytes_written += len(payload)
        if self._first_sequence is None:
            self._first_sequence = self._sequence
        self._last_sequence = self._sequence
        self._records_written += 1
        self._sequence += 1
        # fsync cadence: every N successfully written records.
        if self._records_written % self._fsync_interval == 0:
            self._fsync_or_fail()

    def _write_all(self, payload: bytes) -> None:
        assert self._fd is not None
        view = memoryview(payload)
        total = 0
        while total < len(view):
            try:
                written = os.write(self._fd, view[total:])
            except OSError as e:
                self._mark_failed()
                raise WriterFailedError(f"telemetry write failed: {e}") from e
            if written <= 0:  # pragma: no cover - defensive
                self._mark_failed()
                raise WriterFailedError("os.write returned non-positive byte count.")
            total += written

    def _fsync_or_fail(self) -> None:
        if self._fd is None:  # pragma: no cover - defensive
            return
        try:
            os.fsync(self._fd)
        except OSError as e:
            self._mark_failed()
            raise WriterFailedError(f"telemetry fsync failed: {e}") from e
        self._fsync_count += 1

    def _close_fd_or_fail(self) -> None:
        if self._fd is None:
            return
        try:
            os.close(self._fd)
        except OSError as e:
            self._mark_failed()
            raise WriterFailedError(f"telemetry close failed: {e}") from e
        finally:
            self._fd = None

    def _render_console(self, record: EventRecord | MetricRecord) -> None:
        if not self._console_enabled or self._console_failed_reported:
            return
        try:
            line = render_record(record) + "\n"
            self._console_stream.write(line)
            self._console_stream.flush()
        except OSError:
            self._console_failures += 1
            self._console_failed_reported = True
            self._console_enabled = False
            # Emit one console_write_failed diagnostic into the machine stream,
            # if the machine stream remains usable. Do not recursively retry
            # console diagnostics (rendering is skipped for this synthetic event
            # because the console is already disabled).
            if not self._failed and not self._closed:
                self._emit_console_write_failed()

    def _emit_console_write_failed(self) -> None:
        # Synthesize the diagnostic event directly without console rendering.
        record = EventRecord(
            schema_name="expertforge.telemetry-event",
            schema_version=EVENT_SCHEMA_VERSION,
            stream_format_version=STREAM_FORMAT_VERSION,
            run_id=self._identity.run_id,
            attempt_id=self._identity.attempt_id,
            specification_fingerprint=self._identity.fingerprint_digest_str(),
            component=_LOGGING_COMPONENT,
            rank=self._ctx.rank,
            local_rank=self._ctx.local_rank,
            world_size=self._ctx.world_size,
            sequence=self._sequence,
            timestamp_utc=self._wall_clock(),
            elapsed_ns=self._elapsed_ns(),
            progress=None,
            severity="WARNING",
            event_name="logging.console_write_failed",
            diagnostic_code="console_write_failed",
            fields=(),
            operator_message=None,
        )
        # Write to the machine stream only (console already disabled).
        self._write_record(record)

    def _require_open(self) -> None:
        if self._failed:
            raise WriterFailedError("telemetry writer is in a terminal failed state.")
        if self._closed:
            raise WriterClosedError("telemetry writer is closed.")

    def _mark_failed(self) -> None:
        self._failed = True

    def _safe_unlink(self) -> None:
        try:
            os.close(self._fd)  # type: ignore[arg-type]
        except OSError:
            pass
        self._fd = None
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass


def _coerce_event_fields(fields: tuple[tuple[str, Any], ...]) -> tuple[EventField, ...]:
    """Validate and sort caller-supplied (name, value) event-field pairs."""
    if len(fields) > MAX_EVENT_FIELDS:
        raise ValueError(f"event fields count {len(fields)} exceeds max {MAX_EVENT_FIELDS}.")
    coerced = [EventField(name=name, value=value) for name, value in fields]
    coerced.sort(key=lambda f: f.name)
    return tuple(coerced)


def _default_wall_clock() -> datetime:
    from datetime import UTC

    return datetime.now(UTC)


def _default_monotonic_clock() -> int:
    import time

    return time.monotonic_ns()
