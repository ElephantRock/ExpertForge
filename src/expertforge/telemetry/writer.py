"""Exclusive, synchronous JSONL telemetry writer for Issue #9."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

from expertforge.identity.record import AttemptIdentityRecord
from expertforge.telemetry.console import render_record
from expertforge.telemetry.models import (
    EVENT_SCHEMA_VERSION,
    MAX_EVENT_FIELDS,
    MAX_RECORD_BYTES,
    METRIC_SCHEMA_VERSION,
    STREAM_FORMAT_VERSION,
    DiagnosticCode,
    EventField,
    EventRecord,
    MetricObservation,
    MetricRecord,
    ProcessContext,
    ProgressPosition,
    Severity,
    WriterStats,
    metric_semantic_key,
    sanitize_persisted_string,
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

_LEVELS: dict[str, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
_LOGGING_COMPONENT = "logging"


class ConsoleStream(Protocol):
    def write(self, data: str) -> int: ...

    def flush(self) -> None: ...


class WriterClosedError(Exception):
    """Raised when emission is attempted after a successful close."""


class WriterFailedError(Exception):
    """Raised for terminal telemetry-writer failures."""


class _DeferredWriterStream(ConsoleStream):
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

        return cast(ConsoleStream, sys.stderr)

    return _DeferredWriterStream(_stderr)


class TelemetryWriter:
    """Write one identity-bound, process-local canonical telemetry stream."""

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
        level: str = "INFO",
    ) -> None:
        if fsync_interval_records < 1:
            raise ValueError("fsync_interval_records must be >= 1.")
        if level not in _LEVELS:
            raise ValueError(f"level must be one of {tuple(_LEVELS)}; got {level!r}.")

        self._artifact_root = Path(artifact_root)
        self._identity = identity
        self._ctx = process_context
        self._console_enabled = console_enabled
        self._fsync_interval = fsync_interval_records
        self._level = level
        self._wall_clock = wall_clock or _default_wall_clock
        self._monotonic_clock = monotonic_clock or _default_monotonic_clock
        self._console_stream = console_stream or _default_console_stream()

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
        self._last_elapsed_ns = 0
        self._last_progress: dict[str, int | None] = {
            "step": None,
            "update": None,
            "processed_tokens": None,
        }
        self._fd: int | None = None

        self._monotonic_baseline = self._monotonic_clock()
        self._path = self._compute_path()
        self._open_and_emit_opened()

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
        self._require_open()
        if severity not in _LEVELS:
            raise ValueError(f"unknown event severity {severity!r}.")
        if _LEVELS[severity] < _LEVELS[self._level]:
            return

        event_fields, fields_redacted = _coerce_event_fields(fields)
        sanitized_message: str | None = None
        message_redacted = False
        if operator_message is not None:
            sanitized_message, message_redacted = sanitize_persisted_string(
                operator_message, truncate=True
            )
        if fields_redacted or message_redacted:
            if diagnostic_code not in {None, "redacted_sensitive_value"}:
                raise ValueError(
                    "redaction cannot replace an unrelated diagnostic_code; emit separate events."
                )
            diagnostic_code = "redacted_sensitive_value"

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
            elapsed_ns=self._next_elapsed_ns(),
            progress=progress,
            severity=cast(Severity, severity),
            event_name=event_name,
            diagnostic_code=diagnostic_code,
            fields=event_fields,
            operator_message=sanitized_message,
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
        self._require_open()
        sorted_observations = tuple(sorted(observations, key=metric_semantic_key))
        record = MetricRecord(
            schema_name="expertforge.metric-record",
            schema_version=METRIC_SCHEMA_VERSION,
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
            elapsed_ns=self._next_elapsed_ns(),
            progress=progress,
            observations=sorted_observations,
        )
        self._write_record(record)
        self._render_console(record)

    def flush(self) -> None:
        self._require_open()
        self._fsync_or_fail()

    def close(self, outcome: str, diagnostic_code: DiagnosticCode | None = None) -> None:
        if self._closed:
            return
        if self._failed:
            self._closed = True
            raise WriterFailedError("telemetry writer is in a terminal failed state; cannot close.")
        if outcome not in {"normal", "interrupted", "failed"}:
            raise ValueError(f"close outcome must be normal|interrupted|failed; got {outcome!r}.")
        if outcome == "normal":
            if diagnostic_code is not None:
                raise ValueError("normal close forbids diagnostic_code.")
        elif outcome == "interrupted":
            if diagnostic_code != "handled_interruption":
                raise ValueError(
                    "interrupted close requires diagnostic_code='handled_interruption'."
                )
        elif diagnostic_code is None:
            diagnostic_code = "unhandled_exception"

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
                elapsed_ns=self._next_elapsed_ns(),
                progress=None,
                severity="INFO",
                event_name="logging.stream_closed",
                diagnostic_code=diagnostic_code,
                fields=(EventField(name="outcome", value=outcome),),
                operator_message=None,
            )
            self._write_record(record)
            self._fsync_or_fail()
            self._render_console(record, allow_diagnostic=False)
            self._close_fd_or_fail()
            self._closed = True
        except WriterFailedError:
            self._closed = True
            raise
        except Exception as exc:
            self._failed = True
            self._closed = True
            self._close_fd_silently()
            raise WriterFailedError("telemetry close failed before completion.") from exc

    def __enter__(self) -> TelemetryWriter:
        return self

    def __exit__(self, _exc_type: Any, exc: Any, _tb: Any) -> None:
        if self._closed:
            return
        if exc is None:
            self.close(outcome="normal")
            return
        try:
            self.close(outcome="failed", diagnostic_code="unhandled_exception")
        except WriterFailedError:
            pass

    def _compute_path(self) -> Path:
        return (
            self._artifact_root
            / self._identity.run_id
            / "attempts"
            / self._identity.attempt_id
            / "logs"
            / f"telemetry-rank-{self._ctx.rank:010d}.jsonl"
        )

    def _next_elapsed_ns(self) -> int:
        now = self._monotonic_clock()
        elapsed = now - self._monotonic_baseline
        if elapsed < 0 or elapsed < self._last_elapsed_ns:
            self._terminal_failure(
                f"monotonic clock regressed (last={self._last_elapsed_ns}, current={elapsed})."
            )
        return elapsed

    def _open_and_emit_opened(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
        try:
            self._fd = os.open(self._path, flags)
        except OSError:
            self._failed = True
            raise

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
                sequence=0,
                timestamp_utc=self._wall_clock(),
                elapsed_ns=0,
                progress=None,
                severity="INFO",
                event_name="logging.stream_opened",
                diagnostic_code=None,
                fields=(),
                operator_message=None,
            )
            self._write_record(record)
            self._render_console(record)
        except BaseException as exc:
            if self._records_written == 0:
                self._safe_unlink()
            else:
                self._failed = True
                self._close_fd_silently()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, WriterFailedError):
                raise
            raise WriterFailedError("telemetry writer initialization failed.") from exc

    def _write_record(self, record: EventRecord | MetricRecord) -> None:
        if self._fd is None:
            raise WriterFailedError("writer file descriptor is not open.")
        self._validate_progress(record.progress)
        payload = record.to_deterministic_json() + b"\n"
        if len(payload) > MAX_RECORD_BYTES:
            raise ValueError(
                f"serialized record ({len(payload)} bytes) exceeds max {MAX_RECORD_BYTES}."
            )
        self._write_all(payload)

        self._bytes_written += len(payload)
        if self._first_sequence is None:
            self._first_sequence = record.sequence
        self._last_sequence = record.sequence
        self._records_written += 1
        self._sequence += 1
        self._last_elapsed_ns = record.elapsed_ns
        self._commit_progress(record.progress)

        if self._records_written % self._fsync_interval == 0:
            self._fsync_or_fail()

    def _validate_progress(self, progress: ProgressPosition | None) -> None:
        if progress is None:
            return
        for name in ("step", "update", "processed_tokens"):
            current = getattr(progress, name)
            previous = self._last_progress[name]
            if current is not None and previous is not None and current < previous:
                raise ValueError(f"progress.{name} {current} regresses below previous {previous}.")

    def _commit_progress(self, progress: ProgressPosition | None) -> None:
        if progress is None:
            return
        for name in ("step", "update", "processed_tokens"):
            current = getattr(progress, name)
            if current is not None:
                self._last_progress[name] = current

    def _write_all(self, payload: bytes) -> None:
        if self._fd is None:
            raise WriterFailedError("writer file descriptor is not open.")
        view = memoryview(payload)
        total = 0
        while total < len(view):
            try:
                written = os.write(self._fd, view[total:])
            except OSError as exc:
                self._terminal_failure(f"telemetry write failed: {exc}", exc)
            if written <= 0:
                self._terminal_failure(
                    "telemetry write failed: os.write returned a non-positive count."
                )
            total += written

    def _fsync_or_fail(self) -> None:
        if self._fd is None:
            raise WriterFailedError("writer file descriptor is not open.")
        try:
            os.fsync(self._fd)
        except OSError as exc:
            self._terminal_failure(f"telemetry fsync failed: {exc}", exc)
        self._fsync_count += 1

    def _close_fd_or_fail(self) -> None:
        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        try:
            os.close(fd)
        except OSError as exc:
            self._failed = True
            raise WriterFailedError(f"telemetry close failed: {exc}") from exc

    def _render_console(
        self,
        record: EventRecord | MetricRecord,
        *,
        allow_diagnostic: bool = True,
    ) -> None:
        if not self._console_enabled or self._console_failed_reported:
            return
        line = render_record(record) + "\n"
        try:
            written = self._console_stream.write(line)
            if not isinstance(written, int) or isinstance(written, bool) or written != len(line):
                raise OSError(
                    f"console short write: expected {len(line)} characters, got {written!r}."
                )
            self._console_stream.flush()
        except Exception:
            self._console_failures += 1
            self._console_failed_reported = True
            self._console_enabled = False
            if allow_diagnostic and not self._failed and not self._closed:
                self._emit_console_write_failed()

    def _emit_console_write_failed(self) -> None:
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
            elapsed_ns=self._next_elapsed_ns(),
            progress=None,
            severity="WARNING",
            event_name="logging.console_write_failed",
            diagnostic_code="console_write_failed",
            fields=(),
            operator_message=None,
        )
        self._write_record(record)

    def _require_open(self) -> None:
        if self._failed:
            raise WriterFailedError("telemetry writer is in a terminal failed state.")
        if self._closed:
            raise WriterClosedError("telemetry writer is closed.")

    def _terminal_failure(self, message: str, cause: BaseException | None = None) -> NoReturn:
        self._failed = True
        self._close_fd_silently()
        error = WriterFailedError(message)
        if cause is None:
            raise error
        raise error from cause

    def _close_fd_silently(self) -> None:
        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        try:
            os.close(fd)
        except OSError:
            pass

    def _safe_unlink(self) -> None:
        self._close_fd_silently()
        try:
            self._path.unlink()
        except OSError:
            pass


def _coerce_event_fields(
    fields: tuple[tuple[str, Any], ...],
) -> tuple[tuple[EventField, ...], bool]:
    if len(fields) > MAX_EVENT_FIELDS:
        raise ValueError(f"event fields exceed max {MAX_EVENT_FIELDS}.")
    redaction_occurred = False
    coerced: list[EventField] = []
    for name, raw_value in fields:
        value = raw_value
        if isinstance(raw_value, str):
            value, changed = sanitize_persisted_string(raw_value)
            redaction_occurred = redaction_occurred or changed
        coerced.append(EventField(name=name, value=value))
    coerced.sort(key=lambda field: field.name)
    return tuple(coerced), redaction_occurred


def _default_wall_clock() -> datetime:
    from datetime import UTC

    return datetime.now(UTC)


def _default_monotonic_clock() -> int:
    import time

    return time.monotonic_ns()
