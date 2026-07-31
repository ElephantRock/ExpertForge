"""Canonical authoritative loading and diagnostic scanning for telemetry JSONL."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypeAlias, cast

from pydantic import ValidationError

from expertforge.identity.record import AttemptIdentityRecord
from expertforge.telemetry.models import (
    EVENT_SCHEMA,
    MAX_RECORD_BYTES,
    METRIC_SCHEMA,
    EventRecord,
    MetricRecord,
    ProcessContext,
    ProgressPosition,
    StreamStatus,
)

__all__ = ["ScanResult", "TelemetryLoadError", "load_telemetry_stream", "scan_telemetry_stream"]

Record: TypeAlias = EventRecord | MetricRecord
Binding: TypeAlias = tuple[str, str, str, int, int | None, int]

_OPEN_EVENT = "logging.stream_opened"
_CLOSE_EVENT = "logging.stream_closed"
_CANONICAL_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_PROGRESS_FIELDS = ("step", "update", "processed_tokens")


class TelemetryLoadError(Exception):
    """Raised when a telemetry stream cannot be accepted authoritatively."""


class ScanResult:
    """Immutable-view result containing a validated accepted prefix."""

    __slots__ = ("_records", "_status")

    def __init__(self, status: StreamStatus, records: list[Record]) -> None:
        self._status = status
        self._records = tuple(records)

    @property
    def status(self) -> StreamStatus:
        return self._status

    @property
    def accepted_count(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[Record]:
        return list(self._records)

    def __repr__(self) -> str:
        return f"ScanResult(status={self.status!r}, accepted_count={self.accepted_count})"


class _ParseFailure(Exception):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _ParseFailure(f"duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise _ParseFailure(f"non-standard JSON numeric constant {value!r}.")


def _split_records(raw: bytes) -> tuple[list[bytes], bool]:
    if not raw:
        return [], False
    if raw.endswith(b"\n"):
        body = raw[:-1]
        return ([] if not body else body.split(b"\n")), False
    boundary = raw.rfind(b"\n")
    if boundary < 0:
        return [], True
    body = raw[:boundary]
    return ([] if not body else body.split(b"\n")), True


def _parse_line(line: bytes) -> Record:
    if len(line) + 1 > MAX_RECORD_BYTES:
        raise _ParseFailure(f"record size {len(line) + 1} exceeds {MAX_RECORD_BYTES} bytes.")
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _ParseFailure("record is not valid UTF-8.") from exc
    try:
        parsed: Any = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except _ParseFailure:
        raise
    except json.JSONDecodeError as exc:
        raise _ParseFailure("record is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise _ParseFailure("record must be a JSON object.")
    data = cast(dict[str, Any], parsed)
    timestamp = data.get("timestamp_utc")
    if not isinstance(timestamp, str) or _CANONICAL_TIMESTAMP.fullmatch(timestamp) is None:
        raise _ParseFailure("timestamp_utc is not fixed-width canonical UTC.")
    schema = data.get("schema")
    try:
        if schema == EVENT_SCHEMA:
            record: Record = EventRecord.model_validate_json(line, strict=True)
        elif schema == METRIC_SCHEMA:
            record = MetricRecord.model_validate_json(line, strict=True)
        else:
            raise _ParseFailure(f"unknown record schema {schema!r}.")
    except ValidationError as exc:
        raise _ParseFailure("record failed strict schema validation.") from exc
    if record.to_deterministic_json() != line:
        raise _ParseFailure("record is not canonical deterministic JSON.")
    return record


def _binding(record: Record) -> Binding:
    return (
        record.run_id,
        record.attempt_id,
        record.specification_fingerprint,
        record.rank,
        record.local_rank,
        record.world_size,
    )


def _expected_binding(identity: AttemptIdentityRecord, process_context: ProcessContext) -> Binding:
    return (
        identity.run_id,
        identity.attempt_id,
        identity.fingerprint_digest_str(),
        process_context.rank,
        process_context.local_rank,
        process_context.world_size,
    )


def _is_open(record: Record) -> bool:
    return isinstance(record, EventRecord) and record.event_name == _OPEN_EVENT


def _is_close(record: Record) -> bool:
    return isinstance(record, EventRecord) and record.event_name == _CLOSE_EVENT


def _check_sequence(record: Record, accepted_count: int) -> None:
    if record.sequence != accepted_count:
        raise TelemetryLoadError(
            f"sequence {record.sequence} is not contiguous; expected {accepted_count}."
        )


def _check_monotonic(
    record: Record,
    previous_elapsed: int | None,
    previous_progress: dict[str, int | None],
) -> None:
    if previous_elapsed is not None and record.elapsed_ns < previous_elapsed:
        raise TelemetryLoadError("elapsed_ns regresses within the process stream.")
    progress = record.progress
    if progress is None:
        return
    for name in _PROGRESS_FIELDS:
        current = getattr(progress, name)
        previous = previous_progress[name]
        if current is not None and previous is not None and current < previous:
            raise TelemetryLoadError(f"progress.{name} regresses within the stream.")


def _commit_progress(
    progress: ProgressPosition | None, previous_progress: dict[str, int | None]
) -> None:
    if progress is None:
        return
    for name in _PROGRESS_FIELDS:
        current = getattr(progress, name)
        if current is not None:
            previous_progress[name] = current


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise TelemetryLoadError(f"telemetry stream not found: {path}") from exc
    except OSError as exc:
        raise TelemetryLoadError(f"could not read telemetry stream {path}.") from exc


def load_telemetry_stream(
    path: Path,
    *,
    expected_identity: AttemptIdentityRecord,
    expected_process_context: ProcessContext,
    require_closed: bool = True,
) -> list[Record]:
    """Load a stream only after complete canonical and external-binding checks."""
    raw = _read(path)
    lines, truncated = _split_records(raw)
    if truncated:
        raise TelemetryLoadError("telemetry stream has a truncated trailing record.")
    if not lines:
        raise TelemetryLoadError("telemetry stream contains no complete records.")

    expected = _expected_binding(expected_identity, expected_process_context)
    accepted: list[Record] = []
    previous_elapsed: int | None = None
    previous_progress: dict[str, int | None] = {
        "step": None,
        "update": None,
        "processed_tokens": None,
    }
    seen_close = False

    for line_number, line in enumerate(lines, start=1):
        if seen_close:
            raise TelemetryLoadError("telemetry stream contains a record after closure.")
        try:
            record = _parse_line(line)
        except _ParseFailure as exc:
            raise TelemetryLoadError(f"telemetry stream line {line_number}: {exc}") from exc
        if line_number == 1 and not _is_open(record):
            raise TelemetryLoadError("first record must be logging.stream_opened.")
        if _binding(record) != expected:
            raise TelemetryLoadError(
                f"telemetry stream line {line_number} does not match expected identity/process."
            )
        _check_sequence(record, len(accepted))
        _check_monotonic(record, previous_elapsed, previous_progress)
        accepted.append(record)
        previous_elapsed = record.elapsed_ns
        _commit_progress(record.progress, previous_progress)
        if _is_close(record):
            seen_close = True

    if require_closed and not seen_close:
        raise TelemetryLoadError("telemetry stream is missing logging.stream_closed.")
    return accepted


def scan_telemetry_stream(path: Path) -> ScanResult:
    """Return a validated prefix and complete/incomplete/corrupt status."""
    try:
        raw = _read(path)
    except TelemetryLoadError:
        return ScanResult("corrupt", [])
    lines, truncated = _split_records(raw)
    accepted: list[Record] = []
    internal_binding: Binding | None = None
    previous_elapsed: int | None = None
    previous_progress: dict[str, int | None] = {
        "step": None,
        "update": None,
        "processed_tokens": None,
    }
    seen_close = False

    for line in lines:
        if seen_close:
            return ScanResult("corrupt", accepted)
        try:
            record = _parse_line(line)
            if not accepted:
                if not _is_open(record):
                    return ScanResult("corrupt", accepted)
                internal_binding = _binding(record)
            elif _binding(record) != internal_binding:
                return ScanResult("corrupt", accepted)
            _check_sequence(record, len(accepted))
            _check_monotonic(record, previous_elapsed, previous_progress)
        except (_ParseFailure, TelemetryLoadError):
            return ScanResult("corrupt", accepted)
        accepted.append(record)
        previous_elapsed = record.elapsed_ns
        _commit_progress(record.progress, previous_progress)
        if _is_close(record):
            seen_close = True

    if truncated:
        return ScanResult("incomplete", accepted)
    if not accepted or not seen_close:
        return ScanResult("incomplete", accepted)
    return ScanResult("complete", accepted)
