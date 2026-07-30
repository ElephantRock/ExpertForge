"""Authoritative loader and diagnostic scanner for telemetry streams
(Issue #9 design comment 5136093570 §10).

Two boundaries:

- :func:`load_telemetry_stream` — authoritative verified load. Verifies valid
  UTF-8, newline-terminated records only, no duplicate JSON keys, known record
  schemas and versions, strict model validation (no scalar coercion), exact
  external identity/fingerprint/process binding, contiguous sequence numbers,
  non-decreasing elapsed time and progress, required open/close lifecycle, and no
  records after close. Raises :class:`TelemetryLoadError` on any violation.

- :func:`scan_telemetry_stream` — diagnostic scan that reports an accepted prefix
  plus a ``complete | incomplete | corrupt`` status without claiming completion.
  A missing close event is ``incomplete``; a malformed interior record, identity
  mismatch, sequence gap, or progress regression is ``corrupt``; a truncated
  trailing fragment is ``incomplete``.

This module has NO import-time side effects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from expertforge.identity.record import AttemptIdentityRecord
from expertforge.telemetry.models import (
    EVENT_SCHEMA,
    METRIC_SCHEMA,
    EventRecord,
    MetricRecord,
    ProcessContext,
    StreamStatus,
)

__all__ = ["ScanResult", "TelemetryLoadError", "load_telemetry_stream", "scan_telemetry_stream"]

_EVENT_STREAM_OPENED = "logging.stream_opened"
_EVENT_STREAM_CLOSED = "logging.stream_closed"


class TelemetryLoadError(Exception):
    """Raised when an authoritative telemetry load fails any validation."""


class ScanResult:
    """Diagnostic scan result: an accepted-prefix count and a stream status.

    Immutable by construction. ``records`` holds the validated accepted-prefix
    records (EventRecord | MetricRecord).
    """

    __slots__ = ("_status", "_records")

    def __init__(self, status: StreamStatus, records: list[EventRecord | MetricRecord]) -> None:
        self._status = status
        self._records = list(records)

    @property
    def status(self) -> StreamStatus:
        return self._status

    @property
    def accepted_count(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[EventRecord | MetricRecord]:
        return list(self._records)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ScanResult(status={self._status!r}, accepted_count={self.accepted_count})"


# ---------------------------------------------------------------------------
# Shared low-level parsing
# ---------------------------------------------------------------------------


class _ParseFailure(Exception):
    """Internal: a record line could not be parsed/validated."""

    def __init__(self, reason: str, *, truncated: bool = False) -> None:
        super().__init__(reason)
        self.truncated = truncated


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` that rejects duplicate JSON keys."""
    seen: set[str] = set()
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise _ParseFailure(f"duplicate JSON key {key!r}.")
        seen.add(key)
        out[key] = value
    return out


def _split_records(raw: bytes) -> tuple[list[bytes], bool]:
    """Split raw bytes into newline-terminated record byte-strings.

    Returns ``(records, trailing_partial)``. Each entry in ``records`` is a
    complete newline-terminated record body (the trailing newline stripped).
    ``trailing_partial`` is True when the file does NOT end with a newline,
    meaning the bytes after the last interior newline are a partial (possibly
    empty) fragment that is not a complete record. The partial fragment itself
    is NOT included in ``records``.
    """
    if raw == b"":
        return ([], False)
    if raw.endswith(b"\n"):
        body = raw[:-1]
        if body == b"":
            return ([], False)
        return (body.split(b"\n"), False)
    # No trailing newline: everything after the last interior newline is a
    # partial fragment. The complete records are everything before it.
    last_newline = raw.rfind(b"\n")
    if last_newline < 0:
        # No newline at all: the whole payload is a single partial fragment.
        return ([], True)
    body = raw[:last_newline]
    if body == b"":
        return ([], True)
    return (body.split(b"\n"), True)


def _parse_record(line_bytes: bytes) -> dict[str, Any]:
    """Parse one newline-terminated record body into a mapping, rejecting dup keys
    and non-object JSON. Invalid UTF-8 and JSON errors raise :class:`_ParseFailure`.
    """
    try:
        text = line_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _ParseFailure(f"record is not valid UTF-8: {e}") from e
    try:
        data = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except _ParseFailure:
        raise
    except json.JSONDecodeError as e:
        raise _ParseFailure(f"record is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise _ParseFailure(f"record must be a JSON object; got {type(data).__name__}.")
    return data


def _validate_record(line_bytes: bytes, data: dict[str, Any]) -> EventRecord | MetricRecord:
    """Dispatch by schema name and strict-validate. Raises :class:`_ParseFailure`
    on unknown schema or validation failure.

    Validation uses ``model_validate_json`` so that JSON's native datetime/tuple
    representations (ISO-8601 string, JSON array) deserialize correctly while the
    model's ``strict=True`` config still forbids scalar coercion (e.g. a string
    where an int is required) and ``extra="forbid"`` rejects unknown keys.
    Duplicate-key detection is performed separately in :func:`_parse_record`.
    """
    schema = data.get("schema")
    try:
        if schema == EVENT_SCHEMA:
            return EventRecord.model_validate_json(line_bytes)
        if schema == METRIC_SCHEMA:
            return MetricRecord.model_validate_json(line_bytes)
        raise _ParseFailure(f"unknown record schema {schema!r}.")
    except ValidationError as e:
        raise _ParseFailure(f"record failed strict validation: {e}") from e


# ---------------------------------------------------------------------------
# Authoritative load
# ---------------------------------------------------------------------------


def load_telemetry_stream(
    path: Path,
    *,
    expected_identity: AttemptIdentityRecord,
    expected_process_context: ProcessContext,
    require_closed: bool = True,
) -> list[EventRecord | MetricRecord]:
    """Authoritatively load and verify a telemetry stream.

    Raises :class:`TelemetryLoadError` on: missing file, I/O error, invalid
    UTF-8, non-newline-terminated records, duplicate JSON keys, non-object JSON,
    unknown schema/version, strict-validation failure, identity/process mismatch,
    non-contiguous sequence, elapsed/progress regression, missing open event,
    missing close event (when ``require_closed``), or any record after close.
    """
    raw = _read_or_raise(path)
    records, truncated = _split_records(raw)
    if truncated:
        raise TelemetryLoadError(
            f"telemetry stream {path} is not newline-terminated (truncated tail)."
        )

    validated: list[EventRecord | MetricRecord] = []
    prev_elapsed: int | None = None
    prev_progress: dict[str, int | None] = {"step": None, "update": None, "processed_tokens": None}
    seen_close = False

    for index, line_bytes in enumerate(records):
        if seen_close:
            raise TelemetryLoadError(
                f"telemetry stream {path} has a record after the close event (line {index + 1})."
            )
        try:
            data = _parse_record(line_bytes)
            record = _validate_record(line_bytes, data)
        except _ParseFailure as e:
            raise TelemetryLoadError(f"telemetry stream {path} line {index + 1}: {e}") from e

        _check_identity_binding(
            path, record, expected_identity, expected_process_context, index + 1
        )
        _check_sequence(path, validated, record, index + 1)
        _check_monotonic(path, record, prev_elapsed, prev_progress, index + 1)
        prev_elapsed = record.elapsed_ns
        _merge_progress(prev_progress, record.progress)

        if index == 0:
            if not (isinstance(record, EventRecord) and record.event_name == _EVENT_STREAM_OPENED):
                raise TelemetryLoadError(
                    f"telemetry stream {path} first record must be {_EVENT_STREAM_OPENED}; "
                    f"got {_event_id(record)}."
                )
        if isinstance(record, EventRecord) and record.event_name == _EVENT_STREAM_CLOSED:
            seen_close = True

        validated.append(record)

    if not validated:
        raise TelemetryLoadError(f"telemetry stream {path} contains no records.")

    if not seen_close:
        if require_closed:
            raise TelemetryLoadError(
                f"telemetry stream {path} is missing the required close event."
            )
    return validated


# ---------------------------------------------------------------------------
# Diagnostic scan
# ---------------------------------------------------------------------------


def scan_telemetry_stream(path: Path) -> ScanResult:
    """Diagnostic scan: report an accepted prefix plus a stream status.

    Returns a :class:`ScanResult` with ``status`` in ``complete | incomplete |
    corrupt`` and ``accepted_count`` records that validated cleanly. A malformed
    interior record makes the status ``corrupt``; a missing close or truncated
    trailing fragment makes it ``incomplete``.
    """
    try:
        raw = _read_or_raise(path)
    except TelemetryLoadError as e:
        # Unreadable file: nothing accepted, corrupt.
        del e
        return ScanResult("corrupt", [])

    records, truncated = _split_records(raw)

    accepted: list[EventRecord | MetricRecord] = []
    prev_elapsed: int | None = None
    prev_progress: dict[str, int | None] = {
        "step": None,
        "update": None,
        "processed_tokens": None,
    }
    seen_close = False
    interior_corrupt = False

    for index, line_bytes in enumerate(records):
        try:
            data = _parse_record(line_bytes)
            record = _validate_record(line_bytes, data)
        except _ParseFailure:
            interior_corrupt = True
            break
        # Sequence/monotonicity/identity are structural invariants; a violation
        # corrupts the stream at this point.
        try:
            _check_sequence(path, accepted, record, index + 1)
            _check_monotonic(path, record, prev_elapsed, prev_progress, index + 1)
        except TelemetryLoadError:
            interior_corrupt = True
            break
        prev_elapsed = record.elapsed_ns
        _merge_progress(prev_progress, record.progress)
        if seen_close:
            interior_corrupt = True
            break
        if isinstance(record, EventRecord) and record.event_name == _EVENT_STREAM_CLOSED:
            seen_close = True
        accepted.append(record)

    if interior_corrupt:
        return ScanResult("corrupt", accepted)

    if truncated:
        # A trailing partial fragment (no newline) discards the fragment; the
        # accepted prefix is intact but the stream is not complete.
        return ScanResult("incomplete", accepted)

    if not accepted:
        # Empty file (no bytes): no accepted records, not complete.
        return ScanResult("incomplete", accepted)

    if seen_close:
        return ScanResult("complete", accepted)
    return ScanResult("incomplete", accepted)


# ---------------------------------------------------------------------------
# Internal checks
# ---------------------------------------------------------------------------


def _read_or_raise(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as e:
        raise TelemetryLoadError(f"telemetry stream not found: {path}") from e
    except OSError as e:
        raise TelemetryLoadError(f"could not read telemetry stream {path}: {e}") from e


def _event_id(record: EventRecord | MetricRecord) -> str:
    if isinstance(record, EventRecord):
        return f"event {record.event_name!r}"
    return "metric record"


def _check_identity_binding(
    path: Path,
    record: EventRecord | MetricRecord,
    expected_identity: AttemptIdentityRecord,
    expected_process_context: ProcessContext,
    line: int,
) -> None:
    if record.run_id != expected_identity.run_id:
        raise TelemetryLoadError(
            f"telemetry stream {path} line {line}: run_id {record.run_id!r} "
            f"does not match expected {expected_identity.run_id!r}."
        )
    if record.attempt_id != expected_identity.attempt_id:
        raise TelemetryLoadError(
            f"telemetry stream {path} line {line}: attempt_id {record.attempt_id!r} "
            f"does not match expected {expected_identity.attempt_id!r}."
        )
    expected_fp = expected_identity.fingerprint_digest_str()
    if record.specification_fingerprint != expected_fp:
        raise TelemetryLoadError(
            f"telemetry stream {path} line {line}: specification_fingerprint "
            f"{record.specification_fingerprint!r} does not match expected {expected_fp!r}."
        )
    ctx = expected_process_context
    if record.rank != ctx.rank or record.world_size != ctx.world_size:
        raise TelemetryLoadError(
            f"telemetry stream {path} line {line}: rank/world_size "
            f"({record.rank},{record.world_size}) do not match expected "
            f"({ctx.rank},{ctx.world_size})."
        )
    rec_local = record.local_rank
    exp_local = ctx.local_rank
    if rec_local != exp_local:
        raise TelemetryLoadError(
            f"telemetry stream {path} line {line}: local_rank {rec_local!r} "
            f"does not match expected {exp_local!r}."
        )


def _check_sequence(
    path: Path,
    prior: list[EventRecord | MetricRecord],
    record: EventRecord | MetricRecord,
    line: int,
) -> None:
    expected_seq = len(prior)
    if record.sequence != expected_seq:
        raise TelemetryLoadError(
            f"telemetry stream {path} line {line}: sequence {record.sequence} is not "
            f"contiguous (expected {expected_seq})."
        )


def _check_monotonic(
    path: Path,
    record: EventRecord | MetricRecord,
    prev_elapsed: int | None,
    prev_progress: dict[str, int | None],
    line: int,
) -> None:
    if prev_elapsed is not None and record.elapsed_ns < prev_elapsed:
        raise TelemetryLoadError(
            f"telemetry stream {path} line {line}: elapsed_ns {record.elapsed_ns} "
            f"regresses below previous {prev_elapsed}."
        )
    if record.progress is not None:
        for field in ("step", "update", "processed_tokens"):
            cur = getattr(record.progress, field)
            prev = prev_progress[field]
            if cur is not None and prev is not None and cur < prev:
                raise TelemetryLoadError(
                    f"telemetry stream {path} line {line}: progress.{field} {cur} "
                    f"regresses below previous {prev}."
                )


def _merge_progress(prev_progress: dict[str, int | None], progress: Any) -> None:
    """Update the running progress maxima with the latest supplied counters.

    Omitted fields make no progress claim and do not reset prior values.
    """
    if progress is None:
        return
    for field in ("step", "update", "processed_tokens"):
        cur = getattr(progress, field)
        if cur is not None:
            prev = prev_progress[field]
            prev_progress[field] = cur if prev is None else max(prev, cur)
