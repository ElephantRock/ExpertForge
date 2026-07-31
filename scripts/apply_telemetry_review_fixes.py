"""Apply the exact-head Issue #9 review corrections, then remove this helper."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{relative}: expected one replacement, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


replace_once(
    "src/expertforge/config/models.py",
    'level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")',
    'level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")',
)

replace_once(
    "src/expertforge/telemetry/models.py",
    '    "StreamStatus",\n    "WriterStats",',
    '    "StreamStatus",\n    "Severity",\n    "WriterStats",',
)
replace_once(
    "src/expertforge/telemetry/models.py",
    '''_RE_UNIX_ABS_PATH = re.compile(\n    r"(?<!:)(?<![A-Za-z0-9])/(?:tmp|var|opt|srv|mnt|media|private|etc)/[^\\s\\\"']+"\n)''',
    '''_RE_UNIX_ABS_PATH = re.compile(r"(?<!:)(?<![A-Za-z0-9])/(?!/)[^\\s\\\"']+")''',
)
replace_once(
    "src/expertforge/telemetry/models.py",
    '''    if _RE_IPV4.fullmatch(hostname) or ":" in hostname:\n        netloc = "[redacted]"\n    else:\n        netloc = f"{hostname}{port}"\n    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))''',
    '''    sensitive = (\n        parsed.username is not None\n        or parsed.password is not None\n        or bool(parsed.query)\n        or bool(parsed.fragment)\n        or _RE_IPV4.fullmatch(hostname) is not None\n        or ":" in hostname\n    )\n    if sensitive:\n        return "[redacted]"\n    netloc = f"{hostname}{port}"\n    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))''',
)

replace_once(
    "src/expertforge/telemetry/writer.py",
    '''        except BaseException:\n            if self._records_written == 0:\n                self._safe_unlink()\n            else:\n                self._failed = True\n                self._close_fd_silently()\n            raise''',
    '''        except BaseException as exc:\n            if self._records_written == 0:\n                self._safe_unlink()\n            else:\n                self._failed = True\n                self._close_fd_silently()\n            if not isinstance(exc, Exception):\n                raise\n            if isinstance(exc, WriterFailedError):\n                raise\n            raise WriterFailedError("telemetry writer initialization failed.") from exc''',
)
replace_once(
    "src/expertforge/telemetry/writer.py",
    '''            self._write_record(record)\n            self._fsync_or_fail()\n            self._render_console(record)\n            self._close_fd_or_fail()''',
    '''            self._write_record(record)\n            self._fsync_or_fail()\n            self._render_console(record, allow_diagnostic=False)\n            self._close_fd_or_fail()''',
)
replace_once(
    "src/expertforge/telemetry/writer.py",
    '''        except WriterFailedError:\n            self._closed = True\n            raise\n\n    def __enter__''',
    '''        except WriterFailedError:\n            self._closed = True\n            raise\n        except Exception as exc:\n            self._failed = True\n            self._closed = True\n            self._close_fd_silently()\n            raise WriterFailedError("telemetry close failed before completion.") from exc\n\n    def __enter__''',
)
replace_once(
    "src/expertforge/telemetry/writer.py",
    '''    def _render_console(self, record: EventRecord | MetricRecord) -> None:\n        if not self._console_enabled or self._console_failed_reported:''',
    '''    def _render_console(\n        self,\n        record: EventRecord | MetricRecord,\n        *,\n        allow_diagnostic: bool = True,\n    ) -> None:\n        if not self._console_enabled or self._console_failed_reported:''',
)
replace_once(
    "src/expertforge/telemetry/writer.py",
    '''            if not self._failed and not self._closed:\n                self._emit_console_write_failed()''',
    '''            if allow_diagnostic and not self._failed and not self._closed:\n                self._emit_console_write_failed()''',
)

loader = dedent(r'''\
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
    _CANONICAL_TIMESTAMP = re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$"
    )
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
            raise _ParseFailure(
                f"record size {len(line) + 1} exceeds {MAX_RECORD_BYTES} bytes."
            )
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


    def _expected_binding(
        identity: AttemptIdentityRecord, process_context: ProcessContext
    ) -> Binding:
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
                raise TelemetryLoadError(
                    f"telemetry stream line {line_number}: {exc}"
                ) from exc
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
''')
(ROOT / "src/expertforge/telemetry/loader.py").write_text(loader, encoding="utf-8")

fast_tests = dedent(r'''\
    """Exact-head regressions for the Issue #9 review contract."""

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

    RUN_ID = "run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb"
    ATTEMPT_ID = "attempt-20260101t000000z-cccccccccccccccccccc"
    NOW = datetime(2026, 1, 1, tzinfo=UTC)


    def identity() -> AttemptIdentityRecord:
        return AttemptIdentityRecord(
            specification_fingerprint=specification_fingerprint(b'{"telemetry":1}'),
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
            created_at_utc=NOW,
        )


    def context() -> ProcessContext:
        return ProcessContext(rank=0, world_size=1, local_rank=0)


    class Clock:
        def __init__(self, *values: int) -> None:
            self.values = iter(values or (0, 1, 2, 3, 4, 5, 6, 7, 8))

        def __call__(self) -> int:
            return next(self.values)


    def writer(
        root: Path,
        *,
        monotonic: Any | None = None,
        console: Any | None = None,
        level: str = "INFO",
    ) -> TelemetryWriter:
        return TelemetryWriter(
            root,
            identity(),
            context(),
            console_enabled=console is not None,
            console_stream=console,
            wall_clock=lambda: NOW,
            monotonic_clock=monotonic or Clock(),
            level=level,
        )


    def lines(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


    def canonical(data: dict[str, Any]) -> bytes:
        return json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8") + b"\n"


    def test_writer_rejects_elapsed_regression_without_appending(tmp_path: Path) -> None:
        telemetry = writer(tmp_path, monotonic=Clock(0, 10, 5))
        telemetry.emit_event(component="training", severity="INFO", event_name="training.update")
        before = telemetry.path.read_bytes()
        with pytest.raises(WriterFailedError):
            telemetry.emit_event(component="training", severity="INFO", event_name="training.update")
        assert telemetry.path.read_bytes() == before
        accepted = load_telemetry_stream(
            telemetry.path,
            expected_identity=identity(),
            expected_process_context=context(),
            require_closed=False,
        )
        assert [record.sequence for record in accepted] == [0, 1]


    def test_progress_regression_is_rejected_before_sequence_allocation(tmp_path: Path) -> None:
        telemetry = writer(tmp_path)
        telemetry.emit_event(
            component="training",
            severity="INFO",
            event_name="training.update",
            progress=ProgressPosition(step=5, update=2, processed_tokens=100),
        )
        with pytest.raises(ValueError, match="regresses"):
            telemetry.emit_metric(
                component="training",
                progress=ProgressPosition(step=4),
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
        telemetry.emit_event(
            component="training",
            severity="INFO",
            event_name="training.update",
            progress=ProgressPosition(step=6, update=3, processed_tokens=120),
        )
        telemetry.close("normal")
        loaded = load_telemetry_stream(
            telemetry.path,
            expected_identity=identity(),
            expected_process_context=context(),
        )
        assert [record.sequence for record in loaded] == [0, 1, 2, 3]


    def test_initial_validation_failure_removes_exclusive_path(tmp_path: Path) -> None:
        with pytest.raises(WriterFailedError, match="initialization"):
            TelemetryWriter(
                tmp_path,
                identity(),
                context(),
                console_enabled=False,
                wall_clock=lambda: datetime(2026, 1, 1),
                monotonic_clock=Clock(),
            )
        path = (
            tmp_path
            / RUN_ID
            / "attempts"
            / ATTEMPT_ID
            / "logs"
            / "telemetry-rank-0000000000.jsonl"
        )
        assert not path.exists()
        retry = writer(tmp_path)
        retry.close("normal")


    class FailOnWrite:
        def __init__(self, failure_call: int, *, short: bool = False) -> None:
            self.failure_call = failure_call
            self.short = short
            self.calls = 0

        def write(self, data: str) -> int:
            self.calls += 1
            if self.calls == self.failure_call:
                if self.short:
                    return max(0, len(data) - 1)
                raise ValueError("closed console")
            return len(data)

        def flush(self) -> None:
            return None


    def test_terminal_console_failure_never_appends_after_close(tmp_path: Path) -> None:
        sink = FailOnWrite(2)
        telemetry = writer(tmp_path, console=sink)
        telemetry.close("normal")
        loaded = load_telemetry_stream(
            telemetry.path,
            expected_identity=identity(),
            expected_process_context=context(),
        )
        assert [getattr(record, "event_name", None) for record in loaded] == [
            "logging.stream_opened",
            "logging.stream_closed",
        ]
        assert telemetry.stats.console_failures == 1


    def test_console_short_write_degrades_once(tmp_path: Path) -> None:
        sink = FailOnWrite(1, short=True)
        telemetry = writer(tmp_path, console=sink)
        telemetry.emit_event(component="training", severity="INFO", event_name="training.update")
        telemetry.close("normal")
        records = lines(telemetry.path)
        diagnostics = [
            record for record in records if record.get("diagnostic_code") == "console_write_failed"
        ]
        assert len(diagnostics) == 1
        assert sink.calls == 1


    def test_context_manager_records_unhandled_exception_as_failed(tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            with writer(tmp_path) as telemetry:
                path = telemetry.path
                raise RuntimeError("not persisted")
        loaded = load_telemetry_stream(
            path,
            expected_identity=identity(),
            expected_process_context=context(),
        )
        close = loaded[-1]
        assert isinstance(close, EventRecord)
        assert close.fields == (EventField(name="outcome", value="failed"),)
        assert close.diagnostic_code == "unhandled_exception"


    def test_severity_filter_does_not_allocate_sequences_or_filter_metrics(tmp_path: Path) -> None:
        telemetry = writer(tmp_path, level="WARNING")
        telemetry.emit_event(component="training", severity="DEBUG", event_name="training.debug")
        telemetry.emit_event(component="training", severity="INFO", event_name="training.info")
        telemetry.emit_metric(
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
        telemetry.emit_event(
            component="training", severity="WARNING", event_name="training.warning"
        )
        telemetry.close("normal")
        loaded = load_telemetry_stream(
            telemetry.path,
            expected_identity=identity(),
            expected_process_context=context(),
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
        telemetry = writer(tmp_path)
        telemetry.emit_metric(component="training", observations=tuple(reversed(observations)))
        telemetry.close("normal")
        loaded = load_telemetry_stream(
            telemetry.path,
            expected_identity=identity(),
            expected_process_context=context(),
        )
        metric = loaded[1]
        assert isinstance(metric, MetricRecord)
        assert metric.observations == observations
        with pytest.raises(Exception):
            MetricRecord.model_validate(
                {**metric.model_dump(), "observations": (observations[0], observations[0])}
            )


    def test_all_persisted_event_strings_are_sanitized_with_typed_evidence(tmp_path: Path) -> None:
        raw_secret = "pass" + "word=hunter2"
        telemetry = writer(tmp_path)
        telemetry.emit_event(
            component="training",
            severity="WARNING",
            event_name="training.warning",
            fields=(("detail", raw_secret), ("path", "/workspace/private/run.log")),
            operator_message="endpoint=[2001:db8::1]:9000",
        )
        telemetry.close("normal")
        event = lines(telemetry.path)[1]
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
        telemetry = writer(tmp_path)
        telemetry.close("normal")
        raw_lines = telemetry.path.read_bytes().splitlines()
        opened = json.loads(raw_lines[0])
        opened["timestamp_utc"] = "2026-01-01T00:00:00Z"
        bad_timestamp = tmp_path / "bad-timestamp.jsonl"
        bad_timestamp.write_bytes(canonical(opened) + raw_lines[1] + b"\n")
        with pytest.raises(TelemetryLoadError, match="timestamp"):
            load_telemetry_stream(
                bad_timestamp,
                expected_identity=identity(),
                expected_process_context=context(),
            )
        assert scan_telemetry_stream(bad_timestamp).status == "corrupt"

        oversized = tmp_path / "oversized.jsonl"
        oversized.write_bytes(b" " * MAX_RECORD_BYTES + raw_lines[0] + b"\n")
        with pytest.raises(TelemetryLoadError, match="size"):
            load_telemetry_stream(
                oversized,
                expected_identity=identity(),
                expected_process_context=context(),
            )
        assert scan_telemetry_stream(oversized).status == "corrupt"


    def test_loader_and_scan_enforce_exact_lifecycle_and_internal_binding(tmp_path: Path) -> None:
        telemetry = writer(tmp_path)
        telemetry.emit_event(component="training", severity="INFO", event_name="training.update")
        telemetry.close("normal")
        raw_lines = telemetry.path.read_bytes().splitlines()

        no_open = json.loads(raw_lines[0])
        no_open["component"] = "training"
        no_open["event_name"] = "training.update"
        no_open_path = tmp_path / "no-open.jsonl"
        no_open_path.write_bytes(canonical(no_open) + raw_lines[-1] + b"\n")
        assert scan_telemetry_stream(no_open_path).status == "corrupt"

        switched = json.loads(raw_lines[1])
        switched["attempt_id"] = "attempt-switched"
        switched_path = tmp_path / "switched.jsonl"
        switched_path.write_bytes(
            raw_lines[0] + b"\n" + canonical(switched) + raw_lines[-1] + b"\n"
        )
        assert scan_telemetry_stream(switched_path).status == "corrupt"

        malformed_close = json.loads(raw_lines[-1])
        malformed_close["fields"] = []
        malformed_path = tmp_path / "malformed-close.jsonl"
        malformed_path.write_bytes(raw_lines[0] + b"\n" + canonical(malformed_close))
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(
                malformed_path,
                expected_identity=identity(),
                expected_process_context=context(),
            )
        assert scan_telemetry_stream(malformed_path).status == "corrupt"


    def test_zero_microseconds_are_serialized_fixed_width(tmp_path: Path) -> None:
        telemetry = writer(tmp_path)
        telemetry.close("normal")
        assert b'"timestamp_utc":"2026-01-01T00:00:00.000000Z"' in telemetry.path.read_bytes()
''')
(ROOT / "tests/test_telemetry_review_contract.py").write_text(fast_tests, encoding="utf-8")

integration_tests = dedent(r'''\
    """Portable cross-component regressions for the accepted telemetry contract."""

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
''')
(ROOT / "tests/integration/test_telemetry_review_contract.py").write_text(
    integration_tests, encoding="utf-8"
)
