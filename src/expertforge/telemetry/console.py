"""Deterministic console renderer for telemetry records (Issue #9 §11).

The canonical JSONL stream is the machine source of truth. Console output is a
*deterministic pure rendering* of a validated record. It never contains
information absent from that record, never contains ANSI color, and is one line.

Console output is NEVER parsed as evidence and is not the artifact registered by
Issue #10. A console write failure must not invalidate an already-durable machine
record (the writer disables the failed console sink instead).
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from expertforge.telemetry.models import EventRecord, MetricObservation, MetricRecord

__all__ = ["Record", "render_record"]


class Record(Protocol):
    """Structural protocol for anything the renderer can render."""

    sequence: int
    event_name: str
    severity: str
    component: str
    rank: int
    timestamp_utc: datetime
    observations: tuple[MetricObservation, ...]


def _fmt_ts(ts: datetime) -> str:
    """Fixed-width RFC 3339 with six fractional digits and ``Z``."""
    # Coerce to UTC then format. The record already guarantees aware UTC.
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond:06d}Z"


def _fmt_value(obs: MetricObservation) -> str:
    """Render a metric observation's value/state deterministically."""
    status = obs.value_status
    if status == "finite":
        assert obs.value is not None  # invariant enforced by the model
        v = obs.value
        # Render integers without a trailing .0; floats via repr-like shortest form.
        if isinstance(v, float) and v.is_integer():
            return f"{v:.1f}"
        if isinstance(v, int):
            return str(v)
        return repr(v)
    return status


def render_event(record: EventRecord) -> str:
    """Render an :class:`EventRecord` as one deterministic line."""
    head = (
        f"[{_fmt_ts(record.timestamp_utc)}] seq={record.sequence} "
        f"rank={record.rank} component={record.component} "
        f"severity={record.severity} event={record.event_name}"
    )
    parts: list[str] = []
    if record.diagnostic_code is not None:
        parts.append(f"diagnostic={record.diagnostic_code}")
    if record.progress is not None:
        p = record.progress
        if p.step is not None:
            parts.append(f"step={p.step}")
        if p.update is not None:
            parts.append(f"update={p.update}")
        if p.processed_tokens is not None:
            parts.append(f"tokens={p.processed_tokens}")
    for f in record.fields:
        # Strict value types only; bool renders as true/false (JSON style).
        if isinstance(f.value, bool):
            parts.append(f"{f.name}={'true' if f.value else 'false'}")
        else:
            parts.append(
                f"{f.name}={f.value!r}" if isinstance(f.value, str) else f"{f.name}={f.value}"
            )
    if record.operator_message is not None:
        # Collapse any whitespace runs to keep it a single line.
        collapsed = " ".join(record.operator_message.split())
        parts.append(f"msg={collapsed!r}")
    return head + (" " + " ".join(parts) if parts else "")


def render_metric(record: MetricRecord) -> str:
    """Render a :class:`MetricRecord` as one deterministic line."""
    head = (
        f"[{_fmt_ts(record.timestamp_utc)}] seq={record.sequence} "
        f"rank={record.rank} component={record.component} metrics:"
    )
    pieces = [
        f"{o.namespace}.{o.name}={_fmt_value(o)} ({o.unit},{o.aggregation},{o.window})"
        for o in record.observations
    ]
    return head + " " + " ".join(pieces)


def render_record(record: EventRecord | MetricRecord) -> str:
    """Render an event or metric record as one deterministic, ANSI-free line.

    The output never contains information absent from ``record``.
    """
    if isinstance(record, EventRecord):
        return render_event(record)
    if isinstance(record, MetricRecord):
        return render_metric(record)
    raise TypeError(f"render_record received unsupported record type {type(record).__name__}.")
