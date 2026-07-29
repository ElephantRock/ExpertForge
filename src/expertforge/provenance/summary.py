"""Human-readable provenance summary (Issue #7).

A human-readable summary is available without replacing the machine-readable
artifact. Surfaces the key facts (identity, source, software, hardware) and
degrades gracefully when optional sections are absent. Token patterns are
redacted defense-in-depth.
"""

from __future__ import annotations

from typing import Any

from expertforge.provenance.record import ProvenanceRecord
from expertforge.provenance.software import redact_token_patterns

__all__ = ["format_provenance_summary"]


def _fmt_section(title: str, lines: list[str]) -> list[str]:
    if not lines:
        return [f"{title}: not captured"]
    return [f"{title}:"] + [f"  {ln}" for ln in lines]


def _source_lines(source: dict[str, Any] | None) -> list[str]:
    if not source:
        return []
    out: list[str] = []
    if "commit_sha" in source:
        out.append(f"commit: {source['commit_sha'][:12]}")
    if "is_clean" in source:
        out.append(f"clean: {source['is_clean']}")
        out.append(f"canonical: {source.get('is_canonical', '—')}")
    if "content_digest" in source:
        out.append(f"content digest: {source['content_digest'][:12]}…")
    return out


def _software_lines(software: dict[str, Any] | None) -> list[str]:
    if not software:
        return []
    out: list[str] = []
    py = software.get("python", {})
    if py:
        out.append(f"python: {py.get('version', '?')} ({py.get('implementation', '?')})")
    plat = software.get("platform", {})
    if plat:
        out.append(f"platform: {plat.get('system', '?')}/{plat.get('machine', '?')}")
    deps = software.get("dependencies", {})
    if deps:
        out.append(f"dependencies: {len(deps)} distributions")
    lock = software.get("lockfile", {})
    if lock:
        out.append(f"lockfile: {lock.get('status', '?')}")
    return out


def _hardware_lines(hardware: dict[str, Any] | None) -> list[str]:
    if not hardware:
        return []
    out: list[str] = []
    accel = hardware.get("accelerator", {})
    if accel:
        status = accel.get("status", "?")
        if status == "available":
            out.append(
                f"accelerator: {accel.get('framework', '?')} "
                f"({accel.get('device_count', '?')} device(s))"
            )
        else:
            out.append(f"accelerator: {status}")
    topo = hardware.get("topology", {})
    if topo:
        out.append(f"topology: {topo.get('status', '?')}")
    return out


def format_provenance_summary(record: ProvenanceRecord) -> str:
    """Return a human-readable summary of a provenance record."""
    lines: list[str] = [
        f"run: {record.run_id}",
        f"attempt: {record.attempt_id}",
        f"specification: {record.specification_fingerprint.digest_str}",
        f"started (utc): {record.start_time_utc.isoformat()}",
        "",
    ]
    lines += _fmt_section("source", _source_lines(record.source))
    lines += _fmt_section("software", _software_lines(record.software))
    lines += _fmt_section("hardware", _hardware_lines(record.hardware))
    if record.topology:
        lines.append(f"topology: {record.topology.get('status', '?')}")
    elif not record.hardware:
        lines.append("topology: not captured")
    text = "\n".join(lines)
    # Defense-in-depth: redact any token patterns that slipped through.
    return redact_token_patterns(text)
