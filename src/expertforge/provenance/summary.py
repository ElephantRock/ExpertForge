"""Human-readable provenance summary (Issue #7 review item 4).

Uses the typed nested models. Degrades gracefully when sections are absent.
Token patterns redacted defense-in-depth.
"""

from __future__ import annotations

from expertforge.provenance.record import ProvenanceRecord
from expertforge.provenance.software import redact_token_patterns

__all__ = ["format_provenance_summary"]


def _source_lines(record: ProvenanceRecord) -> list[str]:
    if record.source is None:
        return []
    s = record.source.summary
    return [
        f"commit: {s.commit_sha[:12]}",
        f"branch: {s.branch}",
        f"clean: {s.is_clean}",
        f"canonical: {s.is_canonical}",
        f"content digest: {s.tree_digest[:12]}…",
    ]


def _software_lines(record: ProvenanceRecord) -> list[str]:
    if record.software is None:
        return []
    sw = record.software
    py = sw.python
    plat = sw.platform
    out = [
        f"python: {py.get('version', '?')} ({py.get('implementation', '?')})",
        f"platform: {plat.system}/{plat.machine}",
    ]
    if plat.cpu.status == "available" and plat.cpu.count:
        out.append(f"cpu count: {plat.cpu.count}")
    if plat.memory.status == "available" and plat.memory.total_bytes:
        out.append(f"memory: {plat.memory.total_bytes} bytes")
    if sw.dependencies:
        out.append(f"dependencies: {len(sw.dependencies)} distributions")
    out.append(f"lockfile: {sw.lockfile.status}")
    return out


def _hardware_lines(record: ProvenanceRecord) -> list[str]:
    if record.hardware is None:
        return []
    hw = record.hardware
    if hw.status == "available":
        return [
            f"accelerator: {hw.framework or '?'} ({hw.device_count} device(s))",
        ]
    return [f"accelerator: {hw.status}"]


def format_provenance_summary(record: ProvenanceRecord) -> str:
    lines = [
        f"run: {record.run_id}",
        f"attempt: {record.attempt_id}",
        f"specification: {record.specification_fingerprint.digest_str}",
        f"started (utc): {record.start_time_utc.isoformat()}",
        "",
    ]
    src = _source_lines(record)
    if src:
        lines.append("source:")
        lines.extend(f"  {ln}" for ln in src)
    else:
        lines.append("source: not captured")
    sw = _software_lines(record)
    if sw:
        lines.append("software:")
        lines.extend(f"  {ln}" for ln in sw)
    else:
        lines.append("software: not captured")
    hw = _hardware_lines(record)
    if hw:
        lines.append("hardware:")
        lines.extend(f"  {ln}" for ln in hw)
    else:
        lines.append("hardware: not captured")
    if record.topology is not None:
        lines.append(f"topology: {record.topology.status}")
    text = "\n".join(lines)
    return redact_token_patterns(text)
