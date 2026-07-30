"""Sanitized software-environment capture (Issue #7 review item 4).

Allowlist-first capture returning typed frozen models. Includes CPU count,
discoverable host memory, platform facts, distribution versions, and lockfile
digest. No env/user/host/paths/IPs/serials/raw stderr.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from importlib import metadata
from pathlib import Path
from platform import machine, python_compiler, system

from expertforge.provenance.record import (
    CPUInfo,
    DependencyObservation,
    LockfileDigest,
    MemoryInfo,
    PlatformInfo,
    PythonInfo,
    SoftwareEnvironment,
)

__all__ = [
    "capture_software_environment",
    "dependency_notes",
    "redact_token_patterns",
    "sanitize_repository_url",
]


def dependency_notes() -> tuple[str, ...]:
    """Return any dependency-capture notes (e.g. version conflicts, first wins)."""
    return tuple(_dependency_notes)


_TOKEN_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
]

# Notes populated by :func:`_capture_dependencies` when the SAME normalized
# dependency name is observed at DIFFERENT versions. First observation wins;
# the conflict is recorded here rather than silently overwriting.
_dependency_notes: list[str] = []


def redact_token_patterns(value: str) -> str:
    out = value
    for pat in _TOKEN_PATTERNS:
        out = pat.sub("[redacted]", out)
    return out


def sanitize_repository_url(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    if re.match(r"^[A-Za-z]:[\\/]", raw):
        return None
    if raw.startswith("/") or raw.startswith("\\"):
        return None
    if raw.startswith("file://"):
        return None
    if raw.startswith("git@"):
        return raw
    m = re.match(r"^(?P<scheme>https?://)(?P<userinfo>[^/@]+@)?(?P<rest>.+)$", raw)
    if not m:
        return None
    rest = re.split(r"[?#]", m.group("rest"), maxsplit=1)[0]
    return m.group("scheme") + rest


def _capture_python() -> PythonInfo:
    """Capture typed Python interpreter info (version, implementation, build)."""
    build = python_compiler() or None
    return PythonInfo(
        version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        implementation=sys.implementation.name,
        build=build,
    )


def _capture_cpu() -> CPUInfo:
    count = os.cpu_count()
    return CPUInfo(
        status="available" if count else "unavailable",
        count=count,
        architecture=machine() or None,
    )


def _capture_memory() -> MemoryInfo:
    """Capture host physical memory. Uses no process-specific rlimit values.

    On platforms without a reliable stdlib mechanism, degrades to unavailable
    rather than reporting a misleading process address-space limit.
    """
    total: int | None = None
    try:
        # psutil is not a dependency; degrade gracefully if absent.
        import psutil  # type: ignore[import-untyped]

        total = psutil.virtual_memory().total
    except ImportError:
        pass
    return MemoryInfo(
        status="available" if total else "unavailable",
        total_bytes=total,
    )


def _capture_platform() -> PlatformInfo:
    return PlatformInfo(
        status="available",
        system=system() or "unknown",
        machine=machine() or "unknown",
        cpu=_capture_cpu(),
        memory=_capture_memory(),
    )


def _capture_dependencies() -> tuple[DependencyObservation, ...]:
    """Installed distributions as sorted, deduplicated typed observations.

    Names are PEP 503 normalized. When the SAME normalized name appears with
    DIFFERENT versions, the FIRST observation wins (no silent last-write-wins)
    and a note is recorded via the module-level :data:`_dependency_notes`.
    """
    path_like = re.compile(r"(file://|/Users/|/home/|[A-Za-z]:\\\\)")
    first_version: dict[str, str] = {}
    order: list[str] = []
    conflicts: list[str] = []
    for dist in metadata.distributions():
        raw_name = (dist.metadata["Name"] or "").strip()
        if not raw_name:
            continue
        # Normalize: PEP 503 canonical form (lowercase, runs of -_. → single -).
        normalized = re.sub(r"[-_.]+", "-", raw_name).lower()
        version = (dist.version or "").strip()
        if version and path_like.search(version):
            continue
        if normalized in first_version:
            if version != first_version[normalized] and version:
                # Same normalized name, different version: keep the first
                # (do NOT silently last-write-wins) and note the conflict.
                conflicts.append(
                    f"dependency_version_conflict:{normalized}:"
                    f"{first_version[normalized]}!={version}"
                )
            continue
        first_version[normalized] = version
        order.append(normalized)
    _dependency_notes.clear()
    _dependency_notes.extend(conflicts)
    observations = tuple(
        DependencyObservation(name=name, version=first_version[name]) for name in sorted(order)
    )
    return observations


def _capture_lockfile(repo_root: Path | None) -> LockfileDigest:
    if repo_root is None:
        return LockfileDigest(status="not_applicable")
    lockfile = repo_root / "uv.lock"
    if not lockfile.is_file():
        return LockfileDigest(status="unavailable")
    try:
        digest = hashlib.sha256(lockfile.read_bytes()).hexdigest()
    except OSError:
        return LockfileDigest(status="error", reason="io_error")
    return LockfileDigest(status="available", algorithm="sha256", digest=digest)


def capture_software_environment(*, repo_root: Path | None = None) -> SoftwareEnvironment:
    """Capture the sanitized software environment as a typed frozen model."""
    return SoftwareEnvironment(
        python=_capture_python(),
        platform=_capture_platform(),
        dependencies=_capture_dependencies(),
        lockfile=_capture_lockfile(repo_root),
    )
