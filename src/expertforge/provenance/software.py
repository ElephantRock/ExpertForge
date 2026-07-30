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
from platform import machine, system

from expertforge.provenance.record import (
    CPUInfo,
    LockfileDigest,
    MemoryInfo,
    PlatformInfo,
    SoftwareEnvironment,
)

__all__ = [
    "capture_software_environment",
    "redact_token_patterns",
    "sanitize_repository_url",
]

_TOKEN_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
]


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


def _capture_python() -> dict[str, str]:
    return {
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "implementation": sys.implementation.name,
    }


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
        import psutil  # type: ignore[import-not-found]

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


def _capture_dependencies() -> dict[str, str]:
    """Installed distributions: normalized name → version, sorted, deduplicated."""
    deps: dict[str, str] = {}
    path_like = re.compile(r"(file://|/Users/|/home/|[A-Za-z]:\\\\)")
    for dist in metadata.distributions():
        raw_name = (dist.metadata["Name"] or "").strip()
        if not raw_name:
            continue
        # Normalize: PEP 503 canonical form (lowercase, runs of -_. → single -).
        normalized = re.sub(r"[-_.]+", "-", raw_name).lower()
        version = (dist.version or "").strip()
        if version and path_like.search(version):
            continue
        # Last-write-wins on exact normalized duplicates (same name+version).
        deps[normalized] = version
    # Return sorted by name for deterministic output.
    return dict(sorted(deps.items()))


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
