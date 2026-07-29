"""Sanitized software-environment capture (Issue #7 decision: secret handling).

V1 uses **allowlist-first capture**, not "capture everything and redact later."
Only stable, non-secret facts are recorded:

- Python version (``sys.version_info``);
- platform family (``system`` / ``machine``) — not hostname or username;
- installed distribution name → version (no local ``file://`` URLs);
- lockfile content digest, when a lockfile is present.

Must not record: unrestricted env vars, usernames/hostnames, home/repo/
interpreter absolute paths, coordinator addresses/IPs, hardware serials/MAC/
UUIDs, raw subprocess stderr, signed URL queries, or local dependency URLs.
Repository URLs are structurally sanitized. Token-pattern redaction is
defense-in-depth, not the primary boundary.
"""

from __future__ import annotations

import hashlib
import re
import sys
from importlib import metadata
from pathlib import Path
from platform import machine, system
from typing import Any

__all__ = [
    "capture_software_environment",
    "redact_token_patterns",
    "sanitize_repository_url",
]

# --- token redaction (defense in depth) -----------------------------------

# A token-like pattern set. These are stripped from any captured string as a
# backstop; the primary boundary is allowlist-first capture.
_TOKEN_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
]


def redact_token_patterns(value: str) -> str:
    """Replace token-like substrings in ``value`` with ``[redacted]``.

    Defense-in-depth; the primary security boundary is allowlist-first capture.
    """
    out = value
    for pat in _TOKEN_PATTERNS:
        out = pat.sub("[redacted]", out)
    return out


# --- repository URL sanitization ------------------------------------------


def sanitize_repository_url(raw: str) -> str | None:
    """Return a sanitized remote URL, or ``None`` if it is not a remote.

    Strips userinfo (credentials), query strings, fragments, and signed
    parameters. Local filesystem locations (``C:\\...``, ``/home/...``,
    ``file://...``) return ``None`` — they are not recorded.
    """
    raw = raw.strip()
    if not raw:
        return None
    # Local filesystem locations are not remote URLs.
    if re.match(r"^[A-Za-z]:[\\/]", raw):  # Windows drive path
        return None
    if raw.startswith("/") or raw.startswith("\\"):
        return None
    if raw.startswith("file://"):
        return None

    # SSH form git@host:owner/repo — no userinfo to strip beyond the git user,
    # which is a protocol artifact, not a credential. Keep as-is.
    if raw.startswith("git@"):
        return raw

    # https://[userinfo@]host/path[?query][#fragment]
    m = re.match(r"^(?P<scheme>https?://)(?P<userinfo>[^/@]+@)?(?P<rest>.+)$", raw)
    if not m:
        return None
    rest = m.group("rest")
    # Drop query and fragment.
    rest = re.split(r"[?#]", rest, maxsplit=1)[0]
    return m.group("scheme") + rest


# --- allowlist-first capture ----------------------------------------------


def _capture_python() -> dict[str, Any]:
    return {
        "status": "available",
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "implementation": sys.implementation.name,
    }


def _capture_platform() -> dict[str, Any]:
    # system()/machine() are stable, non-identifying facts (e.g. "Linux"/"x86_64").
    # Hostname and username are deliberately NOT captured.
    return {
        "status": "available",
        "system": system() or "unknown",
        "machine": machine() or "unknown",
    }


def _capture_dependencies() -> dict[str, Any]:
    """Installed distributions: name → version, excluding any path/URL values."""
    deps: dict[str, str] = {}
    path_like = re.compile(r"(file://|/Users/|/home/|[A-Za-z]:\\\\)")
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").strip()
        version = (dist.version or "").strip()
        if not name:
            continue
        # Defense-in-depth: drop any version that looks like a local path/URL.
        if version and path_like.search(version):
            continue
        deps[name] = version
    return deps


def _capture_lockfile(repo_root: Path | None) -> dict[str, Any]:
    """Content digest of ``uv.lock`` if present, else explicit ``unavailable``."""
    if repo_root is None:
        return {"status": "not_applicable"}
    lockfile = repo_root / "uv.lock"
    if not lockfile.is_file():
        return {"status": "unavailable"}
    try:
        digest = hashlib.sha256(lockfile.read_bytes()).hexdigest()
    except OSError as e:
        return {"status": "error", "reason": _sanitize_reason(e)}
    return {"status": "available", "algorithm": "sha256", "digest": digest}


def _sanitize_reason(exc: BaseException) -> str:
    """Reduce an exception to a stable, non-leaking reason code."""
    name = type(exc).__name__
    # Map common cases to stable codes; never surface the raw message.
    if isinstance(exc, (FileNotFoundError, PermissionError, OSError)):
        return "io_error"
    return name.lower()


def capture_software_environment(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Capture the sanitized software environment.

    Allowlist-first: only ``python``, ``platform``, ``dependencies``, and
    ``lockfile`` sections are returned. No environment variables, usernames,
    hostnames, or absolute paths are recorded.
    """
    return {
        "python": _capture_python(),
        "platform": _capture_platform(),
        "dependencies": _capture_dependencies(),
        "lockfile": _capture_lockfile(repo_root),
    }
