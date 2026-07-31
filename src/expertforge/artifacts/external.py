"""External-reference location safety (amendment G).

Credential-bearing locations are **rejected**, not silently sanitized. For
version 1:

- URI/object-store identifiers forbid userinfo, query strings, and fragments;
- filesystem locations must be canonical portable relative form tied to an
  explicit ``external_root_id``, not an arbitrary machine-specific absolute path.

A location alone is never treated as verified evidence; verification requires
re-hashing against ``expected_digest`` through a caller-supplied resolver.
Network/cloud transfer remains a non-goal.
"""

from __future__ import annotations

import re
from urllib.parse import SplitResult, urlsplit

__all__ = [
    "ExternalLocationError",
    "reject_credential_bearing_location",
]

# Permitted URI schemes for external references. ``file:`` is intentionally
# absent — local content lives in canonical bundles, not external references.
_PERMITTED_URI_SCHEMES = frozenset({"https", "s3", "gs", "az"})

# Canonical portable relative path: forward slashes only, no traversal, no
# drive letters, no absolute form.
_REL_PATH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\\-]*$")


class ExternalLocationError(Exception):
    """Raised when an external location carries credentials or is non-portable."""


def _split_for_credentials(value: str) -> SplitResult:
    # urlsplit gives (scheme, netloc, path, query, fragment).
    return urlsplit(value)


def reject_credential_bearing_location(
    *,
    location_type: str,
    location: str,
    external_root_id: str | None,
) -> None:
    """Raise :class:`ExternalLocationError` if ``location`` is credential-bearing
    or non-portable for its ``location_type``."""
    if location_type == "uri":
        parts = _split_for_credentials(location)
        scheme, netloc, _path, query, fragment = parts
        if scheme not in _PERMITTED_URI_SCHEMES:
            raise ExternalLocationError(
                f"URI scheme {scheme!r} is not permitted; allowed "
                f"{sorted(_PERMITTED_URI_SCHEMES)} (credential-bearing or local "
                f"schemes are rejected, not sanitized)."
            )
        if "@" in netloc:
            raise ExternalLocationError(
                "URI userinfo (credentials) is forbidden in external locations; "
                "credential-bearing locations are rejected, not sanitized."
            )
        if query:
            raise ExternalLocationError(
                "URI query strings are forbidden in external locations "
                "(may carry tokens); rejected, not sanitized."
            )
        if fragment:
            raise ExternalLocationError(
                "URI fragments are forbidden in external locations; "
                "rejected, not sanitized."
            )
        return

    if location_type == "object_store":
        # ``<scheme>://<bucket>/<key>`` form, same credential rules as URI.
        parts = _split_for_credentials(location)
        scheme, netloc, _path, query, fragment = parts
        if scheme not in _PERMITTED_URI_SCHEMES:
            raise ExternalLocationError(
                f"object-store scheme {scheme!r} is not permitted; allowed "
                f"{sorted(_PERMITTED_URI_SCHEMES)}."
            )
        if "@" in netloc or query or fragment:
            raise ExternalLocationError(
                "object-store location must not carry userinfo, query, or fragment "
                "(credential-bearing locations are rejected, not sanitized)."
            )
        if not netloc:
            raise ExternalLocationError("object-store location must include a bucket/host.")
        return

    if location_type == "filesystem_path":
        if external_root_id is None:
            raise ExternalLocationError(
                "filesystem_path external location requires an explicit "
                "external_root_id (machine-specific absolute paths are not portable)."
            )
        normalized = location.replace("\\", "/")
        if not normalized:
            raise ExternalLocationError("filesystem_path must be non-empty.")
        if normalized.startswith("/"):
            raise ExternalLocationError(
                "filesystem_path must be relative to external_root_id, not absolute."
            )
        if re.match(r"^[A-Za-z]:/", normalized):
            raise ExternalLocationError(
                "filesystem_path must be relative to external_root_id, not a drive path."
            )
        components = normalized.split("/")
        if any(part in ("..", ".") for part in components):
            raise ExternalLocationError(
                "filesystem_path must not contain '.' or '..' components."
            )
        if any(part == "" for part in components):
            raise ExternalLocationError(
                "filesystem_path must not contain empty components."
            )
        if not _REL_PATH_PATTERN.fullmatch(normalized):
            raise ExternalLocationError(
                "filesystem_path must be canonical portable relative form."
            )
        return

    raise ExternalLocationError(f"unknown location_type {location_type!r}.")
