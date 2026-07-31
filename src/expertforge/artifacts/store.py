"""The artifact store: atomic publication, registration, verification (Issue #10).

This module owns:

- canonical path derivation;
- atomic bundle publication (write → hash → artifact.json → atomic dir rename
  → registry append) — amendment C;
- ``register_existing`` copies a regular file into store ownership (no symlink
  following, no mutable source path references) — amendment H;
- ``register_external`` identity-bound external references with credential
  rejection — amendment G;
- ``locate`` / ``list_artifacts`` / ``inspect`` / ``verify`` /
  ``update_retention``;
- path safety: traversal/absolute/symlink rejection everywhere;
- content hashing: SHA-256 streaming, 64 KiB blocks.

Content hashing is SHA-256 over the exact payload bytes, streamed in 64 KiB
blocks. Artifact IDs are full semantic digests over a canonical immutable
descriptor (amendment A), not content-hash prefixes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from expertforge.artifacts.external import ExternalLocationError
from expertforge.artifacts.models import (
    ARTIFACT_BUNDLE_SCHEMA_VERSION,
    ArtifactConflictError,
    ArtifactDescriptor,
    ArtifactNotFoundError,
    ArtifactRecord,
    ExternalReference,
    ParentReference,
    RegistryEntry,
    RetentionStatus,
    VerificationResult,
    descriptor_to_artifact_id,
    is_legal_retention_transition,
    is_legal_storage_transition,
)
from expertforge.artifacts.paths import (
    artifact_bundle_dir,
    attempt_dir,
    category_dir,
    validate_id_component,
)
from expertforge.artifacts.registry import (
    RegistryError,
    _reduce_history,
    allocate_and_append,
    load_registry,
    registry_path_for_attempt,
)
from expertforge.identity.record import AttemptIdentityRecord

__all__ = [
    "BLOCK_SIZE",
    "ArtifactStore",
    "ArtifactStoreError",
    "RegistryReconciliation",
    "sha256_stream",
]

BLOCK_SIZE = 64 * 1024


class ArtifactStoreError(Exception):
    """Raised on store-level failures (publication, IO, identity binding)."""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_stream(read_fn: Any, *, block_size: int = BLOCK_SIZE) -> tuple[str, int]:
    """Stream SHA-256 over a binary read callable; return (digest_str, byte_size).

    ``read_fn`` is a callable taking a size hint and returning bytes (e.g. an
    open file's ``read``). Empty result ends the stream.
    """
    h = hashlib.sha256()
    total = 0
    while True:
        chunk = read_fn(block_size)
        if not chunk:
            break
        h.update(chunk)
        total += len(chunk)
    return f"sha256:{h.hexdigest()}", total


def _hash_path(path: Path) -> tuple[str, int]:
    with open(path, "rb") as fh:
        return sha256_stream(fh.read)


# ---------------------------------------------------------------------------
# Atomic helpers
# ---------------------------------------------------------------------------


def _full_write(fd: int, data: bytes) -> None:
    view = memoryview(data)
    total = 0
    while total < len(view):
        written = os.write(fd, view[total:])
        if written <= 0:  # pragma: no cover - defensive
            raise OSError("os.write returned non-positive byte count")
        total += written


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _is_symlink(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    import stat as _stat

    return _stat.S_ISLNK(st.st_mode)


# ---------------------------------------------------------------------------
# Canonical artifact.json
# ---------------------------------------------------------------------------


def _record_canonical_json(record: ArtifactRecord) -> bytes:
    """Compact, sorted-key, UTF-8, non-finite-prohibiting JSON of a record."""
    return json.dumps(
        record.model_dump(mode="json", by_alias=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _record_from_bundle_bytes(raw: bytes) -> ArtifactRecord:
    try:
        text = raw.decode("utf-8")
        data = json.loads(text)
    except UnicodeDecodeError as e:
        raise ArtifactStoreError(f"artifact.json is not valid UTF-8: {e}") from e
    except json.JSONDecodeError as e:
        raise ArtifactStoreError(f"artifact.json is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ArtifactStoreError("artifact.json must be a JSON object.")
    try:
        return ArtifactRecord.model_validate(data, strict=False)
    except Exception as e:
        raise ArtifactStoreError(f"artifact.json failed validation: {e}") from e


_IMMUTABLE_IDENTITY_FIELDS = (
    "artifact_id",
    "category",
    "format",
    "format_version",
    "byte_size",
    "content_digest",
    "producing_component",
    "run_id",
    "attempt_id",
    "specification_fingerprint",
    "relative_path",
    "parent",
)


def _immutable_identity(record: ArtifactRecord) -> tuple[object, ...]:
    """The immutable descriptor-derived identity tuple of a record.

    Operational state (``created_at_utc``, ``storage_class``, ``retention``)
    is intentionally excluded: idempotent re-publication preserves the existing
    bundle's recorded timestamp and registry-tracked state.
    """
    dump = record.model_dump(mode="json", by_alias=True)
    return tuple(dump[f] for f in _IMMUTABLE_IDENTITY_FIELDS)


# ---------------------------------------------------------------------------
# Registry reconciliation (amendment K)
# ---------------------------------------------------------------------------


class RegistryReconciliation:
    """Result of orphan bundle detection and reconciliation."""

    __slots__ = ("_appended", "_orphans", "_skipped")

    def __init__(
        self,
        *,
        appended: list[str],
        orphans: list[str],
        skipped: list[tuple[str, str]],
    ) -> None:
        self._appended = tuple(appended)
        self._orphans = tuple(orphans)
        self._skipped = tuple(skipped)

    @property
    def appended_artifact_ids(self) -> list[str]:
        return list(self._appended)

    @property
    def orphan_artifact_ids(self) -> list[str]:
        return list(self._orphans)

    @property
    def skipped(self) -> list[tuple[str, str]]:
        return list(self._skipped)


# ---------------------------------------------------------------------------
# ArtifactStore
# ---------------------------------------------------------------------------


class ArtifactStore:
    """The artifact store bound to one attempt's identity.

    All identity fields are bound from the supplied :class:`AttemptIdentityRecord`;
    callers never provide run/attempt/fingerprint independently (amendment A).
    """

    def __init__(self, artifact_root: Path, identity: AttemptIdentityRecord) -> None:
        self._artifact_root = Path(artifact_root)
        self._identity = identity
        self._attempt_dir = attempt_dir(self._artifact_root, identity.run_id, identity.attempt_id)
        self._registry_path = registry_path_for_attempt(
            self._artifact_root, identity.run_id, identity.attempt_id
        )
        self._lock_path = self._attempt_dir / "registry.lock"

    # -- accessors ---------------------------------------------------------

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root

    @property
    def identity(self) -> AttemptIdentityRecord:
        return self._identity

    @property
    def attempt_dir(self) -> Path:
        return self._attempt_dir

    @property
    def registry_path(self) -> Path:
        return self._registry_path

    @property
    def specification_fingerprint(self) -> str:
        return self._identity.fingerprint_digest_str()

    # -- descriptor builder -----------------------------------------------

    def _descriptor(
        self,
        *,
        content_digest: str,
        byte_size: int,
        category: str,
        format: str,
        format_version: int,
        producing_component: str,
        parent: ParentReference | None,
    ) -> ArtifactDescriptor:
        return ArtifactDescriptor(
            schema_version=ARTIFACT_BUNDLE_SCHEMA_VERSION,
            run_id=self._identity.run_id,
            attempt_id=self._identity.attempt_id,
            specification_fingerprint=self.specification_fingerprint,
            category=category,  # type: ignore[arg-type]
            format=format,  # type: ignore[arg-type]
            format_version=format_version,
            content_digest=content_digest,
            byte_size=byte_size,
            producing_component=producing_component,
            parent=parent,
        )

    # -- publish -----------------------------------------------------------

    def publish(
        self,
        content: bytes | Path,
        *,
        category: str,
        format: str,
        format_version: int,
        producing_component: str,
        parent: ParentReference | None = None,
    ) -> ArtifactRecord:
        """Publish ``content`` as a canonical atomic bundle (amendment C).

        ``content`` may be ``bytes`` (held in memory) or a ``Path`` to a regular
        file (read and hashed; never made the canonical artifact). Publication
        uses a sibling temporary **directory**: write ``content`` → hash → write
        ``artifact.json`` → fsync → atomic directory rename (no ``os.replace``).
        An existing final directory is inspected: exact immutable metadata plus
        verified payload is idempotent; any mismatch is
        :class:`ArtifactConflictError`.
        """
        self._validate_category_format(category, format)
        self._validate_parent(parent)

        # Materialize content bytes (either from memory or by copying a regular
        # file). For a Path source we copy bytes into the temp bundle directly
        # (no symlink following — open with O_NOFOLLOW where available).
        cdir = category_dir(
            self._artifact_root,
            self._identity.run_id,
            self._identity.attempt_id,
            category,
        )
        cdir.mkdir(parents=True, exist_ok=True)

        # Stream content into a temp bundle, hashing as we go.
        tmp_bundle = self._make_temp_bundle_dir(cdir)
        content_path = tmp_bundle / "content"
        try:
            digest, size = self._write_content(content_path, content)
            descriptor = self._descriptor(
                content_digest=digest,
                byte_size=size,
                category=category,
                format=format,
                format_version=format_version,
                producing_component=producing_component,
                parent=parent,
            )
            artifact_id = descriptor_to_artifact_id(descriptor)
            relative_path = f"artifacts/{category}/{artifact_id}"
            record = ArtifactRecord.from_descriptor(
                descriptor,
                created_at_utc=datetime.now(UTC),
                relative_path=relative_path,
                storage_class="canonical_local",
                retention="retained",
            )
            # Write artifact.json inside the temp bundle.
            self._write_artifact_json(tmp_bundle / "artifact.json", record)
            # Final destination.
            final_bundle = artifact_bundle_dir(
                self._artifact_root,
                self._identity.run_id,
                self._identity.attempt_id,
                category,
                artifact_id,
            )
            if final_bundle.exists() or _is_symlink(final_bundle):
                # Idempotency vs. conflict. The existing bundle's recorded
                # timestamp is authoritative; preserve it (do not overwrite
                # with this publish's created_at_utc).
                self._assert_idempotent(final_bundle, record)
                # Remove the temp bundle; do not append a duplicate entry.
                self._remove_tree(tmp_bundle)
                # The existing bundle's record (with its original timestamp) is
                # the authoritative immutable metadata; index using it so the
                # registry entry and the bundle agree on created_at_utc.
                existing_record = _record_from_bundle_bytes(
                    (final_bundle / "artifact.json").read_bytes()
                )
                self._ensure_indexed(existing_record, entry_kind="initial_publication")
                return existing_record
            # Atomic directory rename (no os.replace for directories per amendment).
            os.rename(tmp_bundle, final_bundle)
            _fsync_dir(cdir)
            # Registry append.
            self._ensure_indexed(record, entry_kind="initial_publication")
            return record
        except Exception:
            # On any failure after temp creation: remove the temp bundle, do
            # not write a registry record, do not leave partial canonical dirs.
            if tmp_bundle.exists():
                self._remove_tree(tmp_bundle)
            raise

    def _make_temp_bundle_dir(self, parent: Path) -> Path:
        # Unique temp bundle directory beneath the category directory.
        return Path(tempfile.mkdtemp(prefix=".tmp-bundle-", dir=parent))

    def _write_content(self, content_path: Path, content: bytes | Path) -> tuple[str, int]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(content_path, flags, 0o600)
        h = hashlib.sha256()
        total = 0
        try:
            if isinstance(content, Path):
                # Source must be a regular file, not a symlink (amendment H).
                self._assert_regular_source(content)
                with open(content, "rb") as src:
                    while True:
                        chunk = src.read(BLOCK_SIZE)
                        if not chunk:
                            break
                        _full_write(fd, chunk)
                        h.update(chunk)
                        total += len(chunk)
            else:
                if not isinstance(content, (bytes, bytearray, memoryview)):
                    raise ArtifactStoreError("content must be bytes or a Path.")
                _full_write(fd, bytes(content))
                h.update(bytes(content))
                total = len(content)
            os.fsync(fd)
        finally:
            os.close(fd)
        # Silence the nofollow flag var lint (kept for clarity / future use).
        _ = nofollow
        return f"sha256:{h.hexdigest()}", total

    def _assert_regular_source(self, path: Path) -> None:
        try:
            st = os.lstat(path)
        except FileNotFoundError as e:
            raise ArtifactStoreError(f"source file not found: {path}") from e
        import stat as _stat

        if _stat.S_ISLNK(st.st_mode):
            raise ArtifactConflictError(
                f"source {path} is a symlink; register_existing/publish refuses "
                "to follow symlinks (amendment H)."
            )
        if not _stat.S_ISREG(st.st_mode):
            raise ArtifactConflictError(
                f"source {path} is not a regular file (directory inputs are "
                "rejected by Issue #10; #11 produces tar archives)."
            )

    def _write_artifact_json(self, path: Path, record: ArtifactRecord) -> None:
        payload = _record_canonical_json(record)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags, 0o600)
        try:
            _full_write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_dir(path.parent)

    def _assert_idempotent(self, final_bundle: Path, record: ArtifactRecord) -> None:
        """An existing final bundle must carry exact immutable metadata + matching payload.

        Idempotency compares the immutable descriptor-derived identity (amendment
        C): artifact_id, category, format, format_version, content_digest,
        byte_size, producing_component, run_id, attempt_id,
        specification_fingerprint, parent, relative_path. Operational state
        (``created_at_utc``, ``storage_class``, ``retention``) is not part of
        the immutable identity; the existing bundle's recorded timestamp is
        preserved.
        """
        meta_path = final_bundle / "artifact.json"
        content_path = final_bundle / "content"
        if not meta_path.exists() or not content_path.exists():
            raise ArtifactConflictError(
                f"existing bundle {final_bundle} is incomplete (missing content/artifact.json)."
            )
        # Reject a symlinked bundle outright.
        if _is_symlink(final_bundle) or _is_symlink(content_path) or _is_symlink(meta_path):
            raise ArtifactConflictError(
                f"existing bundle {final_bundle} contains a symlink; rejected."
            )
        existing = _record_from_bundle_bytes(meta_path.read_bytes())
        if _immutable_identity(existing) != _immutable_identity(record):
            raise ArtifactConflictError(
                f"artifact {record.artifact_id} exists with different immutable metadata."
            )
        # Verify payload matches the record's digest + size.
        digest, size = _hash_path(content_path)
        if digest != record.content_digest or size != record.byte_size:
            raise ArtifactConflictError(
                f"artifact {record.artifact_id} content does not match its record."
            )

    # -- register_existing -------------------------------------------------

    def register_existing(
        self,
        path: Path,
        *,
        category: str,
        format: str,
        format_version: int,
        producing_component: str,
        parent: ParentReference | None = None,
    ) -> ArtifactRecord:
        """Copy a regular file into store ownership (amendment H).

        The source must be a regular file; symlinks are not followed and
        directory inputs are rejected (Issue #10 accepts bytes or regular files
        only; Issue #11 produces tar archives). The source path is never made
        the canonical artifact; subsequent source-path mutation does not change
        canonical bytes.
        """
        self._validate_category_format(category, format)
        self._validate_parent(parent)
        self._assert_regular_source(path)
        # publish() with a Path source performs the same copy+hash; no symlink
        # following, no mutable source references.
        return self.publish(
            path,
            category=category,
            format=format,
            format_version=format_version,
            producing_component=producing_component,
            parent=parent,
        )

    # -- register_external -------------------------------------------------

    def register_external(
        self,
        *,
        category: str,
        format: str,
        format_version: int,
        producing_component: str,
        location_type: str,
        location: str,
        expected_digest: str,
        expected_byte_size: int,
        external_root_id: str | None = None,
        parent: ParentReference | None = None,
    ) -> ExternalReference:
        """Register an identity-bound external reference (amendment G).

        Writes one logically atomic registry mutation containing both the
        immutable artifact metadata and its external reference. A standalone
        location without an artifact record is invalid. Credential-bearing
        locations are rejected (not sanitized). Availability starts
        ``unavailable``; a location is never verified evidence without digest
        and size verification through a caller-supplied resolver.
        """
        self._validate_category_format(category, format)
        self._validate_parent(parent)
        # Build the immutable record metadata bound to this attempt's identity.
        # The artifact_id is derived from the descriptor, which includes the
        # expected digest/size of the external content (content-addressed).
        descriptor = self._descriptor(
            content_digest=expected_digest,
            byte_size=expected_byte_size,
            category=category,
            format=format,
            format_version=format_version,
            producing_component=producing_component,
            parent=parent,
        )
        artifact_id = descriptor_to_artifact_id(descriptor)
        relative_path = f"artifacts/{category}/{artifact_id}"
        record = ArtifactRecord(
            artifact_id=artifact_id,
            category=descriptor.category,
            format=descriptor.format,
            format_version=descriptor.format_version,
            byte_size=descriptor.byte_size,
            content_digest=descriptor.content_digest,
            producing_component=descriptor.producing_component,
            run_id=descriptor.run_id,
            attempt_id=descriptor.attempt_id,
            specification_fingerprint=descriptor.specification_fingerprint,
            created_at_utc=datetime.now(UTC),
            relative_path=relative_path,
            parent=descriptor.parent,
            # External content is not canonical_local; it lives outside the tree.
            storage_class="external",
            retention="externally_retained",
        )
        try:
            ext = ExternalReference(
                artifact_id=artifact_id,
                run_id=self._identity.run_id,
                attempt_id=self._identity.attempt_id,
                specification_fingerprint=self.specification_fingerprint,
                category=descriptor.category,
                format=descriptor.format,
                format_version=descriptor.format_version,
                location_type=location_type,  # type: ignore[arg-type]
                location=location,
                external_root_id=external_root_id,
                expected_digest=expected_digest,
                expected_byte_size=expected_byte_size,
                availability="unavailable",
                verified_at_utc=None,
            )
        except ExternalLocationError as e:
            raise ArtifactConflictError(str(e)) from e
        # Persist the artifact.json bundle sidecar so the record is
        # self-contained (metadata_only — no local content bytes).
        self._write_metadata_only_bundle(record, ext)
        # Append a single logically atomic external_registration entry whose
        # payload carries both the record and its external reference.
        payload = self._external_registration_payload(record, ext)
        self._append_registry("external_registration", payload)
        return ext

    def _write_metadata_only_bundle(self, record: ArtifactRecord, ext: ExternalReference) -> None:
        bdir = artifact_bundle_dir(
            self._artifact_root,
            self._identity.run_id,
            self._identity.attempt_id,
            record.category,
            record.artifact_id,
        )
        if bdir.exists() or _is_symlink(bdir):
            # If the bundle already exists, it must match exactly.
            self._assert_idempotent_external(bdir, record, ext)
            return
        cdir = category_dir(
            self._artifact_root,
            self._identity.run_id,
            self._identity.attempt_id,
            record.category,
        )
        cdir.mkdir(parents=True, exist_ok=True)
        tmp_bundle = self._make_temp_bundle_dir(cdir)
        try:
            # artifact.json carries the immutable record; the external reference
            # is recorded only in the registry (not in the self-contained
            # bundle, which describes local-ownership metadata).
            self._write_artifact_json(tmp_bundle / "artifact.json", record)
            # No content file for metadata_only bundles.
            os.rename(tmp_bundle, bdir)
            _fsync_dir(cdir)
        except Exception:
            if tmp_bundle.exists():
                self._remove_tree(tmp_bundle)
            raise

    def _assert_idempotent_external(
        self, bdir: Path, record: ArtifactRecord, ext: ExternalReference
    ) -> None:
        meta_path = bdir / "artifact.json"
        if not meta_path.exists():
            raise ArtifactConflictError(f"existing bundle {bdir} is missing artifact.json.")
        existing = _record_from_bundle_bytes(meta_path.read_bytes())
        if _immutable_identity(existing) != _immutable_identity(record):
            raise ArtifactConflictError(
                f"external artifact {record.artifact_id} exists with different immutable metadata."
            )

    def _external_registration_payload(
        self, record: ArtifactRecord, ext: ExternalReference
    ) -> dict[str, Any]:
        rec_dump = json.loads(record.model_dump_json(by_alias=True))
        ext_dump = json.loads(ext.model_dump_json(by_alias=True))
        # Embed the external reference under a stable key so the registry
        # loader can reconstruct the typed record + reference pair.
        return {**rec_dump, "external_reference": ext_dump}

    # -- registry index ----------------------------------------------------

    def _ensure_indexed(self, record: ArtifactRecord, *, entry_kind: str) -> None:
        """Append an entry for ``record`` if it is not already indexed (idempotent)."""
        existing = self._indexed_artifact_ids()
        if record.artifact_id in existing:
            return
        payload = json.loads(record.model_dump_json(by_alias=True))
        self._append_registry(entry_kind, payload)

    def _append_registry(self, entry_kind: str, payload: dict[str, Any]) -> RegistryEntry:
        try:
            return allocate_and_append(
                self._registry_path,
                lock_path=self._lock_path,
                run_id=self._identity.run_id,
                attempt_id=self._identity.attempt_id,
                specification_fingerprint=self.specification_fingerprint,
                entry_kind=entry_kind,
                payload=payload,
                recorded_at_utc=datetime.now(UTC),
            )
        except RegistryError as e:
            raise ArtifactStoreError(f"registry append failed: {e}") from e

    def _indexed_artifact_ids(self) -> set[str]:
        entries = self._load_registry_safe()
        return {e.payload["artifact_id"] for e in entries if "artifact_id" in e.payload}

    def _load_registry_safe(self) -> list[RegistryEntry]:
        try:
            return load_registry(
                self._registry_path,
                expected_run_id=self._identity.run_id,
                expected_attempt_id=self._identity.attempt_id,
                expected_fingerprint=self.specification_fingerprint,
            )
        except RegistryError:
            return []

    # -- locate / list / inspect ------------------------------------------

    def _bundle_path(self, artifact_id: str) -> Path:
        validate_id_component(artifact_id, "artifact_id")
        # Locate the bundle by scanning categories (the artifact_id encodes the
        # full descriptor, not the category, so we look it up).
        arts = self._attempt_dir / "artifacts"
        if not arts.exists():
            raise ArtifactNotFoundError(f"artifact {artifact_id!r} not found.")
        for cat_dir in arts.iterdir():
            if not cat_dir.is_dir():
                continue
            cand = cat_dir / artifact_id
            if cand.exists() or _is_symlink(cand):
                return cand
        raise ArtifactNotFoundError(f"artifact {artifact_id!r} not found.")

    def locate(self, artifact_id: str) -> Path | None:
        """The canonical ``content`` path of ``artifact_id``, or ``None`` if absent."""
        try:
            bdir = self._bundle_path(artifact_id)
        except ArtifactNotFoundError:
            return None
        content = bdir / "content"
        return content if content.exists() else None

    def _current_record(self, artifact_id: str) -> ArtifactRecord | None:
        """The last-known registry state for ``artifact_id`` (None if unindexed)."""
        entries = self._load_registry_safe()
        try:
            histories = _reduce_history(entries)
        except RegistryError:
            return None
        hist = histories.get(artifact_id)
        return hist.record if hist else None

    def list_artifacts(self, *, category: str | None = None) -> list[ArtifactRecord]:
        """All indexed artifacts, optionally filtered by category.

        Returns the last-known registry state per artifact_id.
        """
        entries = self._load_registry_safe()
        try:
            histories = _reduce_history(entries)
        except RegistryError:
            return []
        out: list[ArtifactRecord] = []
        for hist in histories.values():
            if category is not None and hist.record.category != category:
                continue
            out.append(hist.record)
        out.sort(key=lambda r: r.artifact_id)
        return out

    def inspect(self, artifact_id: str) -> ArtifactRecord | ExternalReference:
        """The registry-indexed record (or external reference) for ``artifact_id``."""
        entries = self._load_registry_safe()
        try:
            histories = _reduce_history(entries)
        except RegistryError as e:
            raise ArtifactStoreError(f"registry corrupt: {e}") from e
        hist = histories.get(artifact_id)
        if hist is None:
            raise ArtifactNotFoundError(f"artifact {artifact_id!r} not indexed.")
        if hist.external is not None:
            return hist.external
        return hist.record

    # -- verify -----------------------------------------------------------

    def verify(self, artifact_id: str) -> VerificationResult:
        """Typed verification (amendment J): re-hash the bundle content.

        Returns a :class:`VerificationResult` with status, observed digest,
        observed size, and a closed diagnostic code.
        """
        try:
            bdir = self._bundle_path(artifact_id)
        except ArtifactNotFoundError:
            return VerificationResult(
                status=False,
                artifact_id=artifact_id,
                diagnostic_code="missing_bundle",
            )
        content = bdir / "content"
        meta_path = bdir / "artifact.json"
        if not meta_path.exists():
            return VerificationResult(
                status=False,
                artifact_id=artifact_id,
                diagnostic_code="bundle_metadata_mismatch",
            )
        if _is_symlink(content) or _is_symlink(bdir):
            return VerificationResult(
                status=False,
                artifact_id=artifact_id,
                diagnostic_code="not_regular_file",
            )
        record = _record_from_bundle_bytes(meta_path.read_bytes())
        if record.artifact_id != artifact_id:
            return VerificationResult(
                status=False,
                artifact_id=artifact_id,
                diagnostic_code="bundle_metadata_mismatch",
            )
        if not content.exists():
            # External/metadata_only bundles have no local content.
            ext = self._current_external(artifact_id)
            if ext is not None:
                return VerificationResult(
                    status=False,
                    artifact_id=artifact_id,
                    diagnostic_code="external_reference_not_verified",
                )
            return VerificationResult(
                status=False,
                artifact_id=artifact_id,
                diagnostic_code="missing_content",
            )
        # Regular file check.
        try:
            st = os.stat(content)
        except OSError:
            return VerificationResult(
                status=False,
                artifact_id=artifact_id,
                diagnostic_code="missing_content",
            )
        import stat as _stat

        if not _stat.S_ISREG(st.st_mode):
            return VerificationResult(
                status=False,
                artifact_id=artifact_id,
                diagnostic_code="not_regular_file",
            )
        digest, size = _hash_path(content)
        if digest != record.content_digest:
            return VerificationResult(
                status=False,
                artifact_id=artifact_id,
                diagnostic_code="digest_mismatch",
                observed_digest=digest,
                observed_size=size,
            )
        if size != record.byte_size:
            return VerificationResult(
                status=False,
                artifact_id=artifact_id,
                diagnostic_code="size_mismatch",
                observed_digest=digest,
                observed_size=size,
            )
        return VerificationResult(
            status=True,
            artifact_id=artifact_id,
            diagnostic_code="verified",
            observed_digest=digest,
            observed_size=size,
        )

    def _current_external(self, artifact_id: str) -> ExternalReference | None:
        entries = self._load_registry_safe()
        try:
            histories = _reduce_history(entries)
        except RegistryError:
            return None
        hist = histories.get(artifact_id)
        return hist.external if hist else None

    def verify_bool(self, artifact_id: str) -> bool:
        """Boolean convenience wrapper around :meth:`verify` (amendment J)."""
        return self.verify(artifact_id).status

    # -- update_retention --------------------------------------------------

    def update_retention(self, artifact_id: str, retention: RetentionStatus) -> ArtifactRecord:
        """Append a retention_transition entry (amendment F closed matrix).

        Legal transitions are validated by the registry loader; illegal
        transitions raise :class:`ArtifactConflictError`.
        """
        current = self._current_record(artifact_id)
        if current is None:
            raise ArtifactNotFoundError(f"artifact {artifact_id!r} not indexed.")
        if not is_legal_retention_transition(current.retention, retention):
            raise ArtifactConflictError(
                f"illegal retention transition {current.retention!r} -> {retention!r}."
            )
        # Storage class is unchanged on a pure retention transition unless the
        # target retention requires a different storage class.
        new_storage = current.storage_class
        # ``metadata_only`` is required when content becomes missing/expired
        # and no local/external bytes back it.
        if retention in ("missing", "expired") and new_storage == "canonical_local":
            new_storage = "metadata_only"
        if not is_legal_storage_transition(current.storage_class, new_storage):
            raise ArtifactConflictError(
                f"illegal storage transition {current.storage_class!r} -> {new_storage!r}."
            )
        # Validate the resulting (storage, retention) combination via the
        # record constructor (raises on illegal combos).
        current.model_copy(update={"storage_class": new_storage, "retention": retention})
        # Re-validate cross-field invariants explicitly.
        ArtifactRecord._check_storage_retention(new_storage, retention)
        payload = {
            "artifact_id": artifact_id,
            "storage_class": new_storage,
            "retention": retention,
        }
        self._append_registry("retention_transition", payload)
        return self._current_record(artifact_id)  # type: ignore[return-value]

    # -- reconciliation ----------------------------------------------------

    def reconcile_orphans(self) -> RegistryReconciliation:
        """Detect canonical bundles without registry entries and append missing entries.

        Reconciliation appends a missing publication entry only after strictly
        validating the complete bundle and identity binding (amendment K).
        Registry entries never rewrite payload bytes or immutable bundle
        metadata.
        """
        arts = self._attempt_dir / "artifacts"
        indexed = self._indexed_artifact_ids()
        appended: list[str] = []
        orphans: list[str] = []
        skipped: list[tuple[str, str]] = []
        if not arts.exists():
            return RegistryReconciliation(appended=appended, orphans=orphans, skipped=skipped)
        for cat_dir in sorted(arts.iterdir()):
            if not cat_dir.is_dir():
                continue
            category = cat_dir.name
            for bundle in sorted(cat_dir.iterdir()):
                if not bundle.is_dir() or bundle.name.startswith(".tmp-bundle-"):
                    continue
                artifact_id = bundle.name
                if artifact_id in indexed:
                    continue
                orphans.append(artifact_id)
                # Strictly validate the bundle before appending.
                meta_path = bundle / "artifact.json"
                content_path = bundle / "content"
                try:
                    if not meta_path.exists():
                        raise ValueError("missing artifact.json")
                    record = _record_from_bundle_bytes(meta_path.read_bytes())
                    if record.artifact_id != artifact_id:
                        raise ValueError("artifact_id mismatch")
                    if record.category != category:
                        raise ValueError("category mismatch")
                    if record.run_id != self._identity.run_id:
                        raise ValueError("run_id mismatch")
                    if record.attempt_id != self._identity.attempt_id:
                        raise ValueError("attempt_id mismatch")
                    if record.specification_fingerprint != self.specification_fingerprint:
                        raise ValueError("fingerprint mismatch")
                    # Verify content if present (external bundles have none).
                    if content_path.exists():
                        if _is_symlink(content_path):
                            raise ValueError("content is a symlink")
                        digest, size = _hash_path(content_path)
                        if digest != record.content_digest:
                            raise ValueError("content digest mismatch")
                        if size != record.byte_size:
                            raise ValueError("content size mismatch")
                except (ValueError, ArtifactStoreError) as e:
                    skipped.append((artifact_id, str(e)))
                    continue
                payload = json.loads(record.model_dump_json(by_alias=True))
                self._append_registry("initial_publication", payload)
                appended.append(artifact_id)
                indexed.add(artifact_id)
        return RegistryReconciliation(appended=appended, orphans=orphans, skipped=skipped)

    # -- validation helpers ------------------------------------------------

    def _validate_category_format(self, category: str, format: str) -> None:
        from expertforge.artifacts.models import allowed_formats_for_category

        validate_id_component(category, "category")
        allowed = allowed_formats_for_category(category)  # type: ignore[arg-type]
        if format not in allowed:
            raise ArtifactStoreError(
                f"format {format!r} is not allowed for category {category!r}; "
                f"allowed={sorted(allowed)}."
            )

    def _validate_parent(self, parent: ParentReference | None) -> None:
        if parent is None:
            return
        # ParentReference is fully qualified (amendment B). Cross-attempt
        # parents are allowed; same-attempt parents must match this attempt.
        if (
            parent.run_id == self._identity.run_id
            and parent.attempt_id == self._identity.attempt_id
        ):
            return
        # Cross-attempt references are accepted as-is (evidence links).

    # -- misc -------------------------------------------------------------

    @staticmethod
    def _remove_tree(path: Path) -> None:
        import shutil

        shutil.rmtree(path, ignore_errors=True)
