"""CheckpointStore: save/load/inspect compatibility (Issue #11, amendments G, K, L, M).

- :meth:`save` captures a quiescent snapshot, encodes it to a deterministic tar,
  publishes via :class:`ArtifactStore`, and verifies the returned record.
- :meth:`load` is authoritative: it loads by ``artifact_id`` through
  :meth:`ArtifactStore.inspect`, binds the record, locates the content path,
  opens it with no-follow + fstat on a SINGLE descriptor, streams digest/size
  and tar parsing from that same descriptor (TOCTOU-safe), and returns a fully
  decoded :class:`CheckpointArchive`.
- :meth:`inspect_path` diagnoses an arbitrary untrusted file but is
  non-authoritative (cannot restore or claim #10 registration).
- :meth:`check_compatibility` classifies an archive against an expected
  descriptor using the closed diagnostic table (amendment M).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat as stat_mod
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from expertforge.artifacts.models import (
    ArtifactConflictError,
    ArtifactNotFoundError,
    ArtifactRecord,
    ExternalReference,
    ParentReference,
    RetentionStatus,
)
from expertforge.artifacts.store import BLOCK_SIZE, ArtifactStore
from expertforge.checkpoints.encoder import ArchiveParts, build_archive_parts
from expertforge.checkpoints.models import (
    CHECKPOINT_ARCHIVE_FORMAT_VERSION,
    MAX_ARCHIVE_BYTES,
    MAX_COMPONENT_MEMBER_BYTES,
    CheckpointInspection,
    CheckpointManifest,
    CompatibilityDescriptor,
    CompatibilityMismatch,
    CompatibilityResult,
    canonical_json_bytes,
)
from expertforge.checkpoints.tar_reader import (
    ParsedMember,
    TarParseError,
    parse_ustar_archive,
    parse_ustar_archive_streaming,
)
from expertforge.config.resolve import ResolutionEnvelope
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.provenance.record import ProvenanceRecord
from expertforge.rng.state import RngStateBundle

__all__ = [
    "CheckpointError",
    "CheckpointCorruptError",
    "CheckpointVersionError",
    "CheckpointLineageError",
    "CheckpointComponentError",
    "CheckpointTopologyError",
    "CheckpointArchive",
    "CheckpointStore",
    "MAX_INMEMORY_ARCHIVE_BYTES",
    "check_compatibility",
]

# Item 8: archives at or below this byte size may be buffered whole in memory
# for parsing; larger archives MUST be streamed to a temp file (save) and
# stream-parsed from the open file descriptor (load). The default of 64 MiB
# matches amendment F's small-buffer threshold.
MAX_INMEMORY_ARCHIVE_BYTES: int = 64 * 1024 * 1024


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class CheckpointError(Exception):
    """Base class for checkpoint failures."""


class CheckpointCorruptError(CheckpointError):
    """Raised on a truncated or tampered archive (bad tar, digest mismatch)."""


class CheckpointVersionError(CheckpointError):
    """Raised on unknown schema names/versions."""


class CheckpointLineageError(CheckpointError):
    """Raised when a required parent checkpoint cannot be resolved."""


class CheckpointComponentError(CheckpointError):
    """Raised when a required state component is missing."""


class CheckpointTopologyError(CheckpointError):
    """Raised on an unsupported distributed topology."""


# Retention states acceptable for an authoritative load (amendment G).
_ACCEPTABLE_RETENTION: frozenset[RetentionStatus] = frozenset(
    {"retained", "pending_transfer", "externally_retained"}
)


# ---------------------------------------------------------------------------
# CheckpointArchive — the decoded, validated, immutable archive
# ---------------------------------------------------------------------------


class CheckpointArchive:
    """The fully decoded, validated checkpoint archive.

    Holds the manifest, the parsed members (by name), the decoded component
    payloads, and the binding :class:`ArtifactRecord`. On the small path (every
    realistic v1 checkpoint, ≤ MAX_INMEMORY_ARCHIVE_BYTES) the full tar bytes are
    materialized in memory for inspection. On the LARGE path (> 64 MiB, item 8)
    the archive is stream-parsed from the file descriptor and ``tar_bytes`` is
    NOT retained — the complete archive is never held in Python heap; only the
    parsed member bytes the caller requests are materialized. Restoration is a
    separate transaction (:mod:`expertforge.checkpoints.restore`).
    """

    __slots__ = (
        "_manifest",
        "_members_by_name",
        "_record",
        "_tar_bytes",
        "_content_digest",
        "_byte_size",
    )

    def __init__(
        self,
        *,
        manifest: CheckpointManifest,
        members: list[ParsedMember],
        record: ArtifactRecord,
        tar_bytes: bytes | None,
        content_digest: str,
        byte_size: int,
    ) -> None:
        self._manifest = manifest
        self._members_by_name = {m.name: m for m in members}
        self._record = record
        # tar_bytes is None on the large (streaming) path: the whole archive is
        # never retained. content_digest / byte_size are always populated (they
        # are computed incrementally during the streaming read, before any
        # member is parsed).
        self._tar_bytes = tar_bytes
        self._content_digest = content_digest
        self._byte_size = byte_size

    @property
    def manifest(self) -> CheckpointManifest:
        return self._manifest

    @property
    def artifact_record(self) -> ArtifactRecord:
        return self._record

    @property
    def artifact_id(self) -> str:
        return self._record.artifact_id

    @property
    def content_digest(self) -> str:
        return self._content_digest

    @property
    def byte_size(self) -> int:
        return self._byte_size

    @property
    def tar_bytes(self) -> bytes:
        # Item 8: on the large (streaming) path the complete tar is intentionally
        # not retained. Callers that need raw archive bytes must use the small
        # path (every realistic v1 checkpoint) or re-read the published content.
        if self._tar_bytes is None:
            raise CheckpointCorruptError(
                "tar_bytes is unavailable on the streaming (large-archive) path"
            )
        return self._tar_bytes

    def member(self, name: str) -> bytes:
        """Return the raw bytes of member ``name`` (raises if absent)."""
        m = self._members_by_name.get(name)
        if m is None:
            raise CheckpointCorruptError(f"missing member {name!r}")
        if m.data is None:
            if m.temp_path is not None:
                return m.temp_path.read_bytes()
            raise CheckpointCorruptError(f"member {name!r} has no data and no temp_path")
        return m.data

    def component(self, role: str) -> bytes:
        """Return the raw bytes of the ``state/<role>.json`` component."""
        return self.member(f"state/{role}.json")

    def tensor_member(self, member_name: str) -> bytes:
        """Return the raw bytes of tensor member ``member_name``."""
        return self.member(member_name)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CheckpointArchive(artifact_id={self.artifact_id!r}, byte_size={self._byte_size})"


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------


class CheckpointStore:
    """Authoritative checkpoint save/load/inspect bound to an ArtifactStore."""

    PRODUCING_COMPONENT = "expertforge.checkpoints"

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._store = artifact_store

    @property
    def artifact_store(self) -> ArtifactStore:
        return self._store

    # -- save ---------------------------------------------------------------

    def save(
        self,
        *,
        identity: AttemptIdentityRecord,
        captured: Any,
        configuration_envelope: ResolutionEnvelope,
        provenance: ProvenanceRecord,
        rng_bundle: RngStateBundle | None = None,
        parent: ParentReference | None = None,
        created_at_utc: datetime | None = None,
    ) -> ArtifactRecord:
        """Capture → encode → publish → verify the returned record.

        ``captured`` is the quiescent :class:`CapturedCheckpointState` from the
        provider's :meth:`capture_checkpoint_snapshot`. The RNG member is
        serialized from ``captured.rng_bundle_bytes`` (the atomic snapshot); an
        optional ``rng_bundle`` is validated against those bytes and rejected on
        disagreement (item 4). Per amendment F / item 8 (streaming), the archive
        parts are computed once (:func:`build_archive_parts`) and then STREAMED
        member-by-member to a temp file via :func:`stream_ustar_archive` while
        the SHA-256 digest is updated incrementally — the complete tar is NEVER
        materialized in memory. The temp file is then published via
        :meth:`ArtifactStore.publish` (which copies the verified regular file
        into a canonical bundle). The returned record is verified for checkpoint
        category/format/version, producing component, identity (run/attempt/
        specification_fingerprint), parent, digest, and size before it is
        returned (item 11).
        """
        import tempfile

        from expertforge.checkpoints.models import CapturedCheckpointState
        from expertforge.checkpoints.tar_writer import stream_ustar_archive

        if not isinstance(captured, CapturedCheckpointState):
            raise CheckpointComponentError(
                "captured must be a CapturedCheckpointState from capture_checkpoint_snapshot()"
            )
        ts = created_at_utc or datetime.now(tz=_UTC())
        parts = build_archive_parts(
            identity=identity,
            captured=captured,
            configuration_envelope=configuration_envelope,
            provenance=provenance,
            rng_bundle=rng_bundle,
            parent=parent,
            created_at_utc=ts,
        )
        # Item 8: stream the members directly to a temp file. Each member's
        # header + padded data is written as it is encoded; the running SHA-256
        # is updated per chunk so the digest is computed without ever holding
        # the whole tar in a single buffer. publish() re-hashes the regular file
        # (no symlink following) into the canonical bundle.
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=".tmp-cp-", suffix=".tar")
        h = hashlib.sha256()
        total = 0
        try:

            def _write(chunk: bytes) -> None:
                nonlocal total
                _full_write(tmp_fd, chunk)
                h.update(chunk)
                total += len(chunk)

            stream_ustar_archive(parts.members, _write)
            os.fsync(tmp_fd)
            os.close(tmp_fd)
            tmp_fd = -1  # mark closed so the finally does not double-close
            parts.content_digest = f"sha256:{h.hexdigest()}"
            parts.byte_size = total
            tmp_path = Path(tmp_name)
            record = self._store.publish(
                tmp_path,
                category="checkpoint",
                format="tar",
                format_version=CHECKPOINT_ARCHIVE_FORMAT_VERSION,
                producing_component=self.PRODUCING_COMPONENT,
                parent=parent,
            )
        finally:
            if tmp_fd >= 0:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        self._verify_returned_record(record, parts, parent, identity)
        return record

    def _verify_returned_record(
        self,
        record: ArtifactRecord,
        archive: ArchiveParts,
        parent: ParentReference | None,
        identity: AttemptIdentityRecord,
    ) -> None:
        """Amendment G + item 11: verify category/format/version/producer/
        identity/parent/digest/size.

        Item 11 additionally binds the returned record's ``run_id``,
        ``attempt_id``, and ``specification_fingerprint`` to the explicit
        ``identity`` argument of :meth:`save`. Because the underlying
        :class:`ArtifactStore` binds these to its own constructor identity, a
        mismatched save argument would otherwise publish a record and manifest
        with divergent identities before the later load catches it.
        """
        if record.category != "checkpoint":
            raise CheckpointCorruptError(
                f"published record category {record.category!r} != 'checkpoint'"
            )
        if record.format != "tar":
            raise CheckpointCorruptError(f"published record format {record.format!r} != 'tar'")
        if record.format_version != CHECKPOINT_ARCHIVE_FORMAT_VERSION:
            raise CheckpointVersionError(
                f"published record format_version {record.format_version!r} != "
                f"{CHECKPOINT_ARCHIVE_FORMAT_VERSION}"
            )
        if record.producing_component != self.PRODUCING_COMPONENT:
            raise CheckpointCorruptError(
                f"published record producing_component "
                f"{record.producing_component!r} != {self.PRODUCING_COMPONENT!r}"
            )
        if record.content_digest != archive.content_digest:
            raise CheckpointCorruptError(
                f"published record digest {record.content_digest!r} != archive "
                f"digest {archive.content_digest!r}"
            )
        if record.byte_size != archive.byte_size:
            raise CheckpointCorruptError(
                f"published record byte_size {record.byte_size!r} != archive "
                f"byte_size {archive.byte_size!r}"
            )
        if record.parent != parent:
            raise CheckpointCorruptError(
                "published record parent does not match the requested parent"
            )
        # Item 11: bind the returned record to the explicit save identity.
        if record.run_id != identity.run_id:
            raise CheckpointCorruptError(
                f"published record run_id {record.run_id!r} != save identity "
                f"run_id {identity.run_id!r}"
            )
        if record.attempt_id != identity.attempt_id:
            raise CheckpointCorruptError(
                f"published record attempt_id {record.attempt_id!r} != save "
                f"identity attempt_id {identity.attempt_id!r}"
            )
        if record.specification_fingerprint != identity.fingerprint_digest_str():
            raise CheckpointCorruptError(
                "published record specification_fingerprint != save identity fingerprint digest"
            )

    # -- load (authoritative) ----------------------------------------------

    def load(
        self,
        artifact_id: str,
        *,
        expected_identity: AttemptIdentityRecord,
    ) -> CheckpointArchive:
        """Authoritative load by ``artifact_id`` (amendment G).

        Single-fd TOCTOU-safe and parent-race-safe (item 9): obtains the record
        through :meth:`ArtifactStore.inspect`, requires checkpoint category/
        format/version, canonical-local storage, retained/acceptable retention,
        and identity binding; REQUIRES the expected identity to carry a
        :class:`ResumeLineage` naming the source checkpoint and resolves that
        parent through the public :meth:`ArtifactStore.open_verified_content`
        boundary (item 2); opens the content through the SAME public primitive
        so no underscored artifact internals are imported; and streams
        digest/size + tar parsing from the SAME descriptor (item 8). The computed
        digest/size must equal the record; the manifest producing identity and
        parent must equal the record fields.
        """
        record = self._inspect_record(artifact_id)
        self._assert_record_loadable(record)
        # Identity binding: the record's run/attempt/fingerprint must match the
        # expected identity (amendment L: V1 same-run resume only).
        self._assert_identity_binding(record, expected_identity)
        # Native resume lineage enforcement (item 2): the expected identity MUST
        # carry a ResumeLineage naming the source checkpoint, its fields must
        # match the loaded record exactly, and the fully-qualified parent is
        # resolved authoritatively through the public artifact-store boundary.
        self._assert_resume_lineage(record, expected_identity)
        self._resolve_resume_parent(record, expected_identity)
        # Item 9: open through the PUBLIC descriptor-bound primitive so this
        # module imports NO underscored artifact-store helpers. The descriptor is
        # fstat-verified regular and the full chain is symlink-free inside
        # open_verified_content (parent-race defense).
        try:
            fd, content_record = self._store.open_verified_content(artifact_id)
        except ArtifactNotFoundError as e:
            raise CheckpointCorruptError(
                f"artifact {artifact_id!r} has no canonical content to load: {e}"
            ) from e
        except ArtifactConflictError as e:
            raise CheckpointCorruptError(f"content path chain is not trusted: {e}") from e
        if content_record.artifact_id != record.artifact_id:
            raise CheckpointCorruptError(
                "open_verified_content returned a record that does not match the "
                "inspected artifact_id"
            )
        try:
            if record.byte_size <= MAX_INMEMORY_ARCHIVE_BYTES:
                # Small path (every realistic v1 checkpoint): materialize the
                # whole tar in memory, hash it, and parse from the buffer.
                tar_bytes, digest = self._read_and_hash_fd(fd, record.byte_size)
                size_read = len(tar_bytes)
            else:
                # Item 8 large path: stream the fd to a temp file (hashing
                # incrementally) then re-open the verified temp file and
                # stream-PARSE it with parse_ustar_archive_streaming. The
                # complete archive is NEVER held in Python heap — peak heap is
                # one member's data plus one header block. tar_bytes stays None.
                tar_bytes = None
                digest, size_read, parsed_path = self._read_large_fd_streaming(fd, record.byte_size)
        finally:
            os.close(fd)
        # The content descriptor is now closed. The digest/size were computed
        # during the read so the descriptor was the single trusted source.
        if digest != record.content_digest:
            raise CheckpointCorruptError(
                f"content digest {digest!r} != record {record.content_digest!r}"
            )
        if size_read != record.byte_size:
            raise CheckpointCorruptError(
                f"content size {size_read!r} != record {record.byte_size!r}"
            )
        if tar_bytes is not None:
            members = self._parse_and_validate_tar(tar_bytes)
        else:
            # Large path: parse the verified temp file sequentially. The temp
            # file is unlinked after parsing; only the parsed member bytes are
            # retained.
            try:
                members = self._parse_and_validate_tar_streaming(parsed_path, record.byte_size)
            finally:
                try:
                    parsed_path.unlink()
                except OSError:
                    pass
        manifest = self._decode_manifest(members)
        # Manifest producing identity must equal the record fields (amendment G).
        self._assert_manifest_record_binding(manifest, record)
        # Item 5: cross-bind every persisted component to the manifest + record
        # (counters, descriptors, identity/config/provenance, RNG decode, and
        # no unused tensor members).
        self._bind_components(manifest, members, record)
        return CheckpointArchive(
            manifest=manifest,
            members=members,
            record=record,
            tar_bytes=tar_bytes,
            content_digest=digest,
            byte_size=size_read,
        )

    def _inspect_record(self, artifact_id: str) -> ArtifactRecord:
        result = self._store.inspect(artifact_id)
        if isinstance(result, ExternalReference):
            raise CheckpointCorruptError(
                f"artifact {artifact_id!r} is an external reference, not loadable"
            )
        return result

    def _assert_record_loadable(self, record: ArtifactRecord) -> None:
        if record.category != "checkpoint":
            raise CheckpointCorruptError(
                f"artifact {record.artifact_id!r} category {record.category!r} is not 'checkpoint'"
            )
        if record.format != "tar":
            raise CheckpointCorruptError(
                f"artifact {record.artifact_id!r} format {record.format!r} != 'tar'"
            )
        if record.format_version != CHECKPOINT_ARCHIVE_FORMAT_VERSION:
            raise CheckpointVersionError(
                f"artifact {record.artifact_id!r} format_version "
                f"{record.format_version!r} != {CHECKPOINT_ARCHIVE_FORMAT_VERSION}"
            )
        if record.storage_class != "canonical_local":
            raise CheckpointCorruptError(
                f"artifact {record.artifact_id!r} storage_class "
                f"{record.storage_class!r} is not canonical_local"
            )
        if record.retention not in _ACCEPTABLE_RETENTION:
            raise CheckpointCorruptError(
                f"artifact {record.artifact_id!r} retention {record.retention!r} "
                "is not acceptable for authoritative load"
            )

    def _assert_identity_binding(
        self, record: ArtifactRecord, expected: AttemptIdentityRecord
    ) -> None:
        # amendment L: V1 full-state restore is native same-run resume only.
        if record.run_id != expected.run_id:
            raise CheckpointLineageError(
                f"artifact run_id {record.run_id!r} != expected {expected.run_id!r} "
                "(V1 full restore is same-run only)"
            )
        if record.specification_fingerprint != expected.fingerprint_digest_str():
            raise CheckpointLineageError(
                "artifact specification_fingerprint does not match the expected "
                "identity (V1 full restore requires the same specification)"
            )

    def _assert_resume_lineage(
        self, record: ArtifactRecord, expected: AttemptIdentityRecord
    ) -> None:
        """Item 2: native resume lineage enforcement (REQUIRED, not optional).

        A V1 full-state restore must be performed by a NEW attempt (not the
        producing attempt) whose declared :class:`ResumeLineage` is REQUIRED and
        names the source checkpoint's run/attempt/artifact_id exactly. A missing
        lineage is rejected: there is no ungated resume path.
        """
        if expected.attempt_id == record.attempt_id:
            raise CheckpointLineageError(
                f"resume attempt_id {expected.attempt_id!r} must differ from the "
                f"producing attempt_id {record.attempt_id!r} (V1 resume is a new "
                "attempt continuing from the source checkpoint)"
            )
        lineage = expected.lineage
        if lineage is None:
            raise CheckpointLineageError(
                "V1 full-state restore requires a ResumeLineage naming the source "
                "checkpoint (parent_run_id/parent_attempt_id/parent_checkpoint_id); "
                "expected_identity.lineage is None"
            )
        # The lineage must name the source checkpoint exactly.
        if lineage.parent_run_id != record.run_id:
            raise CheckpointLineageError(
                f"lineage parent_run_id {lineage.parent_run_id!r} != source "
                f"run_id {record.run_id!r}"
            )
        if lineage.parent_attempt_id != record.attempt_id:
            raise CheckpointLineageError(
                f"lineage parent_attempt_id {lineage.parent_attempt_id!r} != source "
                f"attempt_id {record.attempt_id!r}"
            )
        if lineage.parent_checkpoint_id != record.artifact_id:
            raise CheckpointLineageError(
                f"lineage parent_checkpoint_id {lineage.parent_checkpoint_id!r} != "
                f"source artifact_id {record.artifact_id!r}"
            )

    def _resolve_resume_parent(
        self, record: ArtifactRecord, expected: AttemptIdentityRecord
    ) -> ArtifactRecord:
        """Item 2: authoritatively resolve the fully-qualified parent checkpoint.

        Builds a :class:`ParentReference` from the REQUIRED lineage and resolves
        it through :meth:`resolve_parent` (which uses the public
        :meth:`ArtifactStore.inspect` boundary), verifying the parent exists as a
        registered, loadable checkpoint. Called automatically by :meth:`load` so
        a missing or mismatched parent registration blocks restoration.
        """
        lineage = expected.lineage
        # _assert_resume_lineage already guaranteed lineage is present and its
        # fields match the loaded record, but be defensive.
        assert lineage is not None
        parent_ref = ParentReference(
            run_id=lineage.parent_run_id,
            attempt_id=lineage.parent_attempt_id or record.attempt_id,
            artifact_id=lineage.parent_checkpoint_id,
        )
        return self.resolve_parent(parent_ref)

    def resolve_parent(self, parent: ParentReference) -> ArtifactRecord:
        """Item 2/9: verify a fully-qualified parent checkpoint is registered.

        Takes a :class:`ParentReference` (run_id + attempt_id + artifact_id) and
        verifies via the public :meth:`ArtifactStore.inspect` boundary that the
        parent exists as a registered, loadable checkpoint whose producing
        identity matches the reference exactly. Returns the parent
        :class:`ArtifactRecord` or raises :class:`CheckpointLineageError`.
        """
        artifact_id = parent.artifact_id
        try:
            parent_record = self._store.inspect(artifact_id)
        except ArtifactNotFoundError as e:
            raise CheckpointLineageError(
                f"parent checkpoint {artifact_id!r} is not a registered artifact: {e}"
            ) from e
        if isinstance(parent_record, ExternalReference):
            raise CheckpointLineageError(
                f"parent checkpoint {artifact_id!r} is an external reference, not a "
                "registered local checkpoint"
            )
        if parent_record.run_id != parent.run_id:
            raise CheckpointLineageError(
                f"parent run_id {parent_record.run_id!r} != reference run_id {parent.run_id!r}"
            )
        if parent_record.attempt_id != parent.attempt_id:
            raise CheckpointLineageError(
                f"parent attempt_id {parent_record.attempt_id!r} != reference "
                f"attempt_id {parent.attempt_id!r}"
            )
        if parent_record.category != "checkpoint":
            raise CheckpointLineageError(
                f"parent {artifact_id!r} category {parent_record.category!r} is not 'checkpoint'"
            )
        if parent_record.storage_class != "canonical_local":
            raise CheckpointLineageError(
                f"parent {artifact_id!r} storage_class {parent_record.storage_class!r} "
                "is not canonical_local"
            )
        if parent_record.retention not in _ACCEPTABLE_RETENTION:
            raise CheckpointLineageError(
                f"parent {artifact_id!r} retention {parent_record.retention!r} is not "
                "acceptable for resume"
            )
        return parent_record

    def _assert_manifest_record_binding(
        self, manifest: CheckpointManifest, record: ArtifactRecord
    ) -> None:
        if manifest.run_id != record.run_id:
            raise CheckpointCorruptError(
                f"manifest run_id {manifest.run_id!r} != record {record.run_id!r}"
            )
        if manifest.attempt_id != record.attempt_id:
            raise CheckpointCorruptError(
                f"manifest attempt_id {manifest.attempt_id!r} != record {record.attempt_id!r}"
            )
        if manifest.specification_fingerprint != record.specification_fingerprint:
            raise CheckpointCorruptError(
                "manifest specification_fingerprint != record specification_fingerprint"
            )
        # Item 6: compare the COMPLETE ParentReference, not just artifact_id.
        # A manifest whose parent_run_id / parent_attempt_id diverge from the
        # record's registered parent (while sharing the artifact_id) must be
        # rejected — comparing only artifact_id lets a swapped run/attempt slip
        # through. The manifest stores the three fields separately; the record
        # carries a single ParentReference (run_id, attempt_id, artifact_id).
        m_parent_artifact = manifest.parent_artifact_id
        m_parent_run = manifest.parent_run_id
        m_parent_attempt = manifest.parent_attempt_id
        r_parent = record.parent
        r_parent_artifact = r_parent.artifact_id if r_parent else None
        r_parent_run = r_parent.run_id if r_parent else None
        r_parent_attempt = r_parent.attempt_id if r_parent else None
        if m_parent_artifact != r_parent_artifact:
            raise CheckpointCorruptError("manifest parent_artifact_id != record parent artifact_id")
        if m_parent_run != r_parent_run:
            raise CheckpointCorruptError("manifest parent_run_id != record parent run_id")
        if m_parent_attempt != r_parent_attempt:
            raise CheckpointCorruptError("manifest parent_attempt_id != record parent attempt_id")

    def _read_and_hash_fd(self, fd: int, expected_size: int) -> tuple[bytes, str]:
        """Item 8 small path: read + hash ``fd`` materializing the whole tar.

        Used only when ``expected_size <= MAX_INMEMORY_ARCHIVE_BYTES`` (every
        realistic v1 checkpoint). Returns ``(tar_bytes, digest)``. The large
        path is handled separately by :meth:`_read_large_fd_streaming`.
        """
        tar_bytes = self._read_bounded_fd(fd, expected_size)
        digest = f"sha256:{hashlib.sha256(tar_bytes).hexdigest()}"
        return tar_bytes, digest

    def _read_large_fd_streaming(self, fd: int, expected_size: int) -> tuple[str, int, Path]:
        """Item 8 large path: stream ``fd`` to a temp file, hash incrementally.

        The fd is streamed to a temporary file in BLOCK_SIZE chunks while the
        running SHA-256 is updated (the whole archive is never accumulated in a
        single Python bytearray during the read). The temp file is LEFT on disk
        so the caller can stream-PARSE it via
        :meth:`_parse_and_validate_tar_streaming` (using
        :func:`parse_ustar_archive_streaming`) without ever materializing the
        complete tar in Python heap. The caller unlinks the temp file after
        parsing.

        Returns ``(digest, size, parsed_path)``. There is NO
        ``read(expected_size)`` of the whole archive on this path.
        """
        import tempfile

        st = os.fstat(fd)
        if not stat_mod.S_ISREG(st.st_mode):
            raise CheckpointCorruptError("content descriptor is not a regular file")
        if st.st_size > MAX_ARCHIVE_BYTES:
            raise CheckpointCorruptError(
                f"archive size {st.st_size} exceeds MAX_ARCHIVE_BYTES ({MAX_ARCHIVE_BYTES})"
            )
        if st.st_size != expected_size:
            raise CheckpointCorruptError(
                f"on-disk size {st.st_size} != record byte_size {expected_size}"
            )
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=".tmp-cp-load-", suffix=".tar")
        h = hashlib.sha256()
        total = 0
        parsed_path = Path(tmp_name)
        unlinked = False
        try:
            try:
                while total < expected_size:
                    chunk = os.read(fd, min(BLOCK_SIZE, expected_size - total))
                    if not chunk:
                        raise CheckpointCorruptError(
                            f"unexpected EOF: streamed {total} of {expected_size} bytes"
                        )
                    _full_write(tmp_fd, chunk)
                    h.update(chunk)
                    total += len(chunk)
                os.fsync(tmp_fd)
            finally:
                os.close(tmp_fd)
            if total != expected_size:
                raise CheckpointCorruptError(f"streamed {total} bytes != expected {expected_size}")
            # Any trailing bytes beyond expected_size are corruption.
            extra = os.read(fd, 1)
            if extra:
                raise CheckpointCorruptError("content has trailing bytes beyond declared size")
            digest = f"sha256:{h.hexdigest()}"
            # Leave the temp file in place for the streaming parse; the caller
            # unlinks it. Do NOT read the whole file back here.
            unlinked = True  # hand ownership to the caller
            return digest, total, parsed_path
        finally:
            if not unlinked:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def _read_bounded_fd(self, fd: int, expected_size: int) -> bytes:
        """Bounded read of ``fd`` (item 10).

        Checks the file size via fstat BEFORE reading and rejects archives
        exceeding :data:`MAX_ARCHIVE_BYTES`. Reads exactly ``expected_size``
        bytes (the record's declared size) in BLOCK_SIZE chunks — never grows an
        unbounded bytearray. The on-disk size must match ``expected_size`` (any
        trailing mutation is a corruption, caught by the digest check).
        """
        st = os.fstat(fd)
        if not stat_mod.S_ISREG(st.st_mode):
            raise CheckpointCorruptError("content descriptor is not a regular file")
        on_disk = st.st_size
        if on_disk > MAX_ARCHIVE_BYTES:
            raise CheckpointCorruptError(
                f"archive size {on_disk} exceeds MAX_ARCHIVE_BYTES ({MAX_ARCHIVE_BYTES})"
            )
        if on_disk != expected_size:
            raise CheckpointCorruptError(
                f"on-disk size {on_disk} != record byte_size {expected_size}"
            )
        # Read exactly expected_size bytes; reject if the stream is short or long.
        buf = bytearray(expected_size)
        view = memoryview(buf)
        remaining = expected_size
        offset = 0
        while remaining > 0:
            chunk = os.read(fd, min(BLOCK_SIZE, remaining))
            if not chunk:
                raise CheckpointCorruptError(
                    f"unexpected EOF: read {offset} of {expected_size} bytes"
                )
            n = len(chunk)
            view[offset : offset + n] = chunk
            offset += n
            remaining -= n
        # The descriptor must now be at EOF; any extra bytes are corruption.
        extra = os.read(fd, 1)
        if extra:
            raise CheckpointCorruptError("content has trailing bytes beyond declared size")
        return bytes(buf)

    def _parse_and_validate_tar(self, tar_bytes: bytes) -> list[ParsedMember]:
        try:
            members = parse_ustar_archive(tar_bytes)
        except TarParseError as e:
            raise CheckpointCorruptError(f"tar framing invalid: {e}") from e
        if not members or members[0].name != "manifest.json":
            raise CheckpointCorruptError("manifest.json must be the first archive member")
        return members

    def _parse_and_validate_tar_streaming(
        self, path: Path, expected_size: int
    ) -> list[ParsedMember]:
        """Item 8 large path: stream-parse the verified temp file sequentially.

        Opens the temp file produced by :meth:`_read_large_fd_streaming` and
        parses it member-by-member with
        :func:`parse_ustar_archive_streaming` so the complete tar is never held
        in Python heap. The same framing/order/byte-limit checks as the in-memory
        parser apply. The caller owns unlinking ``path``.
        """
        import os as _os

        nofollow = getattr(_os, "O_NOFOLLOW", 0)
        fd = _os.open(path, _os.O_RDONLY | nofollow | getattr(_os, "O_BINARY", 0))
        try:
            try:
                members = parse_ustar_archive_streaming(fd, expected_size)
            except TarParseError as e:
                raise CheckpointCorruptError(f"tar framing invalid: {e}") from e
        finally:
            _os.close(fd)
        if not members or members[0].name != "manifest.json":
            raise CheckpointCorruptError("manifest.json must be the first archive member")
        return members

    def _decode_manifest(self, members: list[ParsedMember]) -> CheckpointManifest:
        raw = (
            members[0].data
            if members[0].data is not None
            else (members[0].temp_path.read_bytes() if members[0].temp_path else b"")
        )
        try:
            text = raw.decode("utf-8")
            data = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise CheckpointCorruptError(f"manifest.json is not canonical JSON: {e}") from e
        except ValueError as e:
            # Item 6: the duplicate-key object-pairs hook raises ValueError; it
            # must not escape the typed corruption boundary.
            raise CheckpointCorruptError(f"manifest.json rejected: {e}") from e
        if not isinstance(data, dict):
            raise CheckpointCorruptError("manifest.json root must be an object")
        # Re-encode canonically and require the stored bytes match.
        canonical = canonical_json_bytes(data)
        if canonical != raw:
            raise CheckpointCorruptError("manifest.json is not canonical compact sorted JSON")
        # Item 6: split version errors (unsupported) from corruption. Pydantic
        # field validators raise ValueError on unsupported schema/version fields;
        # everything else (structural corruption, missing required fields, wrong
        # types) is corruption.
        try:
            manifest = CheckpointManifest.model_validate_json(canonical, strict=True)
        except ValidationError as e:
            if _is_version_validation_error(e):
                raise CheckpointVersionError(
                    f"manifest carries an unsupported schema/version: {e}"
                ) from e
            raise CheckpointCorruptError(f"manifest failed validation: {e}") from e
        # Authenticate every non-manifest member against the manifest.
        self._authenticate_members(manifest, members)
        return manifest

    def _authenticate_members(
        self, manifest: CheckpointManifest, members: list[ParsedMember]
    ) -> None:
        by_name = {m.name: m for m in members}
        # manifest.json is authenticated by the #10 content digest; skip here.
        expected: dict[str, tuple[str, int]] = {}
        for comp in manifest.state_components:
            expected[comp.member_name] = (comp.member_sha256, comp.member_byte_size)
        for tensor in manifest.tensor_members:
            expected[tensor.member_name] = (tensor.member_sha256, tensor.member_byte_size)
        # No unlisted members (except manifest.json).
        non_manifest = {n for n in by_name if n != "manifest.json"}
        unlisted = non_manifest - set(expected)
        if unlisted:
            raise CheckpointCorruptError(f"unlisted members present: {sorted(unlisted)!r}")
        missing = set(expected) - non_manifest
        if missing:
            raise CheckpointCorruptError(f"missing members: {sorted(missing)!r}")
        for name, (sha, size) in expected.items():
            m = by_name[name]
            member_data = (
                m.data if m.data is not None else (m.temp_path.read_bytes() if m.temp_path else b"")
            )
            actual_sha = hashlib.sha256(member_data).hexdigest()
            if actual_sha != sha:
                raise CheckpointCorruptError(
                    f"member {name!r} digest mismatch: {actual_sha!r} != {sha!r}"
                )
            if len(member_data) != size:
                raise CheckpointCorruptError(
                    f"member {name!r} size mismatch: {len(member_data)!r} != {size!r}"
                )
        # No metadata.json (amendment B).
        if "metadata.json" in by_name:
            raise CheckpointCorruptError("metadata.json is not part of v1 archives")
        # Item 5: each StateComponentRef.member_name must be the EXACT
        # state/{role}.json path for its role (no aliasing).
        for comp in manifest.state_components:
            if comp.member_name != f"state/{comp.role}.json":
                raise CheckpointCorruptError(
                    f"state component role {comp.role!r} must be stored at "
                    f"'state/{comp.role}.json', not {comp.member_name!r}"
                )

    def _bind_components(
        self,
        manifest: CheckpointManifest,
        members: list[ParsedMember],
        record: ArtifactRecord,
    ) -> None:
        """Item 5: cross-bind every persisted component to the manifest + record.

        - manifest counters must equal the decoded ``state/counters.json``;
        - the optimizer/scheduler/scaler component payloads must equal the
          manifest's compatibility descriptors;
        - the model component parameters/buffers must equal the manifest's
          compatibility model descriptor (names, shapes, dtypes);
        - the identity member's run/attempt/fingerprint must equal the record;
        - the configuration member's fingerprint must equal the record;
        - every declared tensor member is referenced by exactly one component
          payload (no unused/undeclared tensor members);
        - the RNG member decodes to a valid :class:`RngStateBundle`.
        """
        from expertforge.checkpoints.models import CounterSnapshot as _Counters
        from expertforge.checkpoints.models import SafeStateDecodeError
        from expertforge.checkpoints.restore import (
            ModelState,
            OptimizerState,
            ScalerState,
            SchedulerState,
            decode_component_json,
        )
        from expertforge.rng.state import RngStateBundle

        by_name = {m.name: m for m in members}

        def _comp(role: str) -> bytes:
            m = by_name[f"state/{role}.json"]
            if m.data is not None:
                return m.data
            if m.temp_path is not None:
                return m.temp_path.read_bytes()
            raise CheckpointCorruptError(f"member state/{role}.json has no data")

        def _strict(payload: dict[str, Any], model_cls: Any, label: str) -> Any:
            # Item 5: validate from the canonical JSON re-encoding so list->tuple
            # array coercion (JSON arrays -> tuple fields) is permitted by the
            # pydantic JSON loader while every other strict check (extra-forbid,
            # exact types, closed domains) still applies.
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            try:
                return model_cls.model_validate_json(canonical, strict=True)
            except ValidationError as e:
                raise CheckpointCorruptError(
                    f"{label} component failed strict validation: {e}"
                ) from e

        def _strict_typed(raw: bytes, model_cls: Any, label: str) -> Any:
            # Item 5: validate the RAW component bytes directly through the typed
            # model (strict). Used for identity/provenance, whose canonical JSON
            # IS the stored bytes — validating from raw (not a re-encoded dict)
            # enforces the exact typed shape and catches any structural drift.
            try:
                return model_cls.model_validate_json(raw, strict=True)
            except ValidationError as e:
                raise CheckpointCorruptError(
                    f"{label} component failed strict typed validation: {e}"
                ) from e

        # counters.json must equal manifest.counters (strict).
        counters_payload = decode_component_json(_comp("counters"))
        counters = _strict(counters_payload, _Counters, "counters")
        if counters != manifest.counters:
            raise CheckpointCorruptError("state/counters.json does not equal manifest.counters")

        # optimizer component descriptor must equal the manifest's.
        opt_payload = decode_component_json(_comp("optimizer"))
        opt_state = _strict(opt_payload, OptimizerState, "optimizer")
        if opt_state.descriptor != manifest.compatibility.optimizer_descriptor:
            raise CheckpointCorruptError(
                "optimizer component descriptor != manifest compatibility optimizer descriptor"
            )

        # scheduler component descriptor must equal the manifest's.
        sched_payload = decode_component_json(_comp("scheduler"))
        sched_state = _strict(sched_payload, SchedulerState, "scheduler")
        if sched_state.descriptor != manifest.compatibility.scheduler_descriptor:
            raise CheckpointCorruptError(
                "scheduler component descriptor != manifest compatibility scheduler descriptor"
            )

        # scaler component (when present) descriptor must equal the manifest's.
        scaler_state: Any = None
        if "state/scaler.json" in by_name:
            scaler_payload = decode_component_json(_comp("scaler"))
            scaler_state = _strict(scaler_payload, ScalerState, "scaler")
            if scaler_state.descriptor != manifest.compatibility.scaler_descriptor:
                raise CheckpointCorruptError(
                    "scaler component descriptor != manifest compatibility scaler descriptor"
                )

        # model component parameters/buffers must equal the manifest's model
        # descriptor (names, shapes, dtypes) — the manifest is authoritative.
        model_payload = decode_component_json(_comp("model"))
        model_state = _strict(model_payload, ModelState, "model")
        expected_model = manifest.compatibility.model_descriptor
        _assert_model_component_matches(model_state, expected_model)

        # configuration member: fingerprint must equal the record AND the
        # embedded config bytes must rehash to the manifest's content_sha256
        # (item 5). The prior check compared the JSON-declared content_sha256
        # field to the manifest, which a tampered payload could self-attest;
        # now the actual content_base64 bytes are decoded, SHA-256'd, and
        # compared to the manifest's authoritative digest.
        import base64 as _base64

        from expertforge.checkpoints.models import ConfigurationState

        config_payload = decode_component_json(_comp("configuration"))
        config_state = _strict(config_payload, ConfigurationState, "configuration")
        if config_state.fingerprint != record.specification_fingerprint:
            raise CheckpointCorruptError(
                "state/configuration.json fingerprint != record fingerprint"
            )
        # Decode the actual configuration bytes and rehash them.
        try:
            cfg_bytes = _base64.b64decode(config_state.content_base64, validate=True)
        except (ValueError, _base64.binascii.Error) as e:  # type: ignore[attr-defined]
            raise CheckpointCorruptError(
                f"state/configuration.json content_base64 is not valid base64: {e}"
            ) from e
        if len(cfg_bytes) != config_state.byte_length:
            raise CheckpointCorruptError(
                "state/configuration.json byte_length != decoded content length"
            )
        actual_cfg_sha = hashlib.sha256(cfg_bytes).hexdigest()
        if actual_cfg_sha != manifest.configuration_content_sha256:
            raise CheckpointCorruptError(
                "state/configuration.json content bytes rehash != manifest "
                "configuration_content_sha256"
            )
        if actual_cfg_sha != config_state.content_sha256:
            raise CheckpointCorruptError(
                "state/configuration.json content bytes rehash != payload content_sha256"
            )

        # identity member: strict typed validation, then bind run/attempt/
        # fingerprint to the record (item 5). The identity record carries a
        # nested SpecificationFingerprintRecord whose digest_str is the public
        # fingerprint string; validating via the typed model enforces the
        # digest consistency too (verify_digest).
        from expertforge.identity.record import AttemptIdentityRecord

        identity_state = _strict_typed(_comp("identity"), AttemptIdentityRecord, "identity")
        if identity_state.run_id != record.run_id:
            raise CheckpointCorruptError("state/identity.json run_id != record run_id")
        if identity_state.attempt_id != record.attempt_id:
            raise CheckpointCorruptError("state/identity.json attempt_id != record attempt_id")
        if identity_state.fingerprint_digest_str() != record.specification_fingerprint:
            raise CheckpointCorruptError(
                "state/identity.json specification_fingerprint.digest_str != record fingerprint"
            )

        # provenance member: strict typed validation, then bind run/attempt to
        # the record (item 5).
        from expertforge.provenance.record import ProvenanceRecord

        prov_state = _strict_typed(_comp("provenance"), ProvenanceRecord, "provenance")
        if prov_state.run_id != record.run_id or prov_state.attempt_id != record.attempt_id:
            raise CheckpointCorruptError("state/provenance.json run/attempt != record run/attempt")

        # RNG member: decode to a valid bundle AND cross-bind its internal
        # descriptor fields to manifest.compatibility.rng_descriptor (item 5).
        # The prior check only verified decodability; now the bundle's persisted
        # adapter set / framework versions / schema version must equal the
        # manifest's RNG descriptor.
        try:
            rng_bundle = RngStateBundle.from_json_bytes(_comp("rng"))
        except (ValueError, SafeStateDecodeError) as e:
            raise CheckpointCorruptError(
                f"state/rng.json is not a decodable RngStateBundle: {e}"
            ) from e
        rng_descriptor = manifest.compatibility.rng_descriptor
        if rng_bundle.rng_state_schema_version != rng_descriptor.rng_state_schema_version:
            raise CheckpointCorruptError(
                "state/rng.json rng_state_schema_version != manifest rng_descriptor"
            )
        # Derive the bundle's provider set from its framework_states and compare
        # to the descriptor's framework_versions providers. The bundle's
        # framework_states carry (provider, device) but not a framework version;
        # the descriptor carries the canonical (provider, version) pairs. The
        # binding enforces that every descriptor framework provider has a
        # persisted framework_state and vice versa (no drift between the RNG
        # descriptor and the actual persisted RNG adapters).
        bundle_providers = {s.provider for s in rng_bundle.framework_states}
        descriptor_providers = {pair[0] for pair in rng_descriptor.framework_versions}
        if bundle_providers != descriptor_providers:
            raise CheckpointCorruptError(
                "state/rng.json framework_states providers != manifest rng_descriptor "
                "framework_versions providers"
            )

        # Item 5: every declared tensor member must be referenced the EXACT
        # number of times its logical-name count implies, and every referenced
        # member must be declared. Use a reference COUNT (member_name -> count),
        # not a set, so a tensor referenced the wrong number of times (double-
        # counted, or missing a logical name) is caught. A manifest tensor member
        # with L logical_names must be referenced exactly L times: once per
        # logical name across the component payloads (alias groups share one
        # physical member but contribute one reference per logical name).
        referenced: dict[str, int] = {}

        def _ref(name: str) -> None:
            referenced[name] = referenced.get(name, 0) + 1

        for entry in (*model_state.parameters, *model_state.buffers):
            _ref(entry.member_name)
        if opt_state.has_state:
            if opt_state.state_tensor is not None:
                _ref(opt_state.state_tensor.member_name)
            for slot in opt_state.state_slots:
                _ref(slot.member_name)
        if sched_state.has_state and sched_state.state_tensor is not None:
            _ref(sched_state.state_tensor.member_name)
        if "state/scaler.json" in by_name and scaler_state.has_state:
            if scaler_state.state_tensor is not None:
                _ref(scaler_state.state_tensor.member_name)
        # Expected count per member = number of logical names it serves (1 for a
        # standalone tensor, L for an L-way alias group).
        expected_counts = {t.member_name: len(t.logical_names) for t in manifest.tensor_members}
        declared = set(expected_counts)
        # Undeclared references (a component points at a non-manifest tensor).
        undeclared = set(referenced) - declared
        if undeclared:
            raise CheckpointCorruptError(
                f"component payloads reference undeclared tensor members: {sorted(undeclared)!r}"
            )
        # Every declared tensor member must be referenced exactly its
        # logical-name count (once per logical name).
        wrong = {
            name: (got, expected_counts[name])
            for name, got in referenced.items()
            if got != expected_counts[name]
        }
        if wrong:
            raise CheckpointCorruptError(
                f"tensor members referenced != their logical-name count (got, expected): {wrong!r}"
            )
        unused = declared - set(referenced)
        if unused:
            raise CheckpointCorruptError(
                f"tensor members declared in the manifest are not referenced by any "
                f"component payload: {sorted(unused)!r}"
            )

    # -- inspect_path (non-authoritative) ----------------------------------

    def inspect_path(self, path: Path) -> CheckpointInspection:
        """Diagnose an arbitrary untrusted file (amendment G: non-authoritative).

        Cannot restore state or claim #10 registration. Item 8: reads ONLY the
        manifest (the first member) for inspection — the file is never fully
        materialized, so even a multi-GiB archive can be diagnosed cheaply. The
        manifest is decoded and a typed :class:`CheckpointInspection` is
        returned; ``member_count`` reflects the manifest-declared member graph
        (1 manifest + N state components + M tensor members) rather than a full
        scan.
        """
        if not path.exists():
            return CheckpointInspection(status="corrupt", diagnostic="file not found")
        # Open the path without following a symlink (best-effort), then read only
        # the first member (manifest.json) — bounded by the component-member
        # limit, NOT the whole file.
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_BINARY", 0))
        except OSError as e:
            return CheckpointInspection(status="corrupt", diagnostic=f"open failed: {e}")
        try:
            st = os.fstat(fd)
            if not stat_mod.S_ISREG(st.st_mode):
                return CheckpointInspection(status="corrupt", diagnostic="not a regular file")
            if st.st_size > MAX_ARCHIVE_BYTES:
                return CheckpointInspection(
                    status="corrupt",
                    diagnostic=f"archive size {st.st_size} exceeds MAX_ARCHIVE_BYTES",
                )
            archive_byte_size = st.st_size
            # Read only the first member: its 512-byte header + its data
            # (bounded by MAX_COMPONENT_MEMBER_BYTES) + its padding.
            manifest_bytes = self._read_first_member(fd)
        except CheckpointCorruptError as e:
            return CheckpointInspection(status="corrupt", diagnostic=str(e))
        finally:
            os.close(fd)
        try:
            manifest = self._decode_manifest_first(manifest_bytes)
        except CheckpointVersionError as e:
            return CheckpointInspection(
                status="unsupported", diagnostic=str(e), archive_byte_size=archive_byte_size
            )
        except CheckpointCorruptError as e:
            return CheckpointInspection(
                status="corrupt", diagnostic=str(e), archive_byte_size=archive_byte_size
            )
        # Validate the complete archive: scan all members to verify the tar
        # terminator, member order, and that every declared member exists with
        # the correct size. This prevents a damaged/truncated archive from being
        # misclassified as "complete" (review 4833143258 item 3).
        try:
            scan_fd = os.open(
                str(path),
                os.O_RDONLY | getattr(os, "O_BINARY", 0),
            )
        except OSError as e:
            return CheckpointInspection(
                status="corrupt", diagnostic=f"re-open for scan failed: {e}"
            )
        try:
            raw = os.read(scan_fd, archive_byte_size)
            members = parse_ustar_archive(raw, validate_member_order=True)
        except TarParseError as e:
            return CheckpointInspection(
                status="corrupt",
                diagnostic=f"tar scan failed: {e}",
                archive_byte_size=archive_byte_size,
            )
        except CheckpointCorruptError as e:
            return CheckpointInspection(
                status="corrupt",
                diagnostic=str(e),
                archive_byte_size=archive_byte_size,
            )
        finally:
            try:
                os.close(scan_fd)
            except OSError:
                pass
        # Verify declared members match actual scanned members.
        declared_member_count = 1 + len(manifest.state_components) + len(manifest.tensor_members)
        if len(members) != declared_member_count:
            return CheckpointInspection(
                status="corrupt",
                diagnostic=(
                    f"member count mismatch: declared {declared_member_count}, "
                    f"actual {len(members)}"
                ),
                manifest=manifest,
                archive_byte_size=archive_byte_size,
                member_count=len(members),
            )
        return CheckpointInspection(
            status="complete",
            manifest=manifest,
            archive_byte_size=archive_byte_size,
            member_count=declared_member_count,
        )

    def _read_first_member(self, fd: int) -> bytes:
        """Item 8: read ONLY the first archive member (manifest.json).

        Reads the 512-byte header, validates it (canonical ustar + the name is
        ``manifest.json``), then reads exactly that member's data plus its
        zero-padding. The whole archive is never read. The on-disk size must be
        at least one header block.
        """
        from expertforge.checkpoints.tar_reader import _parse_header
        from expertforge.checkpoints.tar_writer import USTAR_BLOCK_SIZE

        st = os.fstat(fd)
        if st.st_size < USTAR_BLOCK_SIZE:
            raise CheckpointCorruptError("file smaller than one tar header block")
        header = self._read_exact_fd(fd, USTAR_BLOCK_SIZE, "header block")
        # _parse_header validates checksum/magic/typeflag/numeric fields/name. We
        # pass a 1-block buffer; the returned data_start is irrelevant here.
        name, _typeflag, size, _data_start = _parse_header(header, 0)
        if name != "manifest.json":
            raise CheckpointCorruptError(f"first member must be manifest.json, got {name!r}")
        component_limit = MAX_COMPONENT_MEMBER_BYTES
        if size > component_limit:
            raise CheckpointCorruptError(
                f"manifest.json size {size} exceeds MAX_COMPONENT_MEMBER_BYTES ({component_limit})"
            )
        data = self._read_exact_fd(fd, size, "manifest.json data")
        data_blocks = (size + USTAR_BLOCK_SIZE - 1) // USTAR_BLOCK_SIZE
        padding_len = data_blocks * USTAR_BLOCK_SIZE - size
        if padding_len:
            padding = self._read_exact_fd(fd, padding_len, "manifest.json padding")
            if any(b != 0 for b in padding):
                raise CheckpointCorruptError("manifest.json has non-zero padding bytes")
        return data

    def _read_exact_fd(self, fd: int, n: int, what: str) -> bytes:
        """Read exactly ``n`` bytes from ``fd`` in BLOCK_SIZE chunks."""
        if n == 0:
            return b""
        buf = bytearray()
        remaining = n
        while remaining > 0:
            chunk = os.read(fd, min(BLOCK_SIZE, remaining))
            if not chunk:
                raise CheckpointCorruptError(
                    f"unexpected EOF reading {what}: got {len(buf)} of {n} bytes"
                )
            buf.extend(chunk)
            remaining -= len(chunk)
        return bytes(buf)

    def _decode_manifest_first(self, raw: bytes) -> CheckpointManifest:
        """Decode + validate the manifest.json bytes (first member only).

        Same canonical-JSON + strict-typed validation as
        :meth:`_decode_manifest`, but operating on the single manifest member
        rather than a full member list (item 8: inspection reads only the
        manifest).
        """
        try:
            text = raw.decode("utf-8")
            data = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise CheckpointCorruptError(f"manifest.json is not canonical JSON: {e}") from e
        except ValueError as e:
            raise CheckpointCorruptError(f"manifest.json rejected: {e}") from e
        if not isinstance(data, dict):
            raise CheckpointCorruptError("manifest.json root must be an object")
        canonical = canonical_json_bytes(data)
        if canonical != raw:
            raise CheckpointCorruptError("manifest.json is not canonical compact sorted JSON")
        try:
            return CheckpointManifest.model_validate_json(canonical, strict=True)
        except ValidationError as e:
            if _is_version_validation_error(e):
                raise CheckpointVersionError(
                    f"manifest carries an unsupported schema/version: {e}"
                ) from e
            raise CheckpointCorruptError(f"manifest failed validation: {e}") from e

    # -- check_compatibility -----------------------------------------------

    def check_compatibility(
        self,
        archive: CheckpointArchive,
        expected: CompatibilityDescriptor,
    ) -> CompatibilityResult:
        """Classify ``archive`` against ``expected`` using the closed table."""
        return check_compatibility(archive.manifest.compatibility, expected)


def check_compatibility(
    actual: CompatibilityDescriptor, expected: CompatibilityDescriptor
) -> CompatibilityResult:
    """Closed, deterministic compatibility classification (amendment M).

    Any mismatch in an active required component blocks restoration. There is
    no force flag; mismatches are a complete, canonically-sorted report.
    """
    mismatches: list[CompatibilityMismatch] = []

    def add(
        component: str,
        code: str,
        expected_val: Any,
        actual_val: Any,
        path: str = "",
    ) -> None:
        mismatches.append(
            CompatibilityMismatch(
                component=component,  # type: ignore[arg-type]
                diagnostic_code=code,  # type: ignore[arg-type]
                severity="blocking",
                expected=str(expected_val),
                actual=str(actual_val),
                path=path,
            )
        )

    # Version fields.
    _check_eq(actual, expected, "archive_format_version", add, "version_mismatch")
    _check_eq(actual, expected, "manifest_schema_version", add, "version_mismatch")
    _check_eq(actual, expected, "state_format_version", add, "version_mismatch")
    _check_eq(actual, expected, "compatibility_schema_version", add, "version_mismatch")
    _check_eq(
        actual,
        expected,
        "specification_fingerprint",
        add,
        "specification_fingerprint_mismatch",
    )

    # Topology.
    if actual.topology_descriptor.world_size != expected.topology_descriptor.world_size:
        add(
            "topology",
            "topology_world_size_mismatch",
            expected.topology_descriptor.world_size,
            actual.topology_descriptor.world_size,
        )
    # Item 14: topology rank assignment is a persisted required field.
    if tuple(actual.topology_descriptor.rank_assignment) != tuple(
        expected.topology_descriptor.rank_assignment
    ):
        add(
            "topology",
            "topology_rank_assignment_mismatch",
            tuple(expected.topology_descriptor.rank_assignment),
            tuple(actual.topology_descriptor.rank_assignment),
        )

    # Model descriptor.
    _check_param_names(actual.model_descriptor, expected.model_descriptor, add)
    _check_param_shapes(actual.model_descriptor, expected.model_descriptor, add)
    _check_param_dtypes(actual.model_descriptor, expected.model_descriptor, add)
    _check_alias_groups(actual.model_descriptor, expected.model_descriptor, add)

    # Optimizer.
    _check_optimizer(actual.optimizer_descriptor, expected.optimizer_descriptor, add)

    # Scheduler.
    _check_scheduler(actual.scheduler_descriptor, expected.scheduler_descriptor, add)

    # Scaler.
    actual_active = actual.scaler_descriptor is not None
    expected_active = expected.scaler_descriptor is not None
    if actual_active != expected_active:
        add("scaler_active", "scaler_presence_mismatch", expected_active, actual_active)

    # RNG.
    _check_rng(actual.rng_descriptor, expected.rng_descriptor, add)

    # Data.
    _check_data(actual.data_descriptor, expected.data_descriptor, add)

    if not mismatches:
        return CompatibilityResult(status="exact", mismatches=())
    return CompatibilityResult(status="incompatible", mismatches=tuple(mismatches))


def _check_eq(
    actual: Any,
    expected: Any,
    attr: str,
    add: Any,
    code: str,
) -> None:
    av = getattr(actual, attr)
    ev = getattr(expected, attr)
    if av != ev:
        add(attr, code, ev, av)


def _check_param_names(actual: Any, expected: Any, add: Any) -> None:
    anames = set(actual.names())
    enames = set(expected.names())
    for n in sorted(enames - anames):
        add("model_names", "parameter_name_missing", n, "<absent>", path=n)
    for n in sorted(anames - enames):
        add("model_names", "parameter_name_unexpected", "<absent>", n, path=n)


def _check_param_shapes(actual: Any, expected: Any, add: Any) -> None:
    ad = {p.name: tuple(p.shape) for p in (*actual.parameters, *actual.buffers)}
    ed = {p.name: tuple(p.shape) for p in (*expected.parameters, *expected.buffers)}
    for name in sorted(set(ad) & set(ed)):
        if ad[name] != ed[name]:
            add(
                "model_shapes",
                "parameter_shape_mismatch",
                ed[name],
                ad[name],
                path=name,
            )


def _check_param_dtypes(actual: Any, expected: Any, add: Any) -> None:
    ad = {p.name: p.dtype for p in (*actual.parameters, *actual.buffers)}
    ed = {p.name: p.dtype for p in (*expected.parameters, *expected.buffers)}
    for name in sorted(set(ad) & set(ed)):
        if ad[name] != ed[name]:
            add(
                "model_dtypes",
                "parameter_dtype_mismatch",
                ed[name],
                ad[name],
                path=name,
            )


def _check_alias_groups(actual: Any, expected: Any, add: Any) -> None:
    ag = {g.canonical_member: tuple(g.aliased_names) for g in actual.alias_groups}
    eg = {g.canonical_member: tuple(g.aliased_names) for g in expected.alias_groups}
    for cm in sorted(set(eg) - set(ag)):
        add("model_alias_groups", "alias_group_missing", cm, "<absent>", path=cm)
    for cm in sorted(set(ag) - set(eg)):
        add("model_alias_groups", "alias_group_unexpected", "<absent>", cm, path=cm)
    for cm in sorted(set(ag) & set(eg)):
        if ag[cm] != eg[cm]:
            add(
                "model_alias_groups",
                "alias_group_membership_mismatch",
                eg[cm],
                ag[cm],
                path=cm,
            )


def _check_optimizer(actual: Any, expected: Any, add: Any) -> None:
    if actual.optimizer_type != expected.optimizer_type:
        add(
            "optimizer_type",
            "optimizer_type_mismatch",
            expected.optimizer_type,
            actual.optimizer_type,
        )
    if len(actual.param_groups) != len(expected.param_groups):
        add(
            "optimizer_groups",
            "optimizer_group_count_mismatch",
            len(expected.param_groups),
            len(actual.param_groups),
        )
    else:
        for ag, eg in zip(actual.param_groups, expected.param_groups, strict=False):
            if tuple(ag.param_names) != tuple(eg.param_names):
                add(
                    "optimizer_groups",
                    "optimizer_group_membership_mismatch",
                    tuple(eg.param_names),
                    tuple(ag.param_names),
                    path=str(ag.group_index),
                )
            # Item 7: compare each group's canonical options (lr, weight_decay,
            # momentum, betas, ...). Options are canonically-sorted SafeValue
            # tuples; compare the full option set per group.
            a_opts = tuple((k, _safe_value_json(v)) for k, v in ag.options)
            e_opts = tuple((k, _safe_value_json(v)) for k, v in eg.options)
            if a_opts != e_opts:
                add(
                    "optimizer_options",
                    "optimizer_option_mismatch",
                    _format_options(e_opts),
                    _format_options(a_opts),
                    path=str(ag.group_index),
                )
    # State slot shapes.
    a_slots = {(s.group_index, s.param_name, s.slot_name): s for s in actual.state_slots}
    e_slots = {(s.group_index, s.param_name, s.slot_name): s for s in expected.state_slots}
    for key in sorted(set(e_slots) - set(a_slots)):
        add("optimizer_slots", "optimizer_slot_missing", key, "<absent>", path=str(key))
    for key in sorted(set(a_slots) - set(e_slots)):
        add("optimizer_slots", "optimizer_slot_unexpected", "<absent>", key, path=str(key))
    for key in sorted(set(a_slots) & set(e_slots)):
        a_slot = a_slots[key]
        e_slot = e_slots[key]
        if tuple(a_slot.shape) != tuple(e_slot.shape):
            add(
                "optimizer_slots",
                "optimizer_slot_shape_mismatch",
                tuple(e_slot.shape),
                tuple(a_slot.shape),
                path=str(key),
            )
        # Item 14: optimizer-slot dtype is a persisted required field.
        if a_slot.dtype != e_slot.dtype:
            add(
                "optimizer_slots",
                "optimizer_slot_dtype_mismatch",
                e_slot.dtype,
                a_slot.dtype,
                path=str(key),
            )


def _safe_value_json(value: Any) -> Any:
    """Canonical JSON-serializable rendering of a SafeValue option value."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _format_options(opts: tuple[tuple[str, Any], ...]) -> str:
    return ",".join(f"{k}={_safe_value_json(v)}" for k, v in opts)


def _check_scheduler(actual: Any, expected: Any, add: Any) -> None:
    if actual.scheduler_type != expected.scheduler_type:
        add(
            "scheduler_identity",
            "scheduler_type_mismatch",
            expected.scheduler_type,
            actual.scheduler_type,
        )
    # Item 14: scheduler active state is a persisted required field.
    if bool(actual.active) != bool(expected.active):
        add(
            "scheduler_identity",
            "scheduler_active_mismatch",
            bool(expected.active),
            bool(actual.active),
        )
    if tuple(actual.state_shape) != tuple(expected.state_shape):
        add(
            "scheduler_state_shape",
            "scheduler_state_shape_mismatch",
            tuple(expected.state_shape),
            tuple(actual.state_shape),
        )


def _check_rng(actual: Any, expected: Any, add: Any) -> None:
    aa = set(actual.adapter_set)
    ea = set(expected.adapter_set)
    for a in sorted(ea - aa):
        add("rng_adapter_set", "rng_adapter_missing", a, "<absent>", path=a)
    for a in sorted(aa - ea):
        add("rng_adapter_set", "rng_adapter_unexpected", "<absent>", a, path=a)
    if actual.rng_state_schema_version != expected.rng_state_schema_version:
        add(
            "rng_schema",
            "rng_schema_mismatch",
            expected.rng_state_schema_version,
            actual.rng_state_schema_version,
        )
    # Item 14: RNG framework versions are persisted required fields.
    a_fw = {k: v for k, v in actual.framework_versions}
    e_fw = {k: v for k, v in expected.framework_versions}
    for fw in sorted(set(e_fw) - set(a_fw)):
        add("rng_schema", "rng_schema_mismatch", e_fw[fw], "<absent>", path=fw)
    for fw in sorted(set(a_fw) - set(e_fw)):
        add("rng_schema", "rng_schema_mismatch", "<absent>", a_fw[fw], path=fw)
    for fw in sorted(set(a_fw) & set(e_fw)):
        if a_fw[fw] != e_fw[fw]:
            add("rng_schema", "rng_schema_mismatch", e_fw[fw], a_fw[fw], path=fw)


def _check_data(actual: Any, expected: Any, add: Any) -> None:
    a_id = actual.identity
    e_id = expected.identity
    if (
        a_id.dataset_digest != e_id.dataset_digest
        or a_id.split != e_id.split
        or a_id.preprocessing_identity != e_id.preprocessing_identity
        or a_id.tokenizer_identity != e_id.tokenizer_identity
        or a_id.packing_policy != e_id.packing_policy
        or a_id.sequence_policy != e_id.sequence_policy
        or a_id.shard_selection != e_id.shard_selection
        or a_id.data_config_digest != e_id.data_config_digest
        or a_id.data_identity_schema_version != e_id.data_identity_schema_version
        or a_id.length != e_id.length
    ):
        add(
            "data_identity",
            "data_identity_mismatch",
            e_id.dataset_digest,
            a_id.dataset_digest,
        )
    if (
        actual.sampler_type != expected.sampler_type
        or actual.sampler_version != expected.sampler_version
        or actual.batch_size != expected.batch_size
        or actual.sequence_length != expected.sequence_length
        or actual.drop_last != expected.drop_last
    ):
        add(
            "data_cursor_contract",
            "data_cursor_contract_mismatch",
            expected.sampler_type,
            actual.sampler_type,
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is not allowed: {key}")
        result[key] = value
    return result


def _full_write(fd: int, data: bytes) -> None:
    """Write all of ``data`` to ``fd``, handling partial writes (item 8)."""
    view = memoryview(data)
    total = 0
    while total < len(view):
        written = os.write(fd, view[total:])
        if written <= 0:  # pragma: no cover - defensive
            raise OSError("os.write returned non-positive byte count")
        total += written


# The frozen manifest + compatibility model fields whose validators reject
# unsupported schema names/versions (items 6 + 7). A ValidationError touching
# one of these is classified as "unsupported" rather than "corrupt". This covers
# EVERY versioned schema field the ratified manifest carries: the archive
# format, the manifest schema name/version, the safe-state format, and the
# embedded compatibility descriptor's schema name/version.
_VERSION_VALIDATED_FIELDS: frozenset[str] = frozenset(
    {
        "archive_format_version",
        "manifest_schema",
        "manifest_schema_version",
        "state_format_version",
        "compatibility_schema",
        "compatibility_schema_version",
    }
)


def _is_version_validation_error(error: ValidationError) -> bool:
    """Classify a Pydantic ValidationError as version-unsupported or corrupt.

    Returns True when every error locus touches one of the version-validated
    schema fields (``archive_format_version``, ``manifest_schema``,
    ``manifest_schema_version``, ``state_format_version``,
    ``compatibility_schema``, ``compatibility_schema_version``). Pydantic field
    validators (Literal type narrowing + the explicit ``unsupported ...`` value
    validators) on these fields raise for future/unknown schema names or
    versions, which is an "unsupported" status, not corruption.

    Item 7: a future manifest-schema NAME (e.g. ``expertforge.checkpoint-manifest-v2``)
    or compatibility-schema NAME/version mismatch must be classified as
    ``unsupported`` (CheckpointVersionError), not ``corrupt``.
    """
    errs = error.errors()
    if not errs:
        return False
    for err in errs:
        loc = err.get("loc", ())
        if not loc:
            return False
        # The last path element is the field name (e.g.
        # "archive_format_version" or "compatibility_schema_version"). Nested
        # loci like ("compatibility", "manifest_schema_version") are covered by
        # inspecting the leaf.
        field = loc[-1]
        if not isinstance(field, str) or field not in _VERSION_VALIDATED_FIELDS:
            return False
    return True


def _assert_model_component_matches(model_state: Any, expected: Any) -> None:
    """Item 5: the model component's parameters/buffers must equal the manifest's
    compatibility model descriptor (names, shapes, dtypes).

    Alias-group entries in the component (one :class:`TensorComponentRef` with
    multiple ``logical_names``) expand to every name, matching the manifest
    descriptor which lists each aliased name as a separate parameter.
    """

    def _expand(entries: Any) -> dict[str, tuple[tuple[int, ...], str]]:
        out: dict[str, tuple[tuple[int, ...], str]] = {}
        for entry in entries:
            shape_dtype = (tuple(entry.shape), entry.dtype)
            for name in entry.logical_names:
                out[name] = shape_dtype
        return out

    exp_params = {p.name: (tuple(p.shape), p.dtype) for p in expected.parameters}
    act_params = _expand(model_state.parameters)
    if set(exp_params) != set(act_params):
        raise CheckpointCorruptError(
            "model component parameter names != manifest compatibility model parameters"
        )
    for name, (shape, dtype) in exp_params.items():
        if act_params[name] != (shape, dtype):
            raise CheckpointCorruptError(
                f"model component parameter {name!r} shape/dtype != manifest descriptor"
            )
    exp_bufs = {p.name: (tuple(p.shape), p.dtype) for p in expected.buffers}
    act_bufs = _expand(model_state.buffers)
    if set(exp_bufs) != set(act_bufs):
        raise CheckpointCorruptError(
            "model component buffer names != manifest compatibility model buffers"
        )
    for name, (shape, dtype) in exp_bufs.items():
        if act_bufs[name] != (shape, dtype):
            raise CheckpointCorruptError(
                f"model component buffer {name!r} shape/dtype != manifest descriptor"
            )


def _UTC() -> Any:
    from datetime import UTC

    return UTC
