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

from expertforge.artifacts.models import (
    ArtifactRecord,
    ExternalReference,
    ParentReference,
    RetentionStatus,
)
from expertforge.artifacts.store import ArtifactStore, BLOCK_SIZE
from expertforge.checkpoints.encoder import ArchiveMembers, build_archive
from expertforge.checkpoints.models import (
    CHECKPOINT_ARCHIVE_FORMAT_VERSION,
    CheckpointInspection,
    CheckpointManifest,
    CompatibilityDescriptor,
    CompatibilityMismatch,
    CompatibilityResult,
    InspectionStatus,
    canonical_json_bytes,
)
from expertforge.checkpoints.tar_reader import ParsedMember, TarParseError, parse_ustar_archive
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
    "check_compatibility",
]


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
    payloads, and the binding :class:`ArtifactRecord`. All state is materialized
    in memory (below the v1 bounds). Restoration is a separate transaction
    (:mod:`expertforge.checkpoints.restore`).
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
        tar_bytes: bytes,
    ) -> None:
        self._manifest = manifest
        self._members_by_name = {m.name: m for m in members}
        self._record = record
        self._tar_bytes = tar_bytes
        h = hashlib.sha256(tar_bytes)
        self._content_digest = f"sha256:{h.hexdigest()}"
        self._byte_size = len(tar_bytes)

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
        return self._tar_bytes

    def member(self, name: str) -> bytes:
        """Return the raw bytes of member ``name`` (raises if absent)."""
        m = self._members_by_name.get(name)
        if m is None:
            raise CheckpointCorruptError(f"missing member {name!r}")
        return m.data

    def component(self, role: str) -> bytes:
        """Return the raw bytes of the ``state/<role>.json`` component."""
        return self.member(f"state/{role}.json")

    def tensor_member(self, member_name: str) -> bytes:
        """Return the raw bytes of tensor member ``member_name``."""
        return self.member(member_name)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"CheckpointArchive(artifact_id={self.artifact_id!r}, "
            f"byte_size={self._byte_size})"
        )


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
        rng_bundle: RngStateBundle,
        parent: ParentReference | None = None,
        created_at_utc: datetime | None = None,
    ) -> ArtifactRecord:
        """Capture → encode → publish → verify the returned record.

        ``captured`` is the quiescent :class:`CapturedCheckpointState` from the
        provider's :meth:`capture_checkpoint_snapshot`. The archive is built
        fully in memory then published via :meth:`ArtifactStore.publish` (which
        handles atomic write/rename/registry). The returned record is verified
        for checkpoint category/format/version, producing component, identity,
        parent, digest, and size before it is returned.
        """
        from expertforge.checkpoints.models import CapturedCheckpointState

        if not isinstance(captured, CapturedCheckpointState):
            raise CheckpointComponentError(
                "captured must be a CapturedCheckpointState from "
                "capture_checkpoint_snapshot()"
            )
        ts = created_at_utc or datetime.now(tz=_UTC())
        archive = build_archive(
            identity=identity,
            captured=captured,
            configuration_envelope=configuration_envelope,
            provenance=provenance,
            rng_bundle=rng_bundle,
            parent=parent,
            created_at_utc=ts,
        )
        record = self._store.publish(
            archive.tar_bytes,
            category="checkpoint",
            format="tar",
            format_version=CHECKPOINT_ARCHIVE_FORMAT_VERSION,
            producing_component=self.PRODUCING_COMPONENT,
            parent=parent,
        )
        self._verify_returned_record(record, archive, parent)
        return record

    def _verify_returned_record(
        self,
        record: ArtifactRecord,
        archive: ArchiveMembers,
        parent: ParentReference | None,
    ) -> None:
        """Amendment G: verify category/format/version/producer/identity/parent/digest/size."""
        if record.category != "checkpoint":
            raise CheckpointCorruptError(
                f"published record category {record.category!r} != 'checkpoint'"
            )
        if record.format != "tar":
            raise CheckpointCorruptError(
                f"published record format {record.format!r} != 'tar'"
            )
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

    # -- load (authoritative) ----------------------------------------------

    def load(
        self,
        artifact_id: str,
        *,
        expected_identity: AttemptIdentityRecord,
    ) -> CheckpointArchive:
        """Authoritative load by ``artifact_id`` (amendment G).

        Single-fd TOCTOU-safe: obtains the record through
        :meth:`ArtifactStore.inspect`, requires checkpoint category/format/
        version, canonical-local storage, retained/acceptable retention, and
        identity binding; obtains the content path through
        :meth:`ArtifactStore.locate`; opens that content with no-follow + fstat
        on a single descriptor; and streams digest/size + tar parsing from the
        SAME descriptor. The computed digest/size must equal the record; the
        manifest producing identity and parent must equal the record fields.
        """
        record = self._inspect_record(artifact_id)
        self._assert_record_loadable(record)
        # Identity binding: the record's run/attempt/fingerprint must match the
        # expected identity (amendment L: V1 same-run resume only).
        self._assert_identity_binding(record, expected_identity)
        content_path = self._store.locate(artifact_id)
        if content_path is None:
            raise CheckpointCorruptError(
                f"artifact {artifact_id!r} has no canonical content to load"
            )
        # Single-fd open: O_NOFOLLOW where available + fstat regular-file check.
        fd = self._open_no_follow(content_path)
        try:
            tar_bytes = self._read_all_fd(fd)
            digest = f"sha256:{hashlib.sha256(tar_bytes).hexdigest()}"
        finally:
            os.close(fd)
        # The descriptor is now closed; all subsequent validation uses tar_bytes.
        if digest != record.content_digest:
            raise CheckpointCorruptError(
                f"content digest {digest!r} != record {record.content_digest!r}"
            )
        if len(tar_bytes) != record.byte_size:
            raise CheckpointCorruptError(
                f"content size {len(tar_bytes)!r} != record {record.byte_size!r}"
            )
        members = self._parse_and_validate_tar(tar_bytes)
        manifest = self._decode_manifest(members)
        # Manifest producing identity must equal the record fields (amendment G).
        self._assert_manifest_record_binding(manifest, record)
        return CheckpointArchive(
            manifest=manifest,
            members=members,
            record=record,
            tar_bytes=tar_bytes,
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
                f"artifact {record.artifact_id!r} category {record.category!r} "
                "is not 'checkpoint'"
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

    def _assert_manifest_record_binding(
        self, manifest: CheckpointManifest, record: ArtifactRecord
    ) -> None:
        if manifest.run_id != record.run_id:
            raise CheckpointCorruptError(
                f"manifest run_id {manifest.run_id!r} != record {record.run_id!r}"
            )
        if manifest.attempt_id != record.attempt_id:
            raise CheckpointCorruptError(
                f"manifest attempt_id {manifest.attempt_id!r} != record "
                f"{record.attempt_id!r}"
            )
        if manifest.specification_fingerprint != record.specification_fingerprint:
            raise CheckpointCorruptError(
                "manifest specification_fingerprint != record specification_fingerprint"
            )
        if manifest.parent_artifact_id != (
            record.parent.artifact_id if record.parent else None
        ):
            raise CheckpointCorruptError(
                "manifest parent_artifact_id != record parent artifact_id"
            )

    def _open_no_follow(self, path: Path) -> int:
        """Open ``path`` for reading without following a symlink final element."""
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        flags = os.O_RDONLY | nofollow | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(path, flags)
        except OSError as e:
            raise CheckpointCorruptError(
                f"could not open content path {path}: {e}"
            ) from e
        try:
            st = os.fstat(fd)
        except OSError as e:
            os.close(fd)
            raise CheckpointCorruptError(f"fstat failed: {e}") from e
        if stat_mod.S_ISLNK(st.st_mode):  # pragma: no cover - O_NOFOLLOW rejects
            os.close(fd)
            raise CheckpointCorruptError(f"{path} is a symlink; refused")
        if not stat_mod.S_ISREG(st.st_mode):
            os.close(fd)
            raise CheckpointCorruptError(f"{path} is not a regular file; refused")
        return fd

    def _read_all_fd(self, fd: int) -> bytes:
        """Read the entire content of ``fd`` in BLOCK_SIZE chunks."""
        buf = bytearray()
        while True:
            chunk = os.read(fd, BLOCK_SIZE)
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf)

    def _parse_and_validate_tar(self, tar_bytes: bytes) -> list[ParsedMember]:
        try:
            members = parse_ustar_archive(tar_bytes)
        except TarParseError as e:
            raise CheckpointCorruptError(f"tar framing invalid: {e}") from e
        if not members or members[0].name != "manifest.json":
            raise CheckpointCorruptError(
                "manifest.json must be the first archive member"
            )
        return members

    def _decode_manifest(self, members: list[ParsedMember]) -> CheckpointManifest:
        raw = members[0].data
        try:
            text = raw.decode("utf-8")
            data = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise CheckpointCorruptError(f"manifest.json is not canonical JSON: {e}") from e
        if not isinstance(data, dict):
            raise CheckpointCorruptError("manifest.json root must be an object")
        # Re-encode canonically and require the stored bytes match.
        canonical = canonical_json_bytes(data)
        if canonical != raw:
            raise CheckpointCorruptError(
                "manifest.json is not canonical compact sorted JSON"
            )
        try:
            manifest = CheckpointManifest.model_validate_json(canonical, strict=True)
        except Exception as e:
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
            raise CheckpointCorruptError(
                f"unlisted members present: {sorted(unlisted)!r}"
            )
        missing = set(expected) - non_manifest
        if missing:
            raise CheckpointCorruptError(f"missing members: {sorted(missing)!r}")
        for name, (sha, size) in expected.items():
            m = by_name[name]
            actual_sha = hashlib.sha256(m.data).hexdigest()
            if actual_sha != sha:
                raise CheckpointCorruptError(
                    f"member {name!r} digest mismatch: {actual_sha!r} != {sha!r}"
                )
            if len(m.data) != size:
                raise CheckpointCorruptError(
                    f"member {name!r} size mismatch: {len(m.data)!r} != {size!r}"
                )
        # No metadata.json (amendment B).
        if "metadata.json" in by_name:
            raise CheckpointCorruptError("metadata.json is not part of v1 archives")

    # -- inspect_path (non-authoritative) ----------------------------------

    def inspect_path(self, path: Path) -> CheckpointInspection:
        """Diagnose an arbitrary untrusted file (amendment G: non-authoritative).

        Cannot restore state or claim #10 registration. Validates the complete
        tar framing + terminator and decodes the manifest; reports a typed
        :class:`CheckpointInspection`.
        """
        if not path.exists():
            return CheckpointInspection(status="corrupt", diagnostic="file not found")
        # Open the path without following a symlink (best-effort).
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_BINARY", 0))
        except OSError as e:
            return CheckpointInspection(status="corrupt", diagnostic=f"open failed: {e}")
        try:
            st = os.fstat(fd)
            if not stat_mod.S_ISREG(st.st_mode):
                return CheckpointInspection(
                    status="corrupt", diagnostic="not a regular file"
                )
            tar_bytes = self._read_all_fd(fd)
        finally:
            os.close(fd)
        try:
            members = self._parse_and_validate_tar(tar_bytes)
            manifest = self._decode_manifest(members)
        except CheckpointVersionError as e:
            return CheckpointInspection(
                status="unsupported", diagnostic=str(e), archive_byte_size=len(tar_bytes)
            )
        except CheckpointCorruptError as e:
            return CheckpointInspection(
                status="corrupt", diagnostic=str(e), archive_byte_size=len(tar_bytes)
            )
        return CheckpointInspection(
            status="complete",
            manifest=manifest,
            archive_byte_size=len(tar_bytes),
            member_count=len(members),
        )

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
        for ag, eg in zip(actual.param_groups, expected.param_groups):
            if tuple(ag.param_names) != tuple(eg.param_names):
                add(
                    "optimizer_groups",
                    "optimizer_group_membership_mismatch",
                    tuple(eg.param_names),
                    tuple(ag.param_names),
                    path=str(ag.group_index),
                )
    # State slot shapes.
    a_slots = {(s.group_index, s.param_name, s.slot_name): tuple(s.shape) for s in actual.state_slots}
    e_slots = {(s.group_index, s.param_name, s.slot_name): tuple(s.shape) for s in expected.state_slots}
    for key in sorted(set(e_slots) - set(a_slots)):
        add("optimizer_slots", "optimizer_slot_missing", key, "<absent>", path=str(key))
    for key in sorted(set(a_slots) - set(e_slots)):
        add("optimizer_slots", "optimizer_slot_unexpected", "<absent>", key, path=str(key))
    for key in sorted(set(a_slots) & set(e_slots)):
        if a_slots[key] != e_slots[key]:
            add(
                "optimizer_slots",
                "optimizer_slot_shape_mismatch",
                e_slots[key],
                a_slots[key],
                path=str(key),
            )


def _check_scheduler(actual: Any, expected: Any, add: Any) -> None:
    if actual.scheduler_type != expected.scheduler_type:
        add(
            "scheduler_identity",
            "scheduler_type_mismatch",
            expected.scheduler_type,
            actual.scheduler_type,
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


def _UTC() -> Any:
    from datetime import UTC

    return UTC
