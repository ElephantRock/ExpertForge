"""Source→identity→provenance orchestration (Issue #7 review item 1).

One API performs the full normative runtime order::

    capture source snapshot
    → create ImmutableInput("source.snapshot")
    → emit identity containing that input
    → capture typed provenance against the identity
    → write run-provenance.json before training
    → return (identity, provenance, sidecar_path)

The emitted specification fingerprint contains exactly one ``source.snapshot``
input whose digest is the versioned source-snapshot envelope digest.
"""

from __future__ import annotations

from pathlib import Path

from expertforge.config.resolve import ResolutionEnvelope
from expertforge.identity.emit import AllocationMode, IdentityEmitError, emit_attempt_identity
from expertforge.identity.fingerprint import (
    ImmutableInput,
)
from expertforge.identity.ids import ClockProvider, EntropyProvider, ExistsPredicate
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.provenance.hardware import capture_accelerator, capture_topology
from expertforge.provenance.record import (
    AcceleratorInfo,
    CompletenessInfo,
    ProvenanceRecord,
    SoftwareEnvironment,
    SourceState,
    SourceStateSummary,
    TopologyInfo,
)
from expertforge.provenance.sidecar import ProvenanceSidecarError, write_provenance_sidecar
from expertforge.provenance.software import (
    capture_software_environment,
    sanitize_repository_url,
)
from expertforge.provenance.source_snapshot import (
    DirtySourceError,
    SourceUnavailableError,
    capture_source_snapshot,
    source_snapshot_immutable_input,
)

__all__ = ["ProvenanceOrchestrationError", "prepare_run"]


class ProvenanceOrchestrationError(Exception):
    """Raised when the full source→identity→provenance chain cannot complete."""


def prepare_run(
    *,
    artifact_root: Path,
    config_envelope: ResolutionEnvelope,
    repo: Path,
    allow_dirty: bool = False,
    clock: ClockProvider | None = None,
    entropy: EntropyProvider | None = None,
    exists: ExistsPredicate | None = None,
    extra_immutable_inputs: list[ImmutableInput] | None = None,
    software: SoftwareEnvironment | None = None,
    hardware: AcceleratorInfo | None = None,
    topology: TopologyInfo | None = None,
) -> tuple[AttemptIdentityRecord, ProvenanceRecord, Path]:
    """Perform the full normative runtime order in one call.

    Returns ``(identity, provenance, sidecar_path)``. The identity's
    specification fingerprint contains ``source.snapshot`` (and any extra
    immutable inputs). The provenance sidecar is written before return.

    Raises :class:`ProvenanceOrchestrationError` on any failure.
    """
    # 1. Capture source snapshot.
    try:
        snap = capture_source_snapshot(repo, allow_dirty=allow_dirty)
    except DirtySourceError as e:
        raise ProvenanceOrchestrationError(str(e)) from e
    except SourceUnavailableError as e:
        raise ProvenanceOrchestrationError(str(e)) from e

    # 2. Create the source.snapshot ImmutableInput.
    source_input = source_snapshot_immutable_input(snap)
    immutable_inputs = [source_input]
    if extra_immutable_inputs:
        immutable_inputs.extend(extra_immutable_inputs)

    # 3. Emit identity containing the source input (INDEPENDENT mode by default).
    try:
        identity, _identity_path = emit_attempt_identity(
            artifact_root=artifact_root,
            config_envelope=config_envelope,
            mode=AllocationMode.INDEPENDENT,
            immutable_inputs=immutable_inputs,
            clock=clock,
            entropy=entropy,
            exists=exists,
        )
    except IdentityEmitError as e:
        raise ProvenanceOrchestrationError(str(e)) from e

    # 4. Capture typed provenance against the identity.
    resolved_software = software or capture_software_environment(repo_root=repo)
    resolved_hardware = hardware or capture_accelerator()
    resolved_topology = topology or capture_topology()

    # Build the source state from the snapshot.
    sanitized_url = sanitize_repository_url(snap.remote_url) if snap.remote_url else None
    source_state = SourceState(
        summary=SourceStateSummary(
            commit_sha=snap.commit_sha,
            branch=snap.branch,
            remote_url=sanitized_url,
            is_clean=snap.is_clean,
            is_canonical=snap.is_canonical,
            tree_digest=snap.tree_digest,
            input_digest=snap.input_digest,
        ),
        completeness=CompletenessInfo(
            status="complete",
            warnings=tuple() if snap.is_clean else ("non_canonical_dirty_source",),
        ),
    )

    provenance = ProvenanceRecord.from_identity(
        identity,
        source=source_state,
        software=resolved_software,
        hardware=resolved_hardware,
        topology=resolved_topology,
    )

    # 5. Write run-provenance.json before training can begin.
    try:
        path = write_provenance_sidecar(artifact_root, provenance)
    except ProvenanceSidecarError as e:
        raise ProvenanceOrchestrationError(str(e)) from e

    return identity, provenance, path
