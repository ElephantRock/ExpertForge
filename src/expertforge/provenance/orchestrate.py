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
from expertforge.identity.lineage import ResumeLineage
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.provenance.hardware import capture_accelerator, capture_hardware, capture_topology
from expertforge.provenance.record import (
    AcceleratorInfo,
    ProvenanceRecord,
    SoftwareEnvironment,
    TopologyInfo,
)
from expertforge.provenance.sidecar import ProvenanceSidecarError, write_provenance_sidecar
from expertforge.provenance.software import capture_software_environment
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
    mode: AllocationMode = AllocationMode.INDEPENDENT,
    lineage: ResumeLineage | None = None,
    parent_specification_fingerprint: str | None = None,
    retained_run_id: str | None = None,
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

    # 3. Emit identity containing the source input.
    try:
        identity, _identity_path = emit_attempt_identity(
            artifact_root=artifact_root,
            config_envelope=config_envelope,
            mode=mode,
            lineage=lineage,
            parent_specification_fingerprint=parent_specification_fingerprint,
            retained_run_id=retained_run_id,
            immutable_inputs=immutable_inputs,
            clock=clock,
            entropy=entropy,
            exists=exists,
        )
    except IdentityEmitError as e:
        raise ProvenanceOrchestrationError(str(e)) from e

    # 4. Capture typed provenance against the identity. Use capture_hardware()
    # (which returns a HardwareAggregate bundling accelerator + topology +
    # topology_warnings) instead of calling capture_accelerator() and
    # capture_topology() separately. ``topology_warnings`` carries invalid
    # numeric topology env values that could not be parsed — without
    # capture_hardware() they would be silently lost between the two calls.
    resolved_software = software or capture_software_environment(repo_root=repo)
    if hardware is not None or topology is not None:
        # An explicit override short-circuits the aggregate; honor each
        # supplied section independently and default the other.
        resolved_hardware = hardware or capture_accelerator()
        resolved_topology = topology or capture_topology()
    else:
        aggregate = capture_hardware()
        resolved_hardware = aggregate.accelerator
        # Preserve the captured topology_warnings durably on the topology model.
        resolved_topology = TopologyInfo(
            status=aggregate.topology.status,
            rank=aggregate.topology.rank,
            local_rank=aggregate.topology.local_rank,
            world_size=aggregate.topology.world_size,
            node_count=aggregate.topology.node_count,
            backend=aggregate.topology.backend,
            reason=aggregate.topology.reason,
            topology_warnings=aggregate.topology_warnings,
        )

    # Top-level completeness is derived from ALL sections by from_identity()
    # (completeness=None is the default). Do NOT supply a source-only override —
    # that bypasses whole-record completeness derivation and could mask a
    # hardware/software/topology error as "complete".
    provenance = ProvenanceRecord.from_identity(
        identity,
        source=snap,
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
