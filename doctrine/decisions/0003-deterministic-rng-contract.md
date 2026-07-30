# Decision Record 0003 — Deterministic RNG Contract

**Status:** Proposed in PR for Issue #8
**Date:** 2026-07-30
**Supersedes:** none
**Related:** Issue #8; Issue #11 checkpointing; `doctrine/data-and-training.md`

## Context

ExpertForge requires reproducible initialization, data order, worker behavior,
sampling, and checkpoint recovery. A single integer copied into every library is
insufficient: it couples unrelated streams, makes worker/rank behavior fragile,
and does not preserve the exact next-sample position needed for resume.

The repository currently has one canonical seed at `training.seed` and no
mandatory tensor-framework dependency. The RNG substrate must therefore be
framework-neutral, CPU-testable, explicit about side effects, and ready for
checkpoint embedding without pre-empting Issue #11's file-format ownership.

## Decision

1. `training.seed` is the only root seed and remains part of canonical resolved
   configuration bytes.
2. Every logical random stream is derived with a versioned, domain-separated
   SHA-256 envelope containing stable component, worker, rank, device, and stream
   coordinates.
3. The full digest identifies the stream. Fixed 64-bit and 32-bit projections are
   only provider inputs and do not replace the digest as collision-resistant
   identity.
4. Child streams use readable `<parent>.<suffix>` names when they fit the canonical
   component domain. A long valid parent uses a full SHA-256 namespace over the
   complete parent/suffix pair; names are never truncated.
5. RNG mutation occurs only through explicit `RngManager.initialize()` or
   `RngManager.restore_state()` calls. Importing the package has no RNG or backend
   side effects.
6. Initialization and restoration are failure-atomic where provider APIs permit:
   Python, NumPy, provider RNG state, and provider determinism configuration are
   snapshotted before mutation and rolled back if any later operation fails. The
   owned NumPy generator and initialization evidence remain private until the
   complete operation succeeds.
7. Every run seeds Python's global RNG, NumPy's legacy global RNG, and an owned
   NumPy PCG64 `Generator`. NumPy is therefore runtime infrastructure.
8. Tensor frameworks integrate through explicit adapters. PyTorch is the first
   adapter and is imported lazily; it is not a mandatory dependency at this
   milestone.
9. PyTorch CPU state and every available CUDA device ordinal receive separately
   derived stream identities and seed projections. Framework state ordering is
   canonical: provider order, CPU first, then accelerator kind and numeric device
   ordinal.
10. `reproducible` mode requests deterministic provider behavior and uses either
    fail-fast or stable warning policy. `performance` mode is explicitly
    non-reproducible and may enable faster backend choices.
11. RNG state is a frozen, versioned, non-pickle Pydantic payload. Python and NumPy
    states are structural. Framework byte states use canonical base64 transport
    with SHA-256 verification. Persisted JSON permits structural array-to-tuple
    decoding but forbids scalar/domain coercion. Issue #11 embeds this payload
    into checkpoints.

## Alternatives considered

- **Reuse the root seed everywhere.** Rejected because component additions and
  worker/rank changes perturb unrelated streams and create avoidable collisions.
- **Seed every CUDA device with one value.** Rejected because single-process
  multi-device stochastic operations could consume correlated streams.
- **Truncate long child component names.** Rejected because truncation can erase
  semantic distinctions. Full-hash fallback preserves the complete namespace.
- **Allow partial mutation on provider failure.** Rejected because callers could
  observe a manager that reports initialization failure while process or provider
  RNGs had already advanced to a new state.
- **Use Python `hash()` for derivation.** Rejected because hash randomization and
  implementation details are not a stable cross-process contract.
- **Serialize provider states with pickle.** Rejected because arbitrary object
  deserialization is unsafe and non-canonical.
- **Make PyTorch mandatory now.** Rejected because Milestone 0 has not yet selected
  or implemented the tensor training stack, and CI must remain lightweight and
  CPU-portable.
- **Promise cross-version or cross-hardware bit identity.** Rejected because
  provider algorithms and kernels may change. Reproducibility is scoped to the
  declared software/provider/device conditions unless a provider documents a
  stronger guarantee.

## Evidence

- SHA-256 canonical derivation is independent of Python hash randomization,
  process ordering, timestamps, and host paths.
- Structural state models allow strict validation before mutation and direct
  deterministic JSON embedding in future checkpoints.
- Separate streams permit stable component, worker, rank, device, and sampling
  behavior without reusing the root seed.
- Provider doubles exercise configuration/state snapshots, distinct device
  streams, seed failure, restore failure, and complete rollback.
- Permanent CI tests same-seed equality, different-seed sensitivity, state
  round-trip, subprocess repeatability, adapter behavior, and failure paths.

## Consequences

- NumPy becomes a required runtime dependency for every run.
- Callers must explicitly allocate stable stream coordinates and must not seed
  libraries ad hoc from `training.seed`.
- Framework adapters must expose non-mutating seed planning, configuration
  snapshot/restoration, RNG-state capture, restore validation, and exact restore.
- Checkpoint implementations must include the entire `RngStateBundle` and restore
  it before consuming the next random sample.
- Performance mode results cannot be promoted or described as reproducible without
  separate evidence.
- Optional accelerator tests skip explicitly when the provider or hardware is not
  available; CPU determinism remains the required baseline.

## Reversal conditions

A new decision record may supersede this contract if a selected tensor framework
requires a different canonical state representation, if a stronger portable RNG
standard is adopted across Python/NumPy/frameworks, or if checkpoint format needs
an incompatible state schema. Such a change requires explicit schema/version
bumps, migration policy, regression evidence, and the normal issue/PR review path.
