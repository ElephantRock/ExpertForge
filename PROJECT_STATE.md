# PROJECT_STATE.md

**Index of current project state.** This file is an index, not a substitute for
detailed issue and pull-request records. When this file conflicts with the
underlying issues/PRs or with `main`, those prevail.

Last updated: Milestone 0 in progress; #10 complete, #11 available next, #13 Phase A CI foundation active.

## Current milestone

- **Milestone 0 — Experimental substrate.** Status: **in progress**; this is the
  current priority.
- No large training run begins before Milestone 0 is complete.
- Milestone 0 is complete when the repository provides: validated configuration
  loading; run identity generation; source provenance capture; deterministic seed
  management; logging and metric collection; artifact management; checkpoint
  serialization and restoration; experiment-manifest generation; hardware /
  environment capture; automated tests; a minimal training smoke test
  (umbrella #3; charter; AGENTS; founding spec §20).

## Milestone 0 issue hierarchy

Umbrella: **#3 — Milestone 0: Build the experimental substrate.** Dependency
order (see #3 for the full graph):

- **Foundation** — #4 scaffold · #5 configuration · #6 run identity
- **Reproducibility & observability** — #7 provenance/environment · #8 seed/RNG ·
  #9 logging/metrics
- **Persistence & evidence** — #10 artifacts · #11 checkpoints · #12 manifests
- **Verification & integration** — #13 test/CI harness · #14 smoke-and-recovery gate

#13 integrates tests throughout Milestone 0; #14 is strictly last.

## Active baseline

- No model baseline exists yet. The first model milestone is **D0 (minimal dense
  baseline)**; its entry criteria require Milestone 0 to be complete first
  ([doctrine/model-lineage.md](doctrine/model-lineage.md) §2). The D0 component
  choices are recorded in
  [doctrine/decisions/0002-initial-d0-architecture.md](doctrine/decisions/0002-initial-d0-architecture.md).

## Active issues

- **#11 — M0.8 checkpoint serialization and exact restoration — available next.**
  Its #5/#6/#8/#10 dependencies are complete; its completion unlocks #12.
- **#13 — M0.10 test/CI harness — in progress.** Phase A merged via PR #21,
  squash commit `aaa31187d5d6b63671ee96df14a85a75b125f068`: permanent read-only CI,
  strict fast/integration/smoke/accelerator tiers, pinned toolchain, and
  repository-native policy checks are active. #13 remains open until the full
  Milestone 0 component and integration suite is present.
- **#12 — experiment manifests** follows #5/#6/#7/#9/#10/#11; **#14 — smoke and
  recovery gate** remains last.
- #3 umbrella remains open.
- **#4 — M0.1 repository and Python package scaffold — completed.**
- **#5 — M0.2 validated configuration resolution — completed.**
- **#6 — M0.3 run identity and configuration fingerprints — completed.**
- **#7 — M0.4 source provenance and execution environment — completed** via
  PR #18, squash commit `f59916228d730675151149c74e0dde2ac2a1aadc`.
- **#8 — M0.5 deterministic seed and RNG-state management — completed** via
  PR #24, squash commit `fa49c95f7b926dc7d2255ae923b90a1512a21147`. Versioned
  SHA-256 seed derivation, `RngManager` with Python/NumPy/optional-PyTorch
  adapters, immutable strict checkpoint state serialization, exact
  capture/restoration, failure-atomic initialization and restoration, ADR 0003.
- **#9 — M0.6 structured logging and metrics — completed** via PR #27, squash
  commit `99c5a6b6e6f60be9ef3ad645c1d2bd743b12fbad`. Identity-bound versioned event
  and metric schemas, canonical per-rank JSONL, deterministic console rendering,
  durable writer lifecycle, authoritative loading/diagnostic scanning, structural
  redaction, severity filtering, and permanent fast/integration coverage are in
  place.
- **#10 — M0.7 artifact management and content hashing — completed** via PR #32,
  squash commit `fa4a82d1da1f49a29ebe310bb6d531144a12fa6a`. Strict immutable artifact,
  registry, and external-reference schemas; deterministic content-addressed
  bundles; atomic publication; lock-serialized append-only registry mutation;
  typed verification; telemetry registration; external recovery; retention
  transitions; path safety; and permanent fast/integration coverage are in place.

## Open pull requests

- None after this state synchronization merges.

## Known blockers

- #11 has no remaining dependency blocker and may proceed. Its PR must publish
  checkpoint artifacts through #10 and extend the permanent #13 fast/integration
  tiers with exact-restoration, corruption, compatibility, lineage, and
  deterministic-continuation coverage.
- #13 remains open as a cross-cutting integration responsibility, not a blocker
  to beginning #11.
- #12 waits for #11; #14 waits for the complete Milestone 0 substrate.

## Latest accepted experiment

- None yet. No experiments run; experiments will be recorded through the
  `experiment` issue template (which requires a maturity stage and a research
  family) and the experiment doctrine
  ([doctrine/evaluation-and-experiments.md](doctrine/evaluation-and-experiments.md)).

## Next recommended action

1. Begin **#11 (checkpoint serialization and exact restoration)** — the sole
   next implementation issue on the persistence critical path.
2. After #11, implement **#12 (experiment manifests)**.
3. Continue **#13** incrementally as #11–#12 land; do not close it until the full
   Milestone 0 suite, schema checks, and exact commands satisfy its issue contract.
4. Complete **#14 (smoke-and-recovery gate)** strictly last.
5. After Milestone 0 is complete, begin D0 (minimal dense baseline, F0
   exact-attention control).

## Relationship to ExpertOS

ExpertOS is the external runtime/control-plane counterpart. ExpertForge builds
its own models and does not copy ExpertOS internals; shared interfaces are
governed by versioned schemas and the ExpertOS resource contract defined here
([doctrine/deployment-and-runtime-codesign.md](doctrine/deployment-and-runtime-codesign.md)).
