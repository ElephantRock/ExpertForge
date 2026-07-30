# PROJECT_STATE.md

**Index of current project state.** This file is an index, not a substitute for
detailed issue and pull-request records. When this file conflicts with the
underlying issues/PRs or with `main`, those prevail.

Last updated: Milestone 0 in progress; #8 complete, #11 unblocked, #13 Phase A CI foundation active.

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

- **#9 — M0.6 structured logging and metrics** — available next now that #7 is
  complete; #10 follows #9.
- **#11 — M0.8 checkpoint serialization and exact restoration** — now available
  following #8 completion; depends on #8 seed/RNG for deterministic state
  capture/restoration.
- **#13 — M0.10 test/CI harness — in progress.** Phase A merged via PR #21,
  squash commit `aaa31187d5d6b63671ee96df14a85a75b125f068`: permanent read-only CI,
  strict fast/integration/smoke/accelerator tiers, pinned toolchain, and
  repository-native policy checks are active. #13 remains open until the full
  Milestone 0 component and integration suite is present.
- **#10 — artifact management** follows #9; **#12 — experiment manifests**
  follows #5/#6/#7/#9/#10/#11; **#14 — smoke and recovery gate** remains last.
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

## Open pull requests

- None after this state synchronization merges.

## Known blockers

- #9 has no remaining dependency blocker and may proceed. Its PR must extend the
  permanent #13 fast/integration tiers with component coverage.
- #11 is now unblocked following #8 and may proceed.
- #13 remains open as a cross-cutting integration responsibility, not a blocker
  to beginning #9 or #11.
- #10 waits for #9; #12 waits for #9/#10/#11; #14 waits for the complete
  Milestone 0 substrate.

## Latest accepted experiment

- None yet. No experiments run; experiments will be recorded through the
  `experiment` issue template (which requires a maturity stage and a research
  family) and the experiment doctrine
  ([doctrine/evaluation-and-experiments.md](doctrine/evaluation-and-experiments.md)).

## Next recommended action

1. Begin **#9 (logging/metrics)** — unlocks #10 artifact management.
2. In parallel, **#11 (checkpoints)** may proceed now that #8 is complete —
   unlocks #12 experiment manifests.
3. Continue **#13** incrementally as #9–#12 land; do not close it until the full
   Milestone 0 suite, schema checks, and exact commands satisfy its issue contract.
4. After #9, implement **#10 (artifacts)**. After #9/#10/#11, implement **#12
   (manifests)**, then complete **#14 (smoke-and-recovery gate)** last.
5. After Milestone 0 is complete, begin D0 (minimal dense baseline, F0
   exact-attention control).

## Relationship to ExpertOS

ExpertOS is the external runtime/control-plane counterpart. ExpertForge builds
its own models and does not copy ExpertOS internals; shared interfaces are
governed by versioned schemas and the ExpertOS resource contract defined here
([doctrine/deployment-and-runtime-codesign.md](doctrine/deployment-and-runtime-codesign.md)).
