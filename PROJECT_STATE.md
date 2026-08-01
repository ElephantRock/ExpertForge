# PROJECT_STATE.md

**Index of current project state.** This file is an index, not a substitute for
detailed issue and pull-request records. When this file conflicts with the
underlying issues/PRs or with `main`, those prevail.

Last updated: Milestone 0 in progress; #11 complete, #12 next, #13 active. 9 of 11 child issues complete.

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

**Progress: 9 of 11 child issues complete** (#4–#11). Remaining: #12, #14.

## Active baseline

- No model baseline exists yet. The first model milestone is **D0 (minimal dense
  baseline)**; its entry criteria require Milestone 0 to be complete first
  ([doctrine/model-lineage.md](doctrine/model-lineage.md) §2). The D0 component
  choices are recorded in
  [doctrine/decisions/0002-initial-d0-architecture.md](doctrine/decisions/0002-initial-d0-architecture.md).

## Active issues

- **#12 — M0.9 experiment manifests — next implementation issue.** All dependencies
  (#5/#6/#7/#9/#10/#11) are complete.
- **#13 — M0.10 test/CI harness — in progress.** Phase A merged via PR #21,
  squash commit `aaa31187d5d6b63671ee96df14a85a75b125f068`: permanent read-only CI,
  strict fast/integration/smoke/accelerator tiers, pinned toolchain, and
  repository-native policy checks are active. #13 remains open until the full
  Milestone 0 component and integration suite is present.
- **#14 — smoke and recovery gate** remains last.
- #3 umbrella remains open.
- **#4 — M0.1 repository and Python package scaffold — completed.**
- **#5 — M0.2 validated configuration resolution — completed.**
- **#6 — M0.3 run identity and configuration fingerprints — completed.**
- **#7 — M0.4 source provenance and execution environment — completed** via
  PR #18, squash commit `f59916228d730675151149c74e0dde2ac2a1aadc`.
- **#8 — M0.5 deterministic seed and RNG-state management — completed** via
  PR #24, squash commit `fa49c95f7b926dc7d2255ae923b90a1512a21147`.
- **#9 — M0.6 structured logging and metrics — completed** via PR #27, squash
  commit `99c5a6b6e6f60be9ef3ad645c1d2bd743b12fbad`.
- **#10 — M0.7 artifact management and content hashing — completed** via PR #32,
  squash commit `fa4a82d1da1f49a29ebe310bb6d531144a12fa6a`.
- **#11 — M0.8 checkpoint serialization and exact restoration — completed** via
  PR #35, squash commit `6801184133f9ea1ef49bce2464407fa166e3c1d5`. Deterministic
  ustar tar packaging, safe structured state encoding (SafeValue), exact restoration
  transaction (factory/commit/abort with RNG-last rollback), compatibility checking
  (closed diagnostics, no force flag), #10 publication boundary, streaming load/save,
  TOCTOU-safe content opening, pure-Python reference adapter. 7 review passes.

## Open pull requests

- This state synchronization PR (documentation-only).

## Known blockers

- None. #12 is unblocked and may proceed.
- #13 remains open as a cross-cutting integration responsibility, not a blocker.
- #14 waits for the complete Milestone 0 substrate (#12 + all prior).

## Latest accepted experiment

- None yet. No experiments run; experiments will be recorded through the
  `experiment` issue template (which requires a maturity stage and a research
  family) and the experiment doctrine
  ([doctrine/evaluation-and-experiments.md](doctrine/evaluation-and-experiments.md)).

## Next recommended action

1. Begin **#12 (experiment manifests)** — the sole remaining implementation issue
   before the smoke gate.
2. Continue **#13** incrementally as #12 lands; do not close it until the full
   Milestone 0 suite, schema checks, and exact commands satisfy its issue contract.
3. Complete **#14 (smoke-and-recovery gate)** strictly last.
4. After Milestone 0 is complete, begin D0 (minimal dense baseline, F0
   exact-attention control).

## Critical path

```text
#12 experiment manifests → #14 smoke-and-recovery gate
#13 CI remains active throughout
```

## Relationship to ExpertOS

ExpertOS is the external runtime/control-plane counterpart. ExpertForge builds
its own models and does not copy ExpertOS internals; shared interfaces are
governed by versioned schemas and the ExpertOS resource contract defined here
([doctrine/deployment-and-runtime-codesign.md](doctrine/deployment-and-runtime-codesign.md)).
