# PROJECT_STATE.md

**Index of current project state.** This file is an index, not a substitute for
detailed issue and pull-request records. When this file conflicts with the
underlying issues/PRs or with `main`, those prevail.

Last updated: Milestone 0 in progress; #7 provenance/environment completed on `main`.

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

#13 may begin early; #14 is strictly last.

## Active baseline

- No model baseline exists yet. The first model milestone is **D0 (minimal dense
  baseline)**; its entry criteria require Milestone 0 to be complete first
  ([doctrine/model-lineage.md](doctrine/model-lineage.md) §2). The D0 component
  choices are recorded in
  [doctrine/decisions/0002-initial-d0-architecture.md](doctrine/decisions/0002-initial-d0-architecture.md).

## Active issues

- **#8 — M0.5 deterministic seed and RNG-state management** — available next;
  depends only on #4/#5.
- **#9 — M0.6 structured logging and metrics** — available next now that #7 is
  complete; #10 follows #9.
- **#13 — M0.10 test/CI harness** — may proceed early in parallel.
- **#10 — artifact management** follows #9; **#11 — checkpoints** follows #8;
  **#12 — experiment manifests** follows #5/#6/#7/#9/#10/#11; **#14 — smoke and
  recovery gate** remains last.
- #3 umbrella remains open.
- **#4 — M0.1 repository and Python package scaffold — completed.**
- **#5 — M0.2 validated configuration resolution — completed.**
- **#6 — M0.3 run identity and configuration fingerprints — completed.**
- **#7 — M0.4 source provenance and execution environment — completed** via
  PR #18, squash commit `f59916228d730675151149c74e0dde2ac2a1aadc`.

## Open pull requests

- None after this state synchronization merges.

## Known blockers

- #8, #9, and #13 have no remaining dependency blocker and may proceed.
- #10 waits for #9; #11 waits for #8; #12 waits for #9/#10/#11; #14 waits for
  the complete Milestone 0 substrate.

## Latest accepted experiment

- None yet. No experiments run; experiments will be recorded through the
  `experiment` issue template (which requires a maturity stage and a research
  family) and the experiment doctrine
  ([doctrine/evaluation-and-experiments.md](doctrine/evaluation-and-experiments.md)).

## Next recommended action

1. Begin **#8 (seed/RNG)** and **#9 (logging/metrics)**; **#13 (test/CI harness)**
   may proceed in parallel.
2. After #9, implement **#10 (artifacts)**; after #8, implement **#11
   (checkpoints)**.
3. Implement **#12 (manifests)** after #9/#10/#11, then complete **#14
   (smoke-and-recovery gate)** last.
4. After Milestone 0 is complete, begin D0 (minimal dense baseline, F0
   exact-attention control).

## Relationship to ExpertOS

ExpertOS is the external runtime/control-plane counterpart. ExpertForge builds
its own models and does not copy ExpertOS internals; shared interfaces are
governed by versioned schemas and the ExpertOS resource contract defined here
([doctrine/deployment-and-runtime-codesign.md](doctrine/deployment-and-runtime-codesign.md)).
