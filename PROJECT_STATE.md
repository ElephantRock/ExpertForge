# PROJECT_STATE.md

**Index of current project state.** This file is an index, not a substitute for
detailed issue and pull-request records. When this file conflicts with the
underlying issues/PRs or with `main`, those prevail.

Last updated: Milestone 0 in progress; #7 provenance completed.

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

- **#8 — M0.5 deterministic seed and RNG-state management** — next
  dependency-ordered implementation action (depends only on #4/#5).
- **#9 — M0.6 structured logging and metric collection** — may proceed now
  that #7 (provenance) is complete.
- **#10 — M0.7 artifact management** — may proceed now; registers the
  provenance sidecar (produced by #7) and the identity sidecar (#6).
- **#13 — M0.10 test/CI harness** — may begin early in parallel to establish
  the harness that subsystem tests plug into.
- #3 umbrella and #11/#12/#14 are open and dependency-ordered.
- **#4 — M0.1 repository and Python package scaffold — completed.**
- **#5 — M0.2 validated configuration resolution — completed.**
- **#6 — M0.3 run identity and configuration fingerprints — completed.**
- **#7 — M0.4 source provenance and execution environment — completed.** The
  provenance subsystem (source-content snapshot as behavioral identity;
  identity-bound record; allowlist-first software capture; optional
  hardware/topology providers with typed degradation; durable provenance sidecar;
  human-readable summary) is on `main`.

## Open pull requests

- None. The #7 provenance PR merged.

## Known blockers

- None. After #7, the foundation + reproducibility/observability layers are
  advancing; #8/#9/#10 may proceed, with #11 (checkpoints) after #8, #12
  (manifests) after #5/#6/#7/#9/#10/#11, and #14 (smoke gate) last.

## Latest accepted experiment

- None yet. No experiments run; experiments will be recorded through the
  `experiment` issue template (which now requires a maturity stage and a
  research family) and the experiment doctrine
  ([doctrine/evaluation-and-experiments.md](doctrine/evaluation-and-experiments.md)).

## Next recommended action

1. Begin **#8 (seed/RNG)** — next in the #3 dependency order (depends only on
   #4/#5).
2. In parallel, **#9 (logging)**, **#10 (artifacts)**, and **#13 (test/CI
   harness)** may proceed.
3. #11 (checkpoints) after #8; #12 (manifests) after #5/#6/#7/#9/#10/#11; #14
   (smoke gate) last.
4. After Milestone 0 is complete, begin D0 (minimal dense baseline, F0
   exact-attention control).

## Relationship to ExpertOS

ExpertOS is the external runtime/control-plane counterpart. ExpertForge builds
its own models and does not copy ExpertOS internals; shared interfaces are
governed by versioned schemas and the ExpertOS resource contract defined here
([doctrine/deployment-and-runtime-codesign.md](doctrine/deployment-and-runtime-codesign.md)).
