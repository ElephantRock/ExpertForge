# PROJECT_STATE.md

**Index of current project state.** This file is an index, not a substitute for
detailed issue and pull-request records. When this file conflicts with the
underlying issues/PRs or with `main`, those prevail.

Last updated: post-merge state for the Issue #1 founding-scope correction.

## Current milestone

- **Milestone 0 — Experimental substrate.** Status: **not started**; this is the
  current priority.
- No large training run begins before Milestone 0 is complete.
- Milestone 0 is complete when the repository provides: validated configuration
  loading; run identity generation; source provenance capture; deterministic seed
  management; logging and metric collection; artifact management; checkpoint
  serialization and restoration; experiment-manifest generation; hardware /
  environment capture; automated tests; a minimal training smoke test
  (charter → AGENTS → founding spec §20).

## Founding-scope correction (accepted)

- The founding-scope inversion introduced by the bootstrap was corrected by
  Issue #1 / PR #2 and merged into `main`.
- Issue #1 is **completed** (`Closes #1` on merge). No correction PR remains
  open.
- ExpertForge is established as the complete model-building project (data,
  tokenizer, dense and MoE architectures, training, evaluation, instrumentation,
  reference inference, model–runtime co-design). ExpertOS is the separate
  runtime/control-plane counterpart.

## Active baseline

- No model baseline exists yet. The first model milestone is **D0 (minimal dense
  baseline)**; its entry criteria require Milestone 0 to be complete first
  ([doctrine/model-lineage.md](doctrine/model-lineage.md) §2). The D0 component
  choices are recorded in
  [doctrine/decisions/0002-initial-d0-architecture.md](doctrine/decisions/0002-initial-d0-architecture.md).

## Active issues

- None open. Issue #1 is completed.

## Open pull requests

- None.

## Known blockers

- None.

## Latest accepted experiment

- None yet. No experiments run; experiments will be recorded through the
  `experiment` issue template (which now requires a maturity stage and a
  research family) and the experiment doctrine
  ([doctrine/evaluation-and-experiments.md](doctrine/evaluation-and-experiments.md)).

## Next recommended action

1. Create the Milestone 0 issue hierarchy (config loader, run identity,
   provenance capture, seed management, logging, artifact management,
   checkpoint serialization, experiment manifests, environment capture, tests,
   training smoke test).
2. After Milestone 0 is complete, begin D0 (minimal dense baseline, F0
   exact-attention control).

## Relationship to ExpertOS

ExpertOS is the external runtime/control-plane counterpart. ExpertForge builds
its own models and does not copy ExpertOS internals; shared interfaces are
governed by versioned schemas and the ExpertOS resource contract defined here
([doctrine/deployment-and-runtime-codesign.md](doctrine/deployment-and-runtime-codesign.md)).
