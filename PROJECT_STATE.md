# PROJECT_STATE.md

**Index of current project state.** This file is an index, not a substitute for
detailed issue and pull-request records. When this file conflicts with the
underlying issues/PRs or with `main`, those prevail.

Last updated: Issue #1 correction (founding scope restored).

## Current milestone

- **Milestone 0 — Experimental substrate.** Status: **not started**; this is the
  current priority.
- No large training run begins before Milestone 0 is complete.
- Milestone 0 is complete when the repository provides: validated configuration
  loading; run identity generation; source provenance capture; deterministic seed
  management; logging and metric collection; artifact management; checkpoint
  serialization and restoration; experiment-manifest generation; hardware /
  environment capture; automated tests; a minimal training smoke test
  (charter → AGENTS → spec §20).

## Current priority

- **Issue #1 — Correct founding scope: ExpertForge is the model-building
  project.** Doctrine corrected to restore the full founding specification
  (identity, mission, research question, D0–M5 lineage, scientific and training
  doctrine, promotion/kill criteria, deployment-aware policy, ExpertOS contract,
  Milestone 0). See [doctrine/charter.md](doctrine/charter.md).

## Active baseline

- No model baseline exists yet. The first model milestone is **D0 (minimal dense
  baseline)**; its entry criteria require Milestone 0 to be complete first
  ([doctrine/model-lineage.md](doctrine/model-lineage.md) §2).

## Active issues

- **#1** Correct founding scope (open, joint: web review, local implementation).

## Open pull requests

- PR for Issue #1 on `local/1-correct-founding-scope` (this correction).

## Known blockers

- None. After Issue #1 merges, the Milestone 0 issue hierarchy can be created.

## Latest accepted experiment

- None yet. No experiments run; experiments will be recorded through the
  `experiment` issue template and the experiment doctrine
  ([doctrine/evaluation-and-experiments.md](doctrine/evaluation-and-experiments.md)).

## Next recommended action

1. Review and merge the Issue #1 correction PR against the founding charter.
2. Create the Milestone 0 issue hierarchy (config loader, run identity,
   provenance capture, seed management, logging, artifact management,
   checkpoint serialization, experiment manifests, environment capture, tests,
   training smoke test).
3. After Milestone 0 is complete, begin D0 (minimal dense baseline).

## Relationship to ExpertOS

ExpertOS is the external runtime/control-plane counterpart. ExpertForge builds
its own models and does not copy ExpertOS internals; shared interfaces are
governed by versioned schemas and the ExpertOS resource contract defined here
([doctrine/deployment-and-runtime-codesign.md](doctrine/deployment-and-runtime-codesign.md)).
