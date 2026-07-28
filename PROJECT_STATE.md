# PROJECT_STATE.md

**Index of current project state.** This file is an index, not a substitute for
detailed issue and pull-request records. When this file conflicts with the
underlying issues/PRs or with `main`, those prevail.

Last updated: bootstrap commit (direct to `main`, founding exception).

## Current milestone

- **Milestone 0 — Project foundation.** Status: bootstrap in progress.
- Next milestone (Milestone 1) to be defined by issue once the foundation is on
  `main` and reviewed against the founding charter.

## Active baseline

- **Baseline:** ExpertForge bootstrap foundation on `main`.
- **State:** README, AGENTS, PROJECT_STATE, CONTRIBUTING, doctrine (charter,
  collaboration, model-lineage, decision 0001), PR template, and four issue
  templates.

## Active issues

- None yet. The first post-bootstrap action is a review of the bootstrap commit
  against the founding charter, followed by creation of the Milestone 0 issue
  hierarchy.

## Open pull requests

- None. The bootstrap is committed directly to `main` under the founding
  exception (doctrine §16 / decision record 0001).

## Known blockers

- None at bootstrap.

## Latest accepted experiment

- None yet. ExpertForge holds no experiments at bootstrap; experiments will be
  recorded through the `experiment` issue template and linked here as they are
  accepted.

## Next recommended action

1. Review the bootstrap commit against [doctrine/charter.md](doctrine/charter.md).
2. Create the Milestone 0 issue hierarchy (charter-conformance review, schema
   registry scaffolding, ExpertOS interface inventory).
3. Open the first implementation issues using the templates in
   `.github/ISSUE_TEMPLATE/`.

## Relationship to ExpertOS

ExpertOS is the external runtime/control-plane counterpart. ExpertForge does not
import ExpertOS internals; shared interfaces are governed by versioned schemas
defined here. See [doctrine/charter.md](doctrine/charter.md) and
[AGENTS.md](AGENTS.md) §2.
