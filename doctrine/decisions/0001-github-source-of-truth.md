# Decision Record 0001 — GitHub as Source of Truth

**Status:** Accepted
**Date:** Bootstrap (founding)

## Context

ExpertForge is a complete language-model research and engineering project,
distinct from ExpertOS (the external runtime/control-plane counterpart). It is
worked on by multiple contributors — a web assistant, a local assistant, and a
human operator — operating from different contexts (chat, local working trees,
assistant memory). Without a single source of truth, these contexts will
diverge, and it will become unclear which state is authoritative.

A bootstrap mechanism is also needed: the first commit cannot follow the
issue→branch→PR→review→merge workflow because that workflow does not yet exist
in the repository.

## Alternatives considered

1. **Treat the local working tree as canonical.** Rejected: local state diverges
   across machines and is invisible to other contributors; unpushed work is
   fragile and not shared.
2. **Treat chat agreement as canonical.** Rejected: chat is ephemeral, not
   reviewable after the fact, and not traceable to commits (doctrine §11,
   charter §4).
3. **Treat assistant memory as canonical.** Rejected: memory is mutable, not
   auditable, and not visible to all contributors.
4. **GitHub `main` as the sole canonical accepted state, with a one-time
   bootstrap exception.** Selected.

## Decision

The `main` branch of `ElephantRock/ExpertForge` is the sole canonical accepted
state of the project. Chat, drafts, local working-tree changes, unpushed
commits, and assistant memory are provisional until merged into `main`. When
`main` conflicts with any other statement, `main` prevails.

The bootstrap commit that establishes `AGENTS.md`, `CONTRIBUTING.md`, doctrine,
issue templates, and pull-request rules may be committed directly to `main` as a
one-time founding exception (doctrine §16). After that, all substantive work
uses the normal issue→branch→PR→review→merge workflow.

## Evidence

- Repository `ElephantRock/ExpertForge` exists, private, default branch `main`,
  initialized with README (verified at creation, HTTP 201).
- The bootstrap commit is the only direct-to-`main` commit; subsequent work is
  expected to use branches and PRs.

## Consequences

- Any contributor can determine authoritative state by reading `main`.
- Work not on `main` is explicitly provisional and may be discarded without
  violating a commitment.
- Disagreements are resolved by repository evidence or experiment, not by
  appealing to chat or memory (doctrine §12).
- The bootstrap commit must be reviewed against the founding charter as the
  first post-bootstrap action.

## Reversal conditions

This decision would be reconsidered only if the project moves off GitHub
entirely, or if a stronger provenance substrate (e.g., content-addressed
artifact store with signed state) is adopted and ratified by a new decision
record. A migration would itself follow the normal change workflow from a new
main, and would require human-operator authority over repository access
(doctrine §6).
