# 0011 — Synchronize D0.0 project state and close preparation blockers

**Status:** Accepted for final review  
**Issue:** #42  
**Parent:** #41  
**PR:** #43

## Decision

Synchronize `PROJECT_STATE.md` to the D0 controlled dense-baseline program,
content-address that index with `experiments/d0/project-state-v1.json`, validate
it with `scripts/validate_d0_project_state.py`, add it as the eighth and final
step of the permanent D0 ratification command, and require the prospective
contract to contain an empty `ratification_blockers` list.

The synchronized state is **ratification-ready, not yet ratified**. D0.1 and
D0.2 remain unauthorized until explicit acceptance, merge of PR #43, and
closure of Issue #42. D0.5 remains the first authorization for material
engineering training.

## Context

D0.0 had already frozen and validated the architecture, source identities,
resolved configurations, specification fingerprints, formal experiment
contract, parameter accounting, generation prompts, contamination protocol,
permanent validation command, and content-addressed final review. The remaining
preparation blocker was that `PROJECT_STATE.md` still described the superseded
Milestone 0 transition rather than the active D0 program and its authorization
boundary.

A prose-only edit would not meet ExpertForge's evidence model. The current state
index influences what work is considered active and authorized, so silent drift
between that index and the machine contract must fail CI.

## Consequences

1. `PROJECT_STATE.md` identifies D0 as the active program and D0.0 as
   ratification-ready pending acceptance, merge, and issue closure.
2. The state index records the exact model profiles, token/update budgets,
   immutable source/configuration/prompt/review identities, claim boundary, and
   D0 dependency graph.
3. `experiments/d0/project-state-v1.json` binds the exact UTF-8 bytes of
   `PROJECT_STATE.md` and repeats the immutable evidence identities.
4. `scripts/validate_d0_project_state.py` rejects state digest drift, evidence
   crossover, nonterminal blocker state, missing headings or required claims,
   and unauthorized execution claims.
5. `scripts/validate_d0_ratification.py` executes eight validators in fixed
   order and requires zero preparation blockers.
6. The final review report remains byte-for-byte immutable. Its manifest
   preserves the historical disposition that project-state synchronization was
   the final blocker, while its validator now also confirms the current terminal
   zero-blocker contract state.
7. Closing preparation blockers does not itself ratify the contract or authorize
   D0.1/D0.2. Those transitions occur only after explicit acceptance, merge, and
   Issue #42 closure.

## Rejected alternatives

### Declare D0.0 ratified on the feature branch

Rejected because Issue #42 is still open and PR #43 is not merged. The project
state must distinguish a complete evidence package from an accepted contract on
`main`.

### Leave `PROJECT_STATE.md` outside the permanent validator

Rejected because the state index could then silently regress to stale milestone,
blocker, or authorization claims while the machine contract remained valid.

### Rewrite the content-addressed final review after synchronization

Rejected because the review is a historical acceptance artifact whose exact
bytes and digest are already frozen. The transition is represented by the new
project-state manifest and validator instead.

### Authorize D0.1 and D0.2 when the blocker list becomes empty

Rejected because zero preparation blockers means the package is ready for final
acceptance. Authorization is explicitly tied to Issue #42 closure.

## Evidence boundary

This decision records no corpus processing, contamination result, model
implementation, training, generation, checkpoint, qualification attempt, or
canonical attempt. The source-bound contamination scan remains a D0.1 preflight
before packing or training.
