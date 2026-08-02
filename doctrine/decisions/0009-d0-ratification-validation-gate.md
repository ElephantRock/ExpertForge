# Decision Record 0009 — Permanent D0 Ratification Validation Gate

**Status:** Proposed binding amendment within D0.0  
**Date:** 2026-08-03  
**Issue:** #42  
**Parent:** #41  
**Amends:** Decisions 0004–0008 validation evidence

## Context

D0.0 accumulated separate validators for the baseline contract, immutable source
manifests, resolved configurations and specification fingerprints, prospective
formal experiment definitions, declarative parameter accounting, and fixed
generation prompts with contamination-check mechanics. Running those commands
individually was useful while each evidence tranche was under construction, but
it did not provide one permanent fail-closed entry point or guarantee that CI
would execute every validator in an exact, reviewable order.

A permanent gate must detect omission, reordering, replacement, report-shape
changes, exceptions, and stale ratification-blocker state. It must preserve the
existing claim boundary: validation of prospective artifacts is not corpus
processing, model implementation, training, generation, or a terminal attempt.

## Decision

### One permanent command

The authoritative D0.0 validation command is:

```bash
uv run --locked python -m scripts.validate_d0_ratification
```

The module executes exactly six validators in this order:

1. `baseline_contract`;
2. `source_manifests`;
3. `configuration_bindings`;
4. `formal_experiment_definition`;
5. `parameter_inventory`;
6. `generation_prompts_and_contamination`.

The registry is immutable within this contract version. Missing, additional,
duplicated, or reordered entries fail validation.

### Fail-closed report contract

Each validator must return a mapping report or raise an exception. The aggregate
gate canonicalizes each report as sorted compact JSON, records its SHA-256, and
then hashes the ordered list of validator names and report digests. This makes
the aggregate output deterministic while leaving each underlying validator the
authority for its own artifact semantics.

The gate also requires the machine-readable contract to enumerate exactly these
remaining blockers:

```text
rendered_review_report
PROJECT_STATE_synchronization
```

Any stale reintroduction of the validation-gate blocker, any premature removal
of another blocker, or any ordering drift fails closed.

### CI integration

The quality job contains a dedicated step named:

```text
Validate D0 ratification bundle
```

It runs the authoritative command after formatting, Ruff, and strict mypy, and
before the complete fast CPU suite. Consequently, downstream integration and
smoke/recovery jobs cannot start unless the aggregate D0 contract gate passes.

### Evidence boundary

The aggregate report explicitly states:

```text
actual_corpus_scan_completed: false
actual_corpus_scan_stage: D0.1_preflight_before_packing_or_training
material_execution_authorized: false
```

The gate validates the committed prompt/contamination protocol but does not scan
the selected FineWeb-Edu corpus. It creates no model, packed data, checkpoint,
generated output, run identity, or terminal experiment manifest.

## Validation coverage

Permanent tests verify:

- the committed six-validator bundle succeeds;
- every validator runs exactly once in the declared order;
- registry omission and reordering fail;
- an underlying validator failure stops all later validators;
- non-mapping reports fail;
- aggregate report identity is deterministic;
- stale blocker state fails;
- the command entry point succeeds;
- CI contains exactly one named invocation of the authoritative command.

## Alternatives rejected

### Keep a documentation-only list of commands

Rejected because documentation cannot prevent omission or ordering drift and
cannot block downstream CI jobs.

### Invoke validators through subprocesses

Rejected because it would duplicate environment setup, weaken typed composition,
and make deterministic report aggregation harder. The permanent module imports
the established validation functions directly while retaining each command's
standalone entry point.

### Add the gate only to tests

Rejected because a test could be deselected or renamed without preserving a
clearly visible permanent CI step.

### Treat successful contract validation as execution authorization

Rejected because D0.0 remains a ratification stage. D0.1 corpus preflight and
later implementation evidence are separate obligations.

## Consequences

- D0.0 has one stable fail-closed validation command.
- CI visibly executes that command before fast, integration, and smoke gates.
- Every current ratification validator is covered by deterministic registry and
  report identities.
- The `contract_validation_command_and_CI_gate` blocker is closed.
- Two blockers remain: the rendered review report and `PROJECT_STATE.md`
  synchronization.
- PR #43 remains draft, and no model implementation or material run is
  authorized.

## Evidence

CI run 272 passed on exact implementation head
`187c1dc6ea0aafda91999b8d5bda8f94943fd09e`, including the dedicated aggregate
D0 gate, formatting, Ruff, strict mypy, the complete fast CPU suite, portable
CPU integration, the locked smoke tier, and the one-command interruption and
recovery gate.

## Reversal conditions

This decision may be superseded only by an amendment that:

1. identifies the validator being added, removed, renamed, or reordered;
2. explains the changed ratification dependency;
3. updates the aggregate registry, mutation tests, CI invocation, machine
   contract, rendered review evidence, and project state together;
4. preserves an explicit non-execution boundary unless a later milestone has
   separately authorized material execution.
