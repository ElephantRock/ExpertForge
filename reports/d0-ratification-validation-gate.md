# D0 Ratification Validation Gate Report

**Issue:** #42  
**Parent:** #41  
**PR:** #43  
**Status:** D0.0 draft evidence; permanent validation-gate tranche complete

## Result

D0.0 now has one permanent fail-closed validation command:

```bash
uv run --locked python -m scripts.validate_d0_ratification
```

The command composes every authoritative D0.0 validator, enforces an exact
registry and execution order, records deterministic per-validator report
SHA-256 values, computes an aggregate report identity, and rejects stale
ratification-blocker state.

The command validates prospective contract evidence only. It does not process
the selected corpus, instantiate a model, train, generate output, or publish an
experiment attempt.

## Validator registry

| Order | Validator |
|---:|---|
| 1 | `baseline_contract` |
| 2 | `source_manifests` |
| 3 | `configuration_bindings` |
| 4 | `formal_experiment_definition` |
| 5 | `parameter_inventory` |
| 6 | `generation_prompts_and_contamination` |

The gate rejects missing, added, duplicated, renamed, or reordered validators.
An exception from any validator stops execution before later validators run.
Each validator must return a mapping report.

## Deterministic output

For each validator, the gate:

1. canonicalizes its report as sorted compact UTF-8 JSON;
2. computes a SHA-256 digest of those bytes;
3. records the validator name and digest in registry order.

It then computes an aggregate SHA-256 over that ordered digest inventory. The
per-validator and aggregate identities are runtime report evidence rather than
hard-coded constants; mutation tests require repeat executions over identical
inputs to produce identical output.

## CI binding

The quality job contains a dedicated step:

```text
Validate D0 ratification bundle
```

with the exact invocation:

```bash
uv run --locked python -m scripts.validate_d0_ratification > /dev/null
```

The step runs after formatting, Ruff, and strict mypy and before the complete
fast CPU suite. CPU integration and the locked smoke/recovery gate depend on the
quality job, so they cannot proceed when the D0 contract bundle is invalid.

## Mutation coverage

`tests/test_d0_ratification_gate.py` verifies:

- the committed aggregate bundle succeeds;
- six validators execute exactly once in order;
- omission and reversal of the registry fail;
- validator failure prevents later execution;
- a non-mapping report fails;
- aggregate report identity is deterministic;
- stale blocker state fails;
- the command entry point succeeds;
- CI contains exactly one named permanent invocation.

The existing contract proposal test now requires exactly two remaining blockers.

## Evidence boundary

The aggregate report preserves these explicit claims:

```text
actual_corpus_scan_completed: false
actual_corpus_scan_stage: D0.1_preflight_before_packing_or_training
material_execution_authorized: false
```

The selected FineWeb-Edu source remains unscanned for prompt contamination. D0.1
must execute the source-bound scan and produce a content-addressed zero-hit
report before packing or training.

## Validation evidence

CI run 272 passed on exact implementation head
`187c1dc6ea0aafda91999b8d5bda8f94943fd09e` with:

- formatting and Ruff;
- strict mypy;
- the dedicated aggregate D0 ratification step;
- the complete fast CPU suite;
- package, configuration, lockfile, and repository-policy checks;
- portable CPU integration;
- the locked smoke tier;
- the one-command interruption and recovery gate.

## Closed blocker

```text
contract_validation_command_and_CI_gate
```

## Remaining dependency order

1. final rendered review report;
2. `PROJECT_STATE.md` synchronization.

PR #43 must remain draft. No model implementation, data processing,
qualification run, or canonical run is authorized.
