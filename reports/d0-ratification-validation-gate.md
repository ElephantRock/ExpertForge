# D0 Ratification Validation Gate Report

**Issue:** #42  
**Parent:** #41  
**PR:** #43  
**Status:** Permanent terminal D0.0 validation gate

## Result

D0.0 has one authoritative offline, non-mutating, fail-closed command:

```bash
uv run --locked python -m scripts.validate_d0_ratification
```

The command composes every authoritative D0.0 validator, enforces an exact
registry and execution order, records deterministic per-validator report
SHA-256 values, computes an aggregate report identity, and requires an empty
ratification-blocker list.

The command validates prospective contract evidence only. It does not process
the selected corpus, instantiate a model, train, generate output, or publish an
experiment attempt.

## Terminal validator registry

| Order | Validator |
|---:|---|
| 1 | `baseline_contract` |
| 2 | `source_manifests` |
| 3 | `configuration_bindings` |
| 4 | `formal_experiment_definition` |
| 5 | `parameter_inventory` |
| 6 | `generation_prompts_and_contamination` |
| 7 | `final_review_report` |
| 8 | `project_state` |

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
quality job, so they cannot proceed when the D0 evidence bundle is invalid.

## Mutation coverage

`tests/test_d0_ratification_gate.py` verifies:

- the committed aggregate bundle succeeds;
- eight validators execute exactly once in order;
- omission and reversal of the registry fail;
- validator failure prevents later execution;
- a non-mapping report fails;
- aggregate report identity is deterministic;
- reintroduction of any blocker fails;
- the command entry point succeeds;
- CI contains exactly one named permanent invocation.

The component tests separately protect source identities, configuration binding,
formal experiment definitions, parameter inventory, prompt contamination rules,
final review identity, and synchronized project state.

## Terminal blocker and authorization state

The aggregate report preserves these explicit claims:

```text
remaining_ratification_blockers: []
remaining_ratification_blocker_count: 0
actual_corpus_scan_completed: false
actual_corpus_scan_stage: D0.1_preflight_before_packing_or_training
d0_1_authorized: false
d0_2_authorized: false
material_execution_authorized: false
```

Zero preparation blockers means the repository package is ready for final human
acceptance. It does not mean Issue #42 is closed or that the contract has been
ratified on `main`.

## Project-state binding

The eighth validator binds `PROJECT_STATE.md` to SHA-256
`6df545403d763304fc084d02be2f8a8effb5672c1983233c92216f8545a25f83`
and cross-checks the immutable dataset, tokenizer, specification, prompt, and
final-review identities. It also rejects premature authorization claims.

## Evidence boundary

The selected FineWeb-Edu source remains unscanned for prompt contamination. D0.1
must execute the source-bound scan and produce a content-addressed zero-hit
report before packing or training.

No model implementation, data processing, qualification run, or canonical run
is authorized by this gate. D0.1 and D0.2 become authorized only after explicit
acceptance, merge of PR #43, and closure of Issue #42.
