# D0 Formal Experiment Definition Report

**Issue:** #42  
**Parent:** #41  
**PR:** #43  
**Status:** D0.0 draft evidence; formal-definition tranche complete

## Result

Qualification and canonical D0 have separate prospective formal experiment
definitions accepted by the existing version-1 `ExperimentManifest` model.
Validation is in-memory and non-publishing: no run directory, attempt artifact,
completed status, result, decision, or execution evidence is created.

## Profile contracts

| Profile | Control | Minimum useful effect | Specification fingerprint |
|---|---|---:|---|
| Qualification | Same frozen model at update zero | 0.50 nat fixed-validation loss improvement | `spec-v1-sha256-4f67b477c9d36c3aa06a4e99f0509380fdc91c672f6cc8aac3dd344880ce5cbe` |
| Canonical | Same frozen model at update zero | 1.00 nat fixed-validation loss improvement | `spec-v1-sha256-2167b1f07873c3aed6112c38c7de5cdcece6ac3b94d91acde2d605ec2decfb0b` |

The independent variable is optimizer-update count and equivalent accumulated
target-token exposure. Dependent variables cover validation loss, perplexity,
non-finite events, skipped updates, recovery equality, throughput, peak device
and host memory, checkpoint read/write latency, and generation completion.

## Manifest compatibility fixture

Each profile is validated with a synthetic, fixed identity and this lifecycle:

```text
classification:     formal_experiment
maturity_stage:     D0
research_family:    none/not-applicable
status:             interrupted
outcome_diagnostic: handled_interruption
evidence.status:    partial
```

The exact expected missing-evidence set is:

```text
checkpoint_artifact_missing
configuration_artifact_missing
generated_output_artifact_missing
provenance_artifact_missing
telemetry_artifact_missing
```

Dataset, tokenizer, model, training-budget, and evaluation-protocol identities
are populated from the frozen D0 contract, resolved YAML, and fingerprint
records. Canonical manifest bytes must round-trip through the production parser.

## Validation coverage

The validator rejects:

- terminal or publication claims;
- missing formal experiment fields;
- qualification/canonical fingerprint crossover;
- source, config, budget, model, or threshold drift;
- added, removed, duplicated, reordered, or untracked constraints and variables;
- placeholder language;
- malformed fingerprint records;
- any disagreement with the existing configuration-binding validator;
- any synthetic artifact or publication path.

## Evidence

CI run 222 passed on exact head `a0b616d0b198345a0c0645ad8d609340967b6172`,
including formatting, Ruff, strict mypy, the complete fast CPU suite, portable
CPU integration, the locked smoke tier, and the one-command
interruption/recovery gate.

Commands:

```bash
uv run python scripts/validate_d0_experiment_definition.py
uv run pytest -q tests/test_d0_experiment_definition.py
```

## Closed blocker

```text
formal_experiment_definition_accepted_by_existing_manifest_contract
```

The declarative tensor-inventory/parameter-accounting and fixed
prompt/contamination blockers closed in subsequent tranches.

## Current remaining dependency order

1. permanent D0 contract validation command and CI gate;
2. final rendered review report;
3. `PROJECT_STATE.md` synchronization.

No model implementation, data processing, qualification run, or canonical run
is authorized.
