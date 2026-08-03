# Decision Record 0006 — D0 Formal Experiment Definition

**Status:** Proposed binding amendment within D0.0  
**Date:** 2026-08-03  
**Issue:** #42  
**Parent:** #41  
**Amends:** Decision Record 0003 experiment-contract representation only

## Context

The D0 execution contract and its qualification/canonical configurations are
frozen, but the repository's formal experiment-manifest contract has not yet
accepted a D0 question, hypothesis, control, variables, thresholds, and evidence
requirements.

`ExperimentManifest` is deliberately a terminal-attempt record. A completed
manifest requires complete attempt evidence and binds concrete artifacts. D0.0
has not authorized model implementation, data processing, or training, so
publishing a fabricated completed manifest would incorrectly claim execution.

## Decision

### Prospective definition

The formal D0 experiment contract is committed separately as:

```text
experiments/d0/formal-experiment-definition-v1.json
```

It is explicitly marked `prospective_not_executed` and binds Issue #42, parent
Issue #41, draft PR #43, the version-1 experiment-manifest schema, and the exact
qualification and canonical specification fingerprints.

Each profile freezes:

- a falsifiable hypothesis relative to update zero;
- the update-zero control under the identical specification fingerprint;
- sorted fixed constraints;
- optimizer-update count and equivalent accumulated target-token exposure as
  the independent variable;
- sorted loss, systems, recovery, checkpoint, and generation dependent
  variables;
- profile-specific minimum useful effect;
- canonical JSON encodings of profile/common failure thresholds and kill
  criteria;
- checkpoint and generated-output evidence requirements;
- explicit limitations that the definition is prospective and makes no
  training-seed variance claim.

### Existing manifest-contract validation

`python scripts/validate_d0_experiment_definition.py` loads the formal
definition, baseline contract, resolved configurations, and committed
fingerprint records. For each profile it constructs a strict
`ExperimentManifest` entirely in memory with:

```text
classification:     formal_experiment
maturity_stage:     D0
research_family:    none/not-applicable
status:             interrupted
outcome_diagnostic: handled_interruption
evidence:           partial
```

The synthetic validation identity uses fixed path-safe IDs and a fixed UTC
timestamp. Dataset, tokenizer, model, training-budget, and evaluation-protocol
records are populated from authoritative D0 artifacts. Configuration,
provenance, telemetry, checkpoint, and generated-output artifacts remain absent,
and the manifest's derived missing-evidence set must state those absences
exactly.

The canonical manifest bytes must parse back to the identical typed model. The
validator imports no `ArtifactStore` or `ManifestGenerator`, invokes no publish
operation, creates no run directory, and writes no terminal attempt artifact.

### Threshold representation

The existing manifest schema represents `failure_threshold` and
`kill_criterion` as non-blank strings. D0 stores those values as compact,
sorted-key JSON strings generated from the machine-readable baseline contract.
This preserves every numerical and Boolean threshold without paraphrase or
reinterpretation.

## Alternatives rejected

### Publish a synthetic completed manifest

Rejected because completed manifests require complete real attempt evidence.
Synthetic artifacts would falsely imply that D0 execution occurred.

### Extend the version-1 manifest schema for prospective definitions

Rejected for D0.0. The existing formal-classification fields are sufficient to
validate the scientific contract. A separate prospective definition preserves
the terminal-attempt semantics of `ExperimentManifest`.

### Store thresholds only as prose

Rejected because prose can omit, round, or reinterpret individual failure and
kill values. Canonical JSON strings preserve exact contract content within the
existing field types.

### Treat qualification and canonical as one interchangeable profile

Rejected because their model identities, token budgets, schedules, minimum
useful effects, failure thresholds, and specification fingerprints differ.

## Evidence

CI run 212 passed formatting, Ruff, strict mypy, the complete fast CPU suite,
portable CPU integration, the locked smoke tier, and the one-command
interruption/recovery gate on the formal-definition implementation head.

## Consequences

- The D0 question and decision boundary are machine-readable before execution.
- Both profiles validate against the production formal-manifest model without
  creating run evidence.
- Any hypothesis, control, variable, constraint, threshold, source, config, or
  fingerprint drift fails validation.
- The formal-experiment-definition ratification blocker is closed.
- Five D0.0 ratification blockers remain.
- No model implementation, data processing, qualification run, or canonical run
  is authorized by this decision.

## Reversal conditions

This decision may be superseded only through a dedicated amendment that:

1. identifies a scientific field that the current formal contract cannot
   represent;
2. preserves the terminal-attempt meaning of published experiment manifests;
3. defines migration and comparison effects for both D0 profiles;
4. revalidates every dependent contract, configuration, fingerprint, and CI
   gate before execution begins.
