# D0 Baseline Contract Proposal

**Issue:** #42  
**Parent:** #41  
**PR:** #43  
**Status:** Proposed, not ratified

## Current evidence state

The D0.0 proposal now freezes and validates:

- exact dense-baseline architecture, model sizes, parameter formula, batch,
  schedules, precision, evaluation, acceptance, failure, and kill semantics;
- content-addressed GPT-NeoX tokenizer and FineWeb-Edu `sample-10BT` source
  manifests;
- exact qualification and canonical YAML configurations accepted by the existing
  strict configuration resolver;
- stable, distinct specification fingerprints bound to the dataset and tokenizer
  manifest digests;
- backward compatibility for existing format-version-1 configurations;
- separate prospective qualification and canonical formal experiment contracts
  accepted by the existing version-1 `ExperimentManifest` model without
  publishing attempt evidence;
- a declarative eleven-family tensor inventory whose generic shape expansion
  agrees exactly with the independent closed-form counter and declared totals.

No model implementation, data pipeline, production training loop,
qualification run, or canonical run is authorized by this proposal.

## Frozen model profiles

| Profile | Layers | Width | Heads | Head dim | SwiGLU width | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Qualification | 8 | 256 | 4 | 64 | 768 | 19,685,888 |
| Canonical | 12 | 576 | 9 | 64 | 1,536 | 76,738,176 |

## Source identities

| Input | Revision | Files | Bytes | Manifest SHA-256 |
|---|---|---:|---:|---|
| GPT-NeoX tokenizer | `364ae95407723fadd1d47b023c1efb92a4d891c3` | 5 | 3,647,931 | `eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c` |
| FineWeb-Edu `sample-10BT` | `84e8104e779e409e2267ac60609138e3dda2cbd2` | 14 | 28,518,193,415 | `d4e7108f2455a95c725fd61fcdb4423be24d1a9a5d3c6e2e322e9df6054bf0df` |

The earlier FineWeb-Edu revision in the initial proposal is superseded because
it predates the selected sample inventory.

## Configuration identities

| Profile | Canonical config SHA-256 | Specification fingerprint |
|---|---|---|
| Qualification | `4e1bdd0ad30bf8b2e1b61f83ff9e387acdab1c0e0d1cb975e7627db95273a960` | `spec-v1-sha256-4f67b477c9d36c3aa06a4e99f0509380fdc91c672f6cc8aac3dd344880ce5cbe` |
| Canonical | `fbc699757b91ad0883fc4d30295a938df86d60f784a9fb91fb39385bbe3e6a4e` | `spec-v1-sha256-2167b1f07873c3aed6112c38c7de5cdcece6ac3b94d91acde2d605ec2decfb0b` |

Both fingerprints include `dataset.manifest` and `tokenizer.manifest` as
immutable inputs. Existing `configs/smoke.yaml` retains canonical SHA-256
`f6cf719aab809aaaf0d59b79cfba15bda7138c9138089bc7bbd4495cca087217`.

## Formal experiment binding

`experiments/d0/formal-experiment-definition-v1.json` freezes profile-specific
hypotheses, update-zero controls, fixed constraints, variables, minimum useful
effects, failure thresholds, kill criteria, and evidence requirements. Each
profile is validated using an in-memory interrupted, partial
`formal_experiment` manifest. The validation creates and publishes no run or
attempt artifacts.

## Independent parameter accounting

`experiments/d0/parameter-inventory-v1.json` declares eleven trainable tensor
families, a tied output-head alias, and six zero-parameter component classes.
The generic inventory expansion produces 74 tensor instances and 19,685,888
parameters for qualification, and 110 tensor instances and 76,738,176
parameters for canonical. Those values must equal the separate closed-form
contract derivation and the declared model totals.

## Closed ratification blockers

1. `tokenizer_file_sha256_and_byte_sizes`
2. `content_addressed_dataset_source_manifest`
3. `resolved_yaml_configs_accepted_by_existing_configuration_layer`
4. `formal_experiment_definition_accepted_by_existing_manifest_contract`
5. `independent_parameter_accounting_executable_and_tests`

## Remaining ratification blockers

1. `committed_generation_prompts_and_contamination_checks`
2. `contract_validation_command_and_CI_gate`
3. `rendered_review_report`
4. `PROJECT_STATE_synchronization`

## Validation commands

```bash
uv run python scripts/validate_d0_source_manifests.py
uv run python scripts/validate_d0_config_binding.py
uv run python scripts/validate_d0_experiment_definition.py
uv run python scripts/validate_d0_parameter_inventory.py
uv run python scripts/validate_d0_contract.py
uv run pytest -q \
  tests/test_d0_source_manifests.py \
  tests/test_d0_config_binding.py \
  tests/test_d0_experiment_definition.py \
  tests/test_d0_parameter_inventory.py \
  tests/test_d0_contract_proposal.py
```

## Next dependency-ordered action

Commit the fixed generation prompts and mechanically check their contamination
status against the frozen dataset-source contract. Model implementation and
training remain unauthorized.
