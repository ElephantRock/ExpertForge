# D0 Configuration Binding Report

**Issue:** #42  
**Parent:** #41  
**PR:** #43  
**Status:** D0.0 draft evidence; configuration-binding tranche complete

## Result

The qualification and canonical D0 configurations resolve through the existing
strict, frozen configuration layer. The additive `d0` section remains optional
for legacy configurations and is excluded from canonical serialization when
absent, preserving existing behavioral fingerprints.

CI run 200 passed the quality/fast tier, strict typing, portable CPU integration,
locked smoke tier, and one-command interruption/recovery gate on the final
configuration-binding head. The machine-readable contract closes the resolved
configuration blocker.

No model implementation, data pipeline, qualification run, or canonical run is
authorized by this report.

## Configurations

| Profile | Config | Canonical config SHA-256 | Specification fingerprint |
|---|---|---|---|
| Qualification | `configs/d0/qualification.yaml` | `4e1bdd0ad30bf8b2e1b61f83ff9e387acdab1c0e0d1cb975e7627db95273a960` | `spec-v1-sha256-4f67b477c9d36c3aa06a4e99f0509380fdc91c672f6cc8aac3dd344880ce5cbe` |
| Canonical | `configs/d0/canonical.yaml` | `fbc699757b91ad0883fc4d30295a938df86d60f784a9fb91fb39385bbe3e6a4e` | `spec-v1-sha256-2167b1f07873c3aed6112c38c7de5cdcece6ac3b94d91acde2d605ec2decfb0b` |

Each fingerprint includes these immutable inputs:

| Name | SHA-256 |
|---|---|
| `dataset.manifest` | `d4e7108f2455a95c725fd61fcdb4423be24d1a9a5d3c6e2e322e9df6054bf0df` |
| `tokenizer.manifest` | `eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c` |

## Compatibility boundary

`format_version` remains `1`. Existing configuration sections and defaults are
unchanged. `ConfigRoot.d0` is optional and omitted from `model_dump(mode="json")`
when absent. The existing `configs/smoke.yaml` canonical SHA-256 remains:

`f6cf719aab809aaaf0d59b79cfba15bda7138c9138089bc7bbd4495cca087217`

## Binding coverage

The validator compares each resolved configuration with the machine-readable
D0 contract across:

- immutable dataset and tokenizer identities;
- architecture and sequence semantics;
- model dimensions, head geometry, and declared parameter totals;
- microbatch, accumulation, global batch, and target-token arithmetic;
- optimizer, initialization, schedule, and precision policy;
- seed and recovery semantics;
- validation, generation, throughput, and inference protocol;
- acceptance, failure, and kill thresholds;
- legacy core fields used by the existing runtime substrate.

It also verifies each committed fingerprint record independently.

## Commands

```bash
uv run python scripts/validate_d0_config_binding.py
uv run pytest -q tests/test_d0_config_binding.py
```

## Closed blocker

`resolved_yaml_configs_accepted_by_existing_configuration_layer`

## Current remaining blockers

1. `independent_parameter_accounting_executable_and_tests`
2. `committed_generation_prompts_and_contamination_checks`
3. `contract_validation_command_and_CI_gate`
4. `rendered_review_report`
5. `PROJECT_STATE_synchronization`

The formal experiment-definition blocker closed in the subsequent tranche.

## Next dependency-ordered action

Commit the declarative tensor inventory and independent executable parameter
comparison. Model implementation remains unauthorized.
