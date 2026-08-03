# D0 Baseline Contract Proposal

**Issue:** #42  
**Parent:** #41  
**PR:** #43  
**Status:** Ratification-ready prospective contract; not yet ratified

## Current evidence state

The D0.0 proposal now freezes and validates:

- exact dense-baseline architecture, model sizes, parameter formula, batch,
  schedules, precision, evaluation, acceptance, failure, and kill semantics;
- content-addressed GPT-NeoX tokenizer and FineWeb-Edu `sample-10BT` source
  manifests;
- exact qualification and canonical resolved configurations accepted by the
  strict configuration resolver;
- stable, distinct specification fingerprints bound to the dataset and tokenizer
  manifest digests;
- separate prospective qualification and canonical formal experiment contracts
  accepted by the production experiment-manifest model without publishing
  attempt evidence;
- a declarative eleven-family tensor inventory whose generic shape expansion
  agrees exactly with the independent closed-form counter and declared totals;
- an ordered eight-prompt generation set with raw and canonical SHA-256
  identities and a deterministic zero-hit contamination protocol;
- a content-addressed rendered final review report;
- a content-addressed `PROJECT_STATE.md` index synchronized to the D0 program;
- one permanent fail-closed command covering eight authoritative validators.

The machine contract contains zero preparation blockers. This means the evidence
package is ready for final acceptance review. It does not mean Issue #42 is
closed or that D0.1, D0.2, corpus processing, model implementation, or material
execution is authorized.

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

## Configuration identities

| Profile | Canonical config SHA-256 | Specification fingerprint |
|---|---|---|
| Qualification | `4e1bdd0ad30bf8b2e1b61f83ff9e387acdab1c0e0d1cb975e7627db95273a960` | `spec-v1-sha256-4f67b477c9d36c3aa06a4e99f0509380fdc91c672f6cc8aac3dd344880ce5cbe` |
| Canonical | `fbc699757b91ad0883fc4d30295a938df86d60f784a9fb91fb39385bbe3e6a4e` | `spec-v1-sha256-2167b1f07873c3aed6112c38c7de5cdcece6ac3b94d91acde2d605ec2decfb0b` |

Both fingerprints include `dataset.manifest` and `tokenizer.manifest` as
immutable inputs.

## Formal experiment and parameter binding

`experiments/d0/formal-experiment-definition-v1.json` freezes profile-specific
hypotheses, update-zero controls, fixed constraints, variables, minimum useful
effects, failure thresholds, kill criteria, budgets, replication rules, and
evidence requirements.

`experiments/d0/parameter-inventory-v1.json` declares eleven trainable tensor
families, a tied output-head alias, and six zero-parameter component classes.
The generic inventory expansion produces 74 tensor instances and 19,685,888
parameters for qualification, and 110 tensor instances and 76,738,176
parameters for canonical.

## Generation prompts and contamination

`experiments/d0/generation-prompts-v1.json` freezes eight ordered prompts, probe
substrings, hashes, decoding settings, and exact comparison rules.

| Identity | SHA-256 |
|---|---|
| Raw prompt manifest | `851934e4a49210b515967fad51308038dac1e7bc449774ca603f6bd9f47ccf3e` |
| Canonical prompt payload | `c52ef9f4420ff8160fd5f212370f46cefde71033ac38431bdad2878640070e51` |

The source-bound corpus scan remains a mandatory D0.1 preflight and has not been
executed.

## Final review and project state

| Artifact | Identity |
|---|---|
| Final review report | `c41573a01f7f0568221b7e12edf87a50dd2ab54f39e9d53de37af68a088060b9` |
| Synchronized `PROJECT_STATE.md` | `6df545403d763304fc084d02be2f8a8effb5672c1983233c92216f8545a25f83` |

The final review disposition accepted the prospective contract for project-state
synchronization. The synchronized state now identifies D0 as active, D0.0 as
ratification-ready, and D0.1/D0.2 as blocked until explicit acceptance, merge,
and Issue #42 closure.

## Permanent ratification gate

The authoritative command is:

```bash
uv run --locked python -m scripts.validate_d0_ratification
```

It executes these validators in fixed order:

1. `baseline_contract`
2. `source_manifests`
3. `configuration_bindings`
4. `formal_experiment_definition`
5. `parameter_inventory`
6. `generation_prompts_and_contamination`
7. `final_review_report`
8. `project_state`

CI invokes the command in a dedicated `Validate D0 ratification bundle` step
after strict typing and before the full fast suite. Integration and smoke jobs
cannot start if the aggregate gate fails.

## Ratification blocker state

```text
remaining_ratification_blockers: []
remaining_ratification_blocker_count: 0
```

All repository preparation blockers are closed. Ratification still requires
final exact-head acceptance, merge of PR #43, and closure of Issue #42.

## Claim boundary

No model implementation, selected-corpus processing, source-bound contamination
result, qualification run, or canonical run is represented by this proposal.
Closure of Issue #42 authorizes D0.1 and D0.2 only; D0.5 remains the first
material training authorization.
