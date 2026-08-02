# D0 Baseline Contract Proposal

**Issue:** #42  
**Parent:** #41  
**Status:** proposal under review; not ratified  
**Branch:** `local/42-d0-baseline-contract`

## Review objective

Convert the architecturally fixed D0 baseline into exact candidate dimensions, token accounting, optimization, data/tokenizer identities, precision bounds, evaluation procedures, and mechanically decidable acceptance/failure/kill rules before implementation work begins.

## Proposed model configurations

| Configuration | Layers | Width | Heads | Head dim | SwiGLU width | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Qualification | 8 | 256 | 4 | 64 | 768 | 19,685,888 |
| Canonical | 12 | 576 | 9 | 64 | 1,536 | 76,738,176 |

Parameter accounting is independent of future model code:

```text
V*d + L*(4*d^2 + 3*d*f + 2*d) + d
```

The count assumes tied token embedding/output weights, no biases, two RMSNorm vectors per block, one final RMSNorm vector, and no trainable RoPE parameters.

## Proposed token and update budgets

```text
context target tokens/sequence  = 1,024
devices                         = 1
microbatch sequences/device     = 4
gradient accumulation steps     = 16
global sequences/update         = 64
target tokens/update            = 65,536
```

| Configuration | Target-token budget | Optimizer updates | Warmup updates |
|---|---:|---:|---:|
| Qualification | 262,144,000 | 4,000 | 200 |
| Canonical | 2,097,152,000 | 32,000 | 1,600 |

“Processed tokens” means target tokens participating in training loss. Validation, generation, profiling warmup, and input-only lookback tokens are excluded.

## Proposed immutable external identities

### Dataset

- `HuggingFaceFW/fineweb-edu`
- revision `21974026070c0d94eb843d8eba56d02550f4c0b5`
- configuration `sample-10BT`
- ODC-By 1.0 upstream declaration
- deterministic normalization, exact-text deduplication, split, ordering, and document-termination rules are specified in Decision Record 0003.

### Tokenizer

- `EleutherAI/gpt-neox-20b`
- tokenizer commit `364ae95407723fadd1d47b023c1efb92a4d891c3`
- byte-level BPE, 50,257 tokens
- token ID 0 is BOS/EOS/UNK and the packed-stream document terminator
- `add_prefix_space = false`

Repository-local byte sizes and SHA-256 values for all tokenizer files remain a ratification blocker. Mutable `main` references are not accepted.

## Proposed optimizer and precision policy

- AdamW, betas 0.9/0.95, epsilon 1e-8.
- Weight decay 0.1 on attention and SwiGLU matrix weights only.
- Global gradient clipping at L2 norm 1.0.
- BF16 primary compute, FP32 master parameters, optimizer state, gradient accumulation, and loss reduction.
- Canonical D0 fails closed without native BF16.
- Qualification may use FP32 only as distinct, non-equivalent fallback evidence.
- FP16 is not authorized.

## Proposed acceptance boundary

Qualification requires at least 0.50 nat validation-loss improvement, zero skipped updates, no non-finite state, exact checkpoint round-trip, locked-environment uninterrupted/resumed equality, complete generation, and full systems reporting.

Canonical requires at least 1.00 nat improvement, no material late loss regression, final-window throughput at least 80% of the initial steady-state window, all qualification conditions, and a complete terminal evidence chain.

Any non-finite value, skipped optimizer update, checkpoint/resume state mismatch, repeated recovery failure, or failure to achieve 0.10 nat improvement by half-budget kills the active configuration and requires an amendment.

## Validation command

```bash
uv run python scripts/validate_d0_contract.py
uv run pytest -q tests/test_d0_contract_proposal.py
```

The independent validator checks:

- frozen architecture values;
- immutable 40-character lowercase revision identities;
- head geometry;
- SwiGLU rounding;
- exact parameter totals;
- batch and target-token arithmetic;
- token-budget/update alignment;
- finite and ordered learning rates;
- rejection of approximate placeholder language;
- explicit ratification blockers.

## Ratification blockers

This draft does not close #42. Remaining required work:

1. tokenizer file hashes and sizes;
2. content-addressed dataset source manifest;
3. formal experiment definition accepted by existing manifest contracts;
4. backward-compatible resolved-configuration schema and two valid YAML configs;
5. independently instantiated parameter-count comparison;
6. generation prompts and contamination checks;
7. permanent CI integration;
8. final rendered report;
9. `PROJECT_STATE.md` synchronization.

## Claim boundary

The proposal is a reviewable candidate contract. It does not authorize model implementation or training, and it does not claim that the selected data, hyperparameters, resource envelope, or thresholds will succeed. D0.1 and D0.2 remain blocked until #42 is ratified.
