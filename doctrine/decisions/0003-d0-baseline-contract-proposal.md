# Decision Record 0003 — D0 Baseline Execution Contract Proposal

**Status:** Proposed; not ratified
**Date:** 2026-08-02
**Issue:** #42
**Parent:** #41
**Supersedes:** none
**Depends on:** Decision Record 0002

## Purpose

Freeze one reviewable D0 execution contract before production model code merges or any material training run begins. This proposal is intentionally contract-only. It does not authorize D0.1, D0.2, D0.5, or D0.6 until the ratification blockers at the end of this record are closed and Issue #42 is accepted.

## Fixed architecture

Both configurations use a decoder-only Transformer with exact causal multi-head self-attention, pre-norm RMSNorm, full-head-dimension RoPE, SwiGLU feed-forward blocks, tied token embedding/output weights, no learned position embeddings, no biases, no dropout, and next-token prediction.

### Qualification configuration

| Field | Value |
|---|---:|
| Model identifier | `d0-qualification-19m-v1` |
| Tokenizer vocabulary | 50,257 |
| Context length | 1,024 input tokens |
| Transformer blocks | 8 |
| Model width | 256 |
| Attention heads | 4 |
| Head dimension | 64 |
| SwiGLU intermediate width | 768 |
| Trainable parameters | 19,685,888 |

### Canonical configuration

| Field | Value |
|---|---:|
| Model identifier | `d0-canonical-76m-v1` |
| Tokenizer vocabulary | 50,257 |
| Context length | 1,024 input tokens |
| Transformer blocks | 12 |
| Model width | 576 |
| Attention heads | 9 |
| Head dimension | 64 |
| SwiGLU intermediate width | 1,536 |
| Trainable parameters | 76,738,176 |

The intermediate width rule is `ceil_to_multiple((8 / 3) * model_width, 256)`. The qualification width therefore rounds from 682.666… to 768; the canonical width is exactly 1,536.

## Parameter accounting

With tied token embedding/output weights, no biases, two RMSNorm weight vectors per block, and one final RMSNorm weight vector:

```text
embedding              = vocabulary_size * model_width
one attention block    = 3*d*d + d*d
one SwiGLU block       = 2*d*f + f*d
one block normalization = 2*d
final normalization    = d

total = V*d + L*(4*d^2 + 3*d*f + 2*d) + d
```

RoPE has no trainable parameters. The output head has no independent parameters because it is tied to the token embedding matrix.

## Dataset identity proposal

- Repository: `HuggingFaceFW/fineweb-edu`
- Immutable upstream revision: `21974026070c0d94eb843d8eba56d02550f4c0b5`
- Configuration: `sample-10BT`
- Upstream license declaration: ODC-By 1.0; source content remains subject to source-publisher rights and Common Crawl terms.
- Consumed fields: `id`, `text`, `url`, `dump`, and `file_path`.
- Text normalization: reject invalid UTF-8 and NUL; map CRLF and CR to LF; otherwise preserve code points and whitespace exactly.
- Document terminator: tokenizer token ID 0 (`<|endoftext|>`) appended exactly once after every accepted document.
- Exact-document duplicate key: SHA-256 over normalized UTF-8 text. Later identical documents are rejected according to canonical source order.
- Source order: lexicographic upstream `file_path`, then physical row index.
- Split: let `u` be the first unsigned 64 bits of `SHA256("expertforge-d0-split-v1\0" || UTF8(document_id))`; `u mod 1000 < 995` is train, otherwise validation.
- Training order: deterministic stateless shuffle key `SHA256("expertforge-d0-order-v1\0" || uint64_be(epoch) || uint64_be(data_seed) || UTF8(document_id))`, ascending bytes with `document_id` as the collision tiebreaker.
- Validation order: ascending `SHA256("expertforge-d0-validation-v1\0" || UTF8(document_id))`, then `document_id`.

D0.1 must materialize and publish a content-addressed local manifest that reconstructs this upstream revision and these transformations exactly. A change to any rule above is a D0.0 amendment, not a D0.1 implementation choice.

## Tokenizer identity proposal

- Repository: `EleutherAI/gpt-neox-20b`
- Immutable tokenizer commit: `364ae95407723fadd1d47b023c1efb92a4d891c3`
- Files: `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`, and `special_tokens_map.json`
- Tokenizer family: byte-level BPE
- Vocabulary size consumed by D0: 50,257
- `bos_token_id = eos_token_id = unk_token_id = 0`
- `add_prefix_space = false`
- No padding token is used in training.
- The tokenizer is not retrained for D0.

Ratification requires repository-local SHA-256 values and byte sizes for all five tokenizer files. The commit pin alone is not sufficient to close #42.

## Sequence and token accounting

- Each packed training item contains 1,025 consecutive stream tokens.
- Model inputs are tokens 0 through 1,023; targets are tokens 1 through 1,024.
- Each item therefore contributes exactly 1,024 non-padding target tokens.
- The next item begins at the prior item’s target endpoint, so one lookback token is reused only as input; target tokens are neither repeated nor skipped.
- Cross-document attention is allowed. The explicit end-of-document token is the only document-boundary representation.
- Every target position participates in loss, including end-of-document targets.
- Loss is the arithmetic mean of per-target cross-entropy over the complete optimizer update.
- “Processed tokens” means non-padding target tokens that participate in loss. Input-only lookback tokens, validation tokens, generation tokens, and profiling warmup tokens do not count toward the training budget.

Primary batch semantics for both runs:

```text
devices                         = 1
microbatch sequences/device     = 4
gradient accumulation steps     = 16
global sequences/update         = 64
target tokens/sequence          = 1,024
target tokens/update            = 65,536
```

## Optimization proposal

- AdamW: beta1 0.9, beta2 0.95, epsilon 1e-8.
- Weight decay: 0.1 on matrix weights; excluded from token embeddings and all RMSNorm weights.
- Gradient clipping: global L2 norm 1.0 after accumulation and before optimizer update.
- No dropout.
- Initialization: token embeddings, Q/K/V, and SwiGLU gate/up matrices use normal standard deviation 0.02. Attention output and SwiGLU down matrices use `0.02 / sqrt(2 * number_of_layers)`. RMSNorm weights initialize to 1.
- RMSNorm epsilon: 1e-5.
- RoPE base: 10,000; all 64 head dimensions rotated; no scaling.
- Learning-rate schedule: linear warmup from zero, then cosine decay to 10% of peak at the final semantic update.

| Run | Peak LR | Warmup | Final LR | Training tokens | Updates |
|---|---:|---:|---:|---:|---:|
| Qualification | 0.0006 | 200 updates | 0.00006 | 262,144,000 | 4,000 |
| Canonical | 0.0004 | 1,600 updates | 0.00004 | 2,097,152,000 | 32,000 |

Validation and checkpoint cadence is every 500 updates for qualification and every 2,000 updates for canonical. A fixed generation and profiling pass occurs at the same boundaries.

## Precision and environment proposal

- Primary path: one CUDA accelerator with native BF16, at least 24 GiB device memory, at least 64 GiB host memory, and at least 1 TiB free local artifact/checkpoint storage.
- Parameters and forward/backward activations: BF16 where numerically supported.
- Gradient accumulation and loss reduction: FP32.
- AdamW first and second moments: FP32.
- Master parameters: FP32.
- Canonical D0 requires native BF16 and fails closed otherwise.
- Qualification may use FP32 as an explicitly distinct fallback attempt. It is not evidence of BF16 equivalence and cannot authorize D0.6.
- FP16 is not an authorized fallback.
- Compilation and fused kernels may be used only when they preserve exact causal-attention semantics and pass the D0.2 numerical-reference suite.

## Seed and recovery proposal

- Master seed: 2,026,080,200.
- Named substreams are derived by SHA-256 domain separation for initialization, data order, dropout (reserved but unused), validation, and generation.
- Qualification must include one uninterrupted attempt and one interrupted/resumed attempt from the same initial state and data order. Their computational state must match at an agreed post-resume comparison update in the locked environment.
- Canonical training uses one declared training seed, includes at least one genuine interruption into a distinct attempt, and evaluates generation with seeds 17, 29, and 43.
- This first D0 does not claim training-seed variance. The frozen baseline’s variance statement is limited to deterministic validation and repeated fixed-checkpoint generation under the declared seeds.

## Evaluation proposal

- Validation consumes exactly 8,388,608 target tokens per evaluation boundary.
- Validation loss is mean next-token negative log likelihood in natural-log units; perplexity is `exp(loss)` and is never used alone for acceptance.
- Generation uses eight repository-committed prompts, maximum 128 new tokens, temperature 0.8, top-p 0.95, no top-k truncation, repetition penalty 1.0, and stop on token ID 0 or the length limit.
- Throughput denominator is non-padding target tokens completed in optimizer updates. Warmup, validation, generation, checkpoint, and profiling tokens are excluded.
- Throughput report uses the median of at least 100 consecutive steady-state optimizer updates after five warmup updates.
- Memory report records peak allocated and reserved accelerator bytes plus peak resident host bytes.
- Checkpoint report records logical state bytes, serialized bytes, write latency, read latency, and validation latency.
- Inference latency uses batch 1, input length 512, 128 generated tokens, one warmup and ten measured runs.

## Acceptance, failure, and kill proposal

### Qualification acceptance

All are required:

1. Final validation loss is at least 0.50 nats below the pre-update validation loss.
2. No non-finite loss, gradient, parameter, optimizer state, or validation result occurs.
3. Zero optimizer updates are skipped.
4. Checkpoint round-trip reconstructs the declared computational state exactly.
5. Interrupted/resumed execution matches uninterrupted execution at the declared comparison update in the locked environment.
6. Fixed-protocol generation completes for all prompts.
7. Throughput, accelerator memory, host memory, checkpoint size/latency, and inference latency are reported.
8. Peak accelerator allocation remains at or below 95% of physical memory.

### Canonical acceptance

All qualification conditions apply, plus:

1. Final validation loss is at least 1.00 nat below the pre-update validation loss.
2. Final validation loss is no more than the best earlier validation loss plus 0.05 nat.
3. No three consecutive validation boundaries regress by more than 0.10 nat each.
4. Median steady-state throughput in the final measured window is at least 80% of the first measured steady-state window on the same attempt and environment.
5. Every required manifest, telemetry, checkpoint, generation, comparison, profiling, and terminal-seal artifact validates.

### Failure threshold

The run is failed, but the program is not automatically killed, on any one of:

- one out-of-memory termination;
- checkpoint write or read latency greater than 300 seconds;
- peak host resident memory greater than 90% of available host memory;
- validation loss improvement below 0.10 nat after 25% of the token budget;
- one recovery attempt rejected for compatibility or lineage reasons.

A failed run must publish truthful failed-attempt evidence before retry.

### Kill criterion

The active configuration is killed and requires a reviewed amendment on any one of:

- any non-finite computational value;
- any skipped optimizer update;
- any checkpoint or resume computational-state mismatch;
- repeated out-of-memory failure after one documented batch-preserving remediation attempt;
- two failed recovery attempts;
- validation loss improvement below 0.10 nat after 50% of the token budget;
- inability to meet the token budget without changing frozen batch semantics, model dimensions, tokenizer, dataset, or precision path.

## Ratification blockers

This proposal cannot become Accepted and Issue #42 cannot close until all of the following are committed and validated:

1. SHA-256 and byte-size records for the five tokenizer files.
2. A repository-local, content-addressed dataset source manifest for the pinned FineWeb-Edu revision/configuration.
3. A formal experiment JSON document accepted by the existing experiment-manifest contract.
4. Resolved ExpertForge YAML configurations accepted by the existing configuration layer; required schema extensions must be explicit and backward compatible.
5. An independent parameter-accounting executable and tests matching the totals above.
6. Committed generation prompts and contamination checks.
7. A validation command that rejects approximate, missing, contradictory, mutable, or unpinned contract fields.
8. A rendered review report and synchronized `PROJECT_STATE.md`.

## Amendment policy

After ratification, any material change requires a dedicated amendment issue, a new specification fingerprint, an explicit comparability decision, and identification of invalidated evidence. Silent implementation reinterpretation is prohibited.

## Claim boundary

This proposal fixes a concrete candidate contract for review. It does not establish that the external data is sufficient, that the selected hyperparameters will train successfully, that the resource envelope is available, or that D0 is accepted. Those claims require completion of #42 and the later D0.1–D0.7 evidence chain.
