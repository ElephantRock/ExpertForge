# Decision Record 0002 — Initial D0 Architecture

**Status:** Accepted
**Date:** Issue #1 correction (founding)
**Supersedes:** none
**Related:** `doctrine/model-lineage.md` §1 (dense architecture policy), §2 (D0)

## Context

D0 is the first canonical scientific baseline. Its job is to validate the
complete pipeline — data, tokenizer, model, training loop, checkpoint, and
evaluation — and to establish a frozen, reproducible reference against which all
later mechanisms (the F0–F10 research families and D2 variants) are compared.

For a baseline to be a clean control, D0 should use **conservative, widely
understood components** unless a component is itself the independent variable
under study. The choices below are recorded because `AGENTS.md` requires durable
architecture decisions to have a decision record, and because D0's components
will be referenced as the control by every later family experiment.

The intended initial scale is approximately 20–100M parameters, with 50–100M as
the first major training target.

## Decision

The D0 baseline uses:

```text
architecture: decoder-only Transformer
attention: causal multi-head self-attention
normalization: pre-norm RMSNorm
position encoding: RoPE
feed-forward: SwiGLU
residual structure: standard residual paths
optimizer: AdamW
precision: BF16 where supported, otherwise documented fallback
data representation: packed fixed-length token sequences
objective: next-token prediction
```

## Alternatives considered

For each component, the alternative exists and is **not rejected as wrong** — it
is deferred to a controlled family/variant experiment where it can be the
independent variable:

- **Attention:** alternatives (linear/linear-attention memory, sliding-window,
  grouped-query) are studied under F1–F6 and D2 variants. D0 uses exact causal
  softmax attention so it is a faithful semantic control.
- **Normalization:** LayerNorm vs RMSNorm, pre-norm vs post-norm. RMSNorm
  pre-norm chosen for training stability and simplicity; alternatives are D2
  variant experiments.
- **Position encoding:** RoPE vs learned absolute vs ALiBi. RoPE chosen for
  strong empirical baselines and relative-position behavior; alternatives are D2
  variant experiments.
- **Feed-forward:** SwiGLU vs GeGLU vs vanilla MLP. SwiGLU chosen as a
  well-understood gated FFN; alternatives are D2 variant experiments.
- **Optimizer:** AdamW vs Lion vs Adafactor vs Shampoo. AdamW chosen as the
  well-characterized default; alternatives are studied as training-systems
  experiments, not as model-architecture changes.
- **Precision:** BF16 vs FP16 with loss scaling vs FP32. BF16 chosen where
  supported; fallback is documented per hardware environment.
- **Objective:** next-token prediction only. Auxiliary objectives (e.g.,
  load-balance for MoE) enter only in their respective families (F7+), not in
  the dense D0 baseline.

## Evidence

- The component set is the conservative decoder-only recipe used across modern
  small/mid-scale dense baselines; each component is widely understood and
  independently testable.
- D0's role as the semantic control for F0–F10 requires that no component be a
  confound. Conservative choices minimize the chance that a D0 idiosyncrasy is
  mistaken for a mechanism effect.
- No ExpertOS source or external model internals are copied; this is a
  from-scratch ExpertForge implementation.

## Consequences

- D0 establishes the reference that every later comparison normalizes against
  (matched parameters, active compute, sequence length, token budget —
  model-lineage §3, §5).
- Each D0 component remains **replaceable only through a controlled experiment**
  (a D2 family/variant or the relevant F-family), never by unrecorded
  substitution. Replacing a component creates a new variant, not a silent edit
  to D0.
- Promotion of any replacement follows the promotion criteria
  (evaluation-and-experiments §3); promotion never erases D0.
- The component set is deliberately unremarkable: if a later mechanism shows an
  effect against D0, the effect cannot be dismissed as a D0 artifact.

## Reversal conditions

This decision would be superseded by a new decision record if:

- a D0 component is shown to be a confound for a class of mechanisms under
  study, requiring a different control baseline (recorded as a new baseline
  under model-lineage §8);
- the target scale moves outside the range where these components are
  well-characterized (e.g., very large scale where different
  parallelism/optimizer choices dominate), warranting a new baseline decision;
- a controlled D2/family experiment promotes a replacement that becomes the new
  canonical optimized dense baseline (D1/D2), at which point this record remains
  the provenance of the original D0 but no longer describes the active baseline.

Any such reversal follows the normal change workflow (issue → branch → PR →
review → merge) and a new decision record.
