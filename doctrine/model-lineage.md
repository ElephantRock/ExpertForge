# Model Lineage Policy

**Status:** Normative
**Source:** Issue #1 founding technical specification §6, §7, §8.

The platform is continuous; models are controlled variants. The mature
architecture is not predetermined. Each model below is a milestone with entry
and exit criteria, not a fixed product.

## 1. Dense architecture policy (D0)

D0 should use conservative, widely understood components unless one is the
independent variable under study:

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

The intended initial scale is approximately 20–100M parameters, with 50–100M as
the first major training target.

## 2. Model lineage

### D0 — Minimal dense baseline

**Purpose:** Validate the complete pipeline and establish the first canonical
scientific baseline.

**Entry criteria:**

- Milestone 0 experimental substrate is complete;
- tokenizer and dataset versions exist and are immutable for the run;
- core model components have unit tests;
- training, checkpoint, validation, and generation smoke tests pass.

**Required characteristics:** conservative decoder-only architecture defined
above; minimal custom optimization or novelty; complete provenance and
deterministic configuration resolution.

**Exit criteria:**

- stable run from random initialization;
- validation loss improves over a declared token budget;
- checkpoint round-trip and interruption recovery are demonstrated;
- generation works at expected scale quality;
- training/inference throughput and memory are reported;
- canonical D0 configuration, tokenizer, dataset, evaluation suite, and variance
  assumptions are frozen.

**Prohibited scope:** MoE, deployment penalties, architectural novelty unrelated
to correctness, or uncontrolled combinations of optimizations.

### D1 — Optimized dense baseline

**Purpose:** Improve systems efficiency without materially changing model
semantics.

**Entry criteria:** Frozen and reproducible D0.

**Candidate work:** fused operations, compilation, activation checkpointing,
improved data loading, distributed execution, optimizer-state efficiency, and
robust checkpoint I/O.

**Exit criteria:**

- each promoted optimization has a controlled D0 comparison;
- numerical correctness and convergence remain within declared tolerance;
- throughput and memory effects are measured;
- recovery and reproducibility remain valid;
- a canonical optimized dense baseline is frozen.

**Prohibited scope:** claiming architectural-quality gains from systems-only
changes without evidence.

### D2 — Efficient dense variants

**Purpose:** Test architecture-level dense efficiency techniques.

**Entry criteria:** Stable D1 and frozen evaluation protocol.

**Candidate work:** grouped-query attention, sliding-window attention,
depth/width allocation, parameter sharing, alternative positional encodings,
feed-forward sizing, and normalization variants.

**Exit criteria:**

- every variant has a falsifiable hypothesis and controlled baseline;
- quality, convergence, memory, throughput, latency, and complexity are
  measured;
- promoted changes satisfy minimum-useful-effect and quality-risk thresholds;
- rejected and null results are retained.

**Prohibited scope:** combining several unablated architectural changes and
attributing the result to one technique.

### M0 — Minimal MoE

**Purpose:** Establish a correct sparse feed-forward implementation and honest
dense-versus-MoE comparison.

**Entry criteria:** Trustworthy dense baseline, fixed tokenizer/dataset/
evaluation suite, and tested token dispatch/combine primitives.

**Required characteristics:**

- small expert count;
- top-1 or top-2 routing;
- explicit capacity semantics;
- load-balancing auxiliary loss;
- documented overflow/drop/reroute behavior;
- native routing instrumentation;
- comparison at matched or clearly normalized active compute and training-token
  budget.

**Exit criteria:**

- stable training;
- verified routing, capacity, dispatch, and combine behavior;
- checkpoint recovery;
- expert utilization, entropy, imbalance, overflow, and dropped-token metrics;
- honest comparison with the dense baseline covering total parameters, active
  parameters, compute, memory, quality, and latency.

### M1 — Balanced-routing MoE

**Purpose:** Improve utilization and training stability while measuring the
quality cost of balance constraints.

**Entry criteria:** Stable M0 with reproducible routing metrics.

**Exit criteria:**

- controlled experiments over balance/capacity objectives;
- utilization and overflow improvement is demonstrated;
- quality and convergence cost is quantified;
- selected configuration has explicit promotion or rejection evidence.

### M2 — Instrumented MoE

**Purpose:** Turn the model into an expert-level scientific instrument.

**Entry criteria:** Stable M1 and a versioned routing-trace schema.

**Required observability:** expert selection by token/layer, router logits and
probabilities, entropy, capacity use, overflow, output norms, contribution
estimates, gradient/parameter norms, co-activation, route transitions, reuse
distance, and workload/sequence-conditioned utilization.

**Required interventions:** expert masking, substitution, permutation, output
scaling, precision changes, eviction simulation, and transfer-delay simulation
where technically valid.

**Exit criteria:**

- instrumentation correctness is tested;
- overhead is measured;
- traces and intervention results export through stable schemas;
- ExpertOS can consume at least one compatible evidence artifact without
  architecture-specific source copying.

### M3 — Locality-trained MoE

**Purpose:** Test objectives that shape expert working sets, transitions, reuse,
predictability, and placement cost.

**Entry criteria:** M2 evidence is sufficient to define and measure locality and
deployment cost.

**Exit criteria:**

- each objective is independently defined and ablated;
- ordinary MoE is the control;
- at least one deployment metric improves beyond measurement variance;
- quality remains within a declared risk threshold;
- causal mechanism and alternative explanations are documented;
- ExpertOS evaluator confirms or rejects runtime value.

### M4 — Structured-expert MoE

**Purpose:** Test expert representations designed for memory reduction, sharing,
compression, substitution, or partial residency.

**Candidate structures:** shared base plus expert deltas, expert families,
low-rank expert adaptation, clustered experts, shared intermediate projections,
and partial expert materialization.

**Entry criteria:** M2 instrumentation and an identified runtime bottleneck that
ordinary placement policy does not adequately solve.

**Exit criteria:**

- memory/residency benefit is measured;
- specialization and quality are preserved within declared thresholds;
- kernel and runtime feasibility is demonstrated rather than assumed;
- complexity cost and fallback behavior are documented;
- ExpertOS evaluation shows end-to-end value or the technique is rejected.

### M5 — Native ExpertOS-aware MoE

**Purpose:** Establish genuine model–runtime co-design.

**Entry criteria:** At least one validated M3 or M4 mechanism and stable
interoperability schemas.

**Required capabilities:**

- formal machine-readable resource contract;
- expert identities and tensor boundaries;
- byte sizes and precision capabilities;
- routing traces and prefetch signals;
- placement constraints and safe fallback paths;
- supported substitution relationships;
- quality-risk metadata;
- consumption of deployment evidence in architecture/training decisions.

**Exit criteria:**

- train → profile in ExpertOS → identify bottleneck → modify model/training →
  retrain/adapt → execute through ExpertOS → measure quality/latency/memory/
  transfer loop is reproducible;
- at least one co-designed model change outperforms runtime-only optimization
  under a declared constrained-hardware scenario;
- schemas and compatibility rules are versioned;
- the result survives controlled ablation and review.

## 3. Scientific baseline rules

Every canonical baseline must freeze:

- source revision and uncommitted-change status;
- tokenizer version;
- dataset version and sampling rules;
- training-token budget;
- model configuration;
- optimizer and schedule;
- initialization and seed policy;
- precision and distributed strategy;
- hardware environment;
- evaluation suite.

Baselines may be superseded but never retroactively modified. Prior baselines
must remain reproducible.

## 4. Baseline-change review

Any change to a referenced baseline or lineage is a baseline change and requires
explicit review before merge (collaboration §10), plus a decision record under
`doctrine/decisions/` when the change is durable. Promotion never erases the
prior baseline.
