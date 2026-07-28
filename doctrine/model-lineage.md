# Model Lineage Policy

**Status:** Normative
**Source:** Founding technical specification §6, §7, §8 (maturity stages and
dense policy) and the canonical research-family decision §F0–F10 (Issue #1
comment `5109231939`). On merge, this committed policy on `main` is
authoritative.

The platform is continuous; models are controlled variants. The mature
architecture is not predetermined. ExpertForge distinguishes **maturity stages**
(platform capability and integration level) from **research families**
(orthogonal experimental branches that isolate individual architectural
mechanisms).

## 0. Lineage structure

The lineage has two orthogonal axes:

- **Maturity stages (D0–M5)** describe platform capability and integration
  level. They are top-level milestones with entry and exit criteria.
- **Research families (F0–F10)** isolate individual architectural mechanisms.
  Each family has its own baseline, hypothesis, metrics, entry/exit criteria,
  kill criteria, and combination eligibility.

Definitions used throughout this document:

- **maturity stages** describe platform capability and integration level;
- **research families** isolate individual architectural mechanisms;
- **model variants** are concrete configurations produced within a family;
- **experiments** compare variants under a frozen protocol;
- **promotion** occurs only through evidence, not because a mechanism appears in
  a target architecture.

**No monolithic "K3-like" architecture will be implemented as a single lineage
step.** Each mechanism must first exist as a separately controlled research
family.

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

These D0 component choices are durable architecture decisions recorded in
`doctrine/decisions/0002-initial-d0-architecture.md`. Each component remains
replaceable only through a controlled family experiment (e.g., a D2-family or
relevant F-family experiment), never by unrecorded substitution.

## 2. Maturity stages (D0–M5)

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

**Purpose:** Test architecture-level dense efficiency and memory-mechanism
techniques. D2 is the maturity stage within which the dense research families
(F1–F6) are exercised; the order within D2 is evidence-driven, not an obligation
to promote every family.

**Entry criteria:** Stable D1 and frozen evaluation protocol.

**Candidate work:** grouped-query attention, sliding-window attention,
depth/width allocation, parameter sharing, alternative positional encodings,
feed-forward sizing, normalization variants, and the recurrent/exact/hybrid
memory mechanisms studied under F1–F6.

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

## 3. Research families (F0–F10)

Research families are orthogonal experimental branches that isolate individual
architectural mechanisms. Each family has its own baseline, hypothesis,
metrics, entry/exit criteria, kill criteria, and combination eligibility. No
two novel families may be combined into a promoted baseline without satisfying
the combination gate (§4).

### F0 — Exact-attention dense baseline

**Purpose:** Establish the semantic and systems control against which later
memory mechanisms are compared.

Includes:

- standard causal softmax attention;
- conventional KV caching for autoregressive inference;
- D0 correctness and reproducibility baseline;
- D1 systems-only optimizations such as FlashAttention, fused kernels,
  compilation, activation checkpointing, and improved data loading.

**Rule:** systems optimizations that preserve model semantics belong here and
must not be confused with architectural memory changes.

### F1 — Additive linear-attention memory

**Purpose:** Isolate the effect of replacing explicit token-addressable KV
memory with a fixed-size recurrent state.

Includes:

- feature-map linear attention;
- additive state updates;
- recurrent and chunkwise implementations;
- state-capacity and interference measurements.

**Control:** matched dense softmax model and, where possible, matched parameter
count, active computation, sequence length, and training-token budget.

### F2 — Delta-rule memory

**Purpose:** Measure whether targeted corrective writes improve finite-state
associative memory over purely additive updates.

Includes:

- DeltaNet-style read-before-write correction;
- write-strength ablations;
- associative recall and interference tests;
- recurrent versus chunkwise-equivalent implementations.

**Dependency:** F1 must be correct and characterized before F2 begins.

### F3 — Gated recurrent memory

**Purpose:** Isolate the value and cost of explicit forgetting.

Includes:

- scalar decay or retention gates;
- DeltaNet plus gating;
- context-switch and stale-memory tests;
- write/forget interaction ablations.

**Dependency:** compare directly against F2 under the same state size and budget.

### F4 — Fine-grained recurrent memory

**Purpose:** Test whether channel-wise or otherwise fine-grained decay improves
use of finite recurrent state.

Includes:

- KDA-style vector or channel-wise gating;
- gate-granularity ablations;
- state-component survival analysis;
- stability, saturation, and effective-capacity measurement.

**Dependency:** F3 must establish the scalar-gated control.

### F5 — Hybrid sequence memory

**Purpose:** Test combinations of compressed recurrent memory and periodic exact
retrieval.

Includes:

- alternating or interleaved recurrent-memory and exact-attention layers;
- ratio and placement ablations;
- KDA/MLA-like hybrid structures without assuming any specific external
  implementation;
- quality, KV-cache, state-memory, prefill, and decode trade-offs.

**Dependency:** F0 and the relevant recurrent-memory family must each pass
independently before combination.

### F6 — Depth-selective residual memory

**Purpose:** Isolate learned retrieval across network depth from changes to
sequence memory or expert routing.

Includes:

- AttnRes-style learned weighting over prior layer or block representations;
- blockwise versus full-depth variants;
- representation-age and selection-entropy measurements;
- gradient-flow and memory-overhead analysis.

**Rule:** first evaluate F6 on a dense exact-attention baseline. Do not
initially combine it with recurrent attention or MoE.

### F7 — Conventional sparse MoE

**Purpose:** Establish the sparse-capacity control.

Includes:

- M0 minimal top-1 or top-2 routing;
- M1 balance and capacity-policy experiments;
- M2 instrumentation and expert intervention;
- ordinary full-width experts before latent or shared-structure variants.

**Control:** matched dense baseline with clearly normalized total parameters,
active parameters, computation, memory, and token budget.

### F8 — Latent and structured experts

**Purpose:** Isolate expert-representation changes from sparse routing itself.

Includes:

- lower-dimensional latent expert computation;
- projection-cost accounting;
- shared-base-plus-delta experts;
- expert families, clustering, and low-rank expert structure;
- resident bytes, transferred bytes, kernel cost, and specialization quality.

**Dependency:** F7 must provide a stable and instrumented conventional-MoE
control.

### F9 — Deployment-aware expert routing

**Purpose:** Shape routing behavior for constrained hardware without conflating
the effect with expert representation changes.

Includes:

- locality regularization;
- expert-switching penalties;
- working-set and residency-budget objectives;
- route-predictability objectives;
- bandwidth, placement, and transfer-cost penalties.

**Dependency:** begin with conventional experts from F7. Structured experts from
F8 may be combined only in a later factorial experiment.

### F10 — Integrated memory-system interactions

**Purpose:** Study interactions among sequence memory, parametric memory, and
depth memory only after each mechanism is independently characterized.

Candidate interactions:

- recurrent-memory gate × MoE route;
- exact-retrieval contribution × MoE route;
- depth-selection weight × expert route;
- recurrent-memory gate × depth-selection weight;
- hybrid attention × structured expert locality.

This family is not an early implementation target. It corresponds to late M4/M5
research and requires explicit factorial or staged ablations.

## 4. Combination gate

No two novel families may be combined into a promoted baseline until:

1. each has a validated implementation;
2. each has been compared independently against the same declared control;
3. each has a documented positive, negative, or null result;
4. the interaction has a separate hypothesis;
5. the combined experiment includes ablations that can attribute the result;
6. the added complexity is justified by a measured effect.

A failed or null family remains in the lineage as evidence and is not silently
folded into another architecture. No compound K3-like model may be promoted
before each constituent mechanism passes independently and the combined
experiment has attribution-preserving interaction ablations.

## 5. Experimental contract for every family

Every research family must define:

```text
family identifier
research question
falsifiable hypothesis
semantic control
systems control
independent variable
dependent variables
fixed constraints
implementation-equivalence tests
quality metrics
systems metrics
minimum useful effect
failure threshold
kill criterion
replication policy
combination eligibility
```

At minimum, report:

```text
validation loss and task quality
training stability and convergence
training tokens and effective batch
parameter count and active parameter count
training FLOPs or a declared proxy
wall-clock throughput
prefill latency
decode latency
peak accelerator memory
host memory
persistent inference state
memory-transfer volume where measurable
kernel/runtime limitations
```

Mechanism-specific instrumentation is additionally required.

## 6. Mapping: maturity stages ↔ research families

The D0–M5 labels remain top-level maturity stages; research families are
orthogonal experimental branches:

```text
D0  exact-attention dense correctness baseline (F0)
D1  systems-optimized dense baseline (F0)
D2  controlled dense architecture families (F1–F6)
M0  minimal conventional MoE (F7)
M1  balanced and stable routing (F7)
M2  instrumented and intervention-capable MoE (F7)
M3  deployment-aware routing (F9)
M4  latent and structured experts (F8), plus approved interactions
M5  ExpertOS-aware integrated model/runtime co-design (F10)
```

The order within D2 is evidence-driven, not an obligation to promote every
family.

## 7. Scientific baseline rules

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

## 8. Baseline-change review

Any change to a referenced baseline or lineage is a baseline change and requires
explicit review before merge (collaboration §10), plus a decision record under
`doctrine/decisions/` when the change is durable. Promotion never erases the
prior baseline.
