# Evaluation, Experiment, Promotion, and Kill Doctrine

**Status:** Normative
**Source:** Issue #1 founding technical specification §12, §13, §14, §15.

## 1. Evaluation doctrine (spec §12)

Initial evaluation includes:

- validation loss;
- perplexity where meaningful;
- generation sanity tests;
- contamination / memorization checks;
- throughput;
- peak accelerator memory;
- host memory;
- checkpoint size;
- inference latency;
- stability indicators.

MoE evaluation additionally includes:

- expert utilization;
- entropy;
- imbalance;
- capacity overflow;
- dropped-token rate;
- specialization indicators;
- contribution;
- transition statistics;
- reuse distance;
- active / resident parameter counts.

**No single metric is sufficient for architectural acceptance.**

## 2. Experiment doctrine (spec §13)

Every formal experiment defines:

```text
question
hypothesis
independent variable
dependent variables
control
fixed constraints
expected mechanism
quality metrics
systems metrics
minimum useful effect
failure threshold
kill criterion
token budget
replication policy
result
decision
```

**Experiment classes:** exploratory, confirmatory, diagnostic, replication,
systems, ablation, scaling, integration.

Important comparisons should use **multiple seeds where affordable**. Otherwise
the report must include: limitations, repeated evaluation, matched-token curves,
profiling repetitions, confidence intervals where applicable, and sensitivity
analysis.

Reports distinguish **observed result**, **plausible mechanism**, **causal
conclusion**, and **speculation**.

## 3. Promotion criteria (spec §14)

A technique may enter a later baseline only when all hold:

1. implementation is tested;
2. configuration is explicit;
3. comparison is controlled;
4. result is reproducible within declared limits;
5. quality impact is understood;
6. systems impact is measured;
7. complexity cost is documented;
8. no unresolved correctness issue remains;
9. evidence justifies replacing the baseline.

**Promotion never erases the prior baseline.**

## 4. Kill criteria (spec §15)

Stop, suspend, or redesign work when any applies:

- the effect cannot be measured reliably;
- implementation cannot be independently validated;
- comparison cannot be controlled;
- quality regression exceeds the declared threshold;
- systems benefit is within measurement variance;
- complexity is disproportionate;
- the mechanism duplicates existing work without new evidence;
- required scale exceeds resources without a credible scaling argument;
- the result depends on undocumented behavior;
- repeated instability prevents interpretation;
- a simpler technique achieves an equivalent result.

**Kill decisions and negative results are retained as project evidence.**

## 5. Relationship to ExpertOS

Evaluation results that concern runtime behavior — deployment cost, residency,
placement, transfer latency — are validated jointly with ExpertOS through the
evidence-exchange loop
([deployment-and-runtime-codesign.md](deployment-and-runtime-codesign.md) §3).
ExpertOS feedback is evidence, not an automatic training command.
