# Decision Record 0007 — D0 Declarative Parameter Inventory

**Status:** Proposed binding amendment within D0.0  
**Date:** 2026-08-03  
**Issue:** #42  
**Parent:** #41  
**Amends:** Decision Record 0003 parameter-accounting evidence only

## Context

The D0 baseline contract declares exact qualification and canonical parameter
totals and already contains a compact closed-form expression. That expression is
not sufficient by itself to demonstrate that every intended trainable tensor
family is represented exactly once, that layer multiplicities are correct, or
that the tied output head is not double-counted.

D0.0 does not authorize a production dense-model implementation. Therefore the
independent accounting evidence must be executable without constructing model
layers, allocating framework tensors, or implying that an implementation or
training attempt exists.

## Decision

### Declarative tensor inventory

The normative prospective inventory is committed as:

```text
experiments/d0/parameter-inventory-v1.json
```

It declares eleven trainable parameter families using symbolic dimensions
`V`, `L`, `d`, and `f`. Each family carries an exact shape and either unit or
layer multiplicity. The generic inventory counter resolves those dimensions
from each profile, multiplies shape dimensions, expands multiplicity, and sums
parameter elements without importing model code.

The inventory contains:

1. one shared token-embedding/output-projection matrix;
2. two RMSNorm scale vectors per Transformer block;
3. four attention projection matrices per block;
4. three SwiGLU projection matrices per block;
5. one final RMSNorm scale vector.

The output head is represented as a shared-storage alias of
`token_embedding.weight` and contributes zero additional elements. Attention,
feed-forward, and normalization biases; trainable RoPE state; dropout state; and
a separate output-head matrix are explicitly recorded as zero-parameter
component classes.

### Independent comparison

`python scripts/validate_d0_parameter_inventory.py` compares three values for
each profile:

1. the total produced by generic shape and multiplicity expansion of the
   declarative inventory;
2. the result returned by the pre-existing closed-form contract validator;
3. the trainable-parameter total declared in the machine-readable D0 contract.

The inventory counter does not import the closed-form `parameter_count` or
`swiglu_width` helpers. The existing contract validator remains the separate
closed-form path.

The required exact results are:

| Profile | Tensor families | Expanded tensor instances | Trainable parameters | Non-trainable parameters |
|---|---:|---:|---:|---:|
| Qualification | 11 | 74 | 19,685,888 | 0 |
| Canonical | 11 | 110 | 76,738,176 | 0 |

### Validation boundary

The validator rejects missing, added, reordered, or renamed tensor families;
shape or layer-multiplicity drift; altered profile dimensions or model identity;
subtotal or total disagreement; output-head double-counting; non-zero excluded
component classes; unknown symbolic dimensions; Boolean or otherwise non-exact
integer dimensions; and disagreement between the inventory, closed-form result,
and contract total.

This evidence is an implementation-independent instantiation contract. A later
D0.1 model implementation must enumerate its concrete named parameter tensors
and match this inventory exactly before it can be accepted. This decision does
not claim that such an implementation already exists.

## Alternatives rejected

### Rely only on the closed-form equation

Rejected because the same compact equation can conceal omitted tensor families,
incorrect sharing, or a mistaken interpretation of per-layer terms.

### Instantiate a provisional framework model in D0.0

Rejected because model-layer implementation is deliberately excluded from
D0.0. A provisional implementation would blur the contract/implementation
boundary and could become an accidental source of truth.

### Count the tied output head separately

Rejected because the baseline contract requires shared embedding/output storage.
Counting an additional `V × d` matrix would change both model identities and
parameter totals.

### Leave zero-parameter components implicit

Rejected because hidden bias, positional, dropout, or output-head parameters
would materially change the architecture while remaining invisible in a
positive-only tensor list.

## Consequences

- Every intended trainable tensor family and excluded component class is
  machine-readable.
- Qualification and canonical totals are reproduced by two structurally
  independent executable methods.
- Parameter-family, tensor-instance, sharing, and subtotal drift fail before
  model implementation or training.
- The parameter-accounting blocker may close after all repository CI,
  integration, locked smoke, and recovery gates pass on the synchronized head.
- D0.1 remains responsible for proving that an actual implementation's concrete
  tensor inventory matches this prospective contract.
- No model implementation, data processing, qualification run, or canonical run
  is authorized.

## Reversal conditions

This decision may be superseded only by a dedicated amendment that:

1. identifies the tensor family, shape, sharing rule, or parameter class that
   must change;
2. produces new exact qualification and canonical totals;
3. updates the model configurations, specification fingerprints, formal
   experiment definition, and every dependent validation artifact;
4. identifies any later runs or implementations made non-comparable by the
   change.
