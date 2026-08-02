# D0 Parameter Accounting Report

**Issue:** #42  
**Parent:** #41  
**PR:** #43  
**Status:** D0.0 draft evidence; declarative parameter-accounting tranche implemented

## Result

A versioned declarative tensor inventory now provides an independently
executable derivation of the qualification and canonical parameter totals. The
generic inventory counter expands symbolic tensor shapes and layer
multiplicities, while the existing D0 contract validator independently evaluates
the frozen closed-form expression. Both methods must equal the declared contract
total.

No production model object, framework parameter, data pipeline, training loop,
qualification run, or canonical run is created by this evidence.

## Exact comparison

| Profile | Tensor families | Expanded tensor instances | Inventory total | Closed-form total | Contract total |
|---|---:|---:|---:|---:|---:|
| Qualification | 11 | 74 | 19,685,888 | 19,685,888 | 19,685,888 |
| Canonical | 11 | 110 | 76,738,176 | 76,738,176 | 76,738,176 |

Both profiles declare zero non-trainable parameters.

## Inventory derivation

For symbolic dimensions vocabulary size `V`, block count `L`, model width `d`,
and SwiGLU width `f`, the inventory expands these parameter families:

| Family | Shape | Multiplicity |
|---|---|---:|
| `token_embedding.weight` | `[V, d]` | 1 |
| `blocks.attention_norm.weight` | `[d]` | `L` |
| `blocks.attention.q_proj.weight` | `[d, d]` | `L` |
| `blocks.attention.k_proj.weight` | `[d, d]` | `L` |
| `blocks.attention.v_proj.weight` | `[d, d]` | `L` |
| `blocks.attention.out_proj.weight` | `[d, d]` | `L` |
| `blocks.ffn_norm.weight` | `[d]` | `L` |
| `blocks.ffn.gate_proj.weight` | `[f, d]` | `L` |
| `blocks.ffn.up_proj.weight` | `[f, d]` | `L` |
| `blocks.ffn.down_proj.weight` | `[d, f]` | `L` |
| `final_norm.weight` | `[d]` | 1 |

The expanded tensor-instance count is `9L + 2`: 74 for eight blocks and 110
for twelve blocks.

`output_head.weight` is a shared-storage alias of `token_embedding.weight` and
adds zero elements. The inventory explicitly records zero parameter elements
for attention biases, feed-forward biases, normalization biases, trainable RoPE
state, dropout state, and a separate output head.

## Independence boundary

The inventory counter computes each term from only its declared shape,
multiplicity, and resolved positive-integer dimensions. It does not import the
closed-form `parameter_count` or `swiglu_width` helpers.

The second path remains the existing contract derivation:

```text
V*d + L*(4*d^2 + 3*d*f + 2*d) + d
```

Agreement is required among the generic inventory expansion, that closed-form
result, and the declared profile total. This is an implementation-independent
prospective contract, not a claim that production model tensors have already
been instantiated. D0.1 must later compare concrete named tensors against this
inventory before implementation acceptance.

## Validation coverage

The validator and mutation tests reject:

- missing, added, reordered, or renamed tensor families;
- shape and layer-multiplicity drift;
- incorrect per-family subtotals or aggregate totals;
- profile-dimension or model-identity crossover;
- tied output-head double-counting;
- non-zero excluded parameter classes;
- unknown symbolic dimensions;
- zero, negative, Boolean, or otherwise non-exact integer dimensions;
- disagreement with the independent closed-form result;
- disagreement with the machine-readable contract total.

A test deliberately supplied a Boolean symbolic dimension and exposed a Python
integer-subtype edge case. The generic resolver now revalidates every resolved
dimension as an exact positive integer.

## Commands

```bash
uv run python scripts/validate_d0_parameter_inventory.py
uv run pytest -q tests/test_d0_parameter_inventory.py
```

## Blocker state

The blocker

```text
independent_parameter_accounting_executable_and_tests
```

may close after the complete repository CI, portable CPU integration, locked
smoke, and one-command interruption/recovery gates pass on the synchronized
contract-and-evidence head.

## Remaining dependency order

1. committed generation prompts and contamination checks;
2. permanent D0 contract validation command and CI gate;
3. final rendered review report;
4. `PROJECT_STATE.md` synchronization.

No model implementation or material run is authorized.
