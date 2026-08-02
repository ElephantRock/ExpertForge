# Milestone 0 End-to-End Smoke & Recovery Gate — Acceptance Report

**Issue:** #14 (closes #13 and #14 together)
**Maturity stage:** Milestone 0 (experimental substrate)
**Classification:** `smoke_test` (infrastructure gate, not a model-quality claim)
**Normative references:** design `5151917939`, amendment `5152078541` (A–P),
authorization + clarifications `5152259715`, and blocking review `4835261100`
(ten findings F1–F9, each corrected below with a dedicated regression).
**Decision:** **PASS**

## 1. Objective

Prove the entire Milestone 0 substrate (configuration, identity, provenance,
RNG, telemetry, artifacts, checkpoints, experiment manifests) works together
through one end-to-end path: train a toy model → checkpoint at a mid-training
boundary K → interrupt → authoritatively load the checkpoint → restore through
the full transaction (RNG last) → continue to N → validate → generate → publish
terminal manifests. The acceptance comparison is **U0@N versus R1@N** over
*computational state* — never over identity-bound containers (run/attempt IDs,
timestamps, provenance records, artifact IDs, manifest bytes, or checkpoint tar
bytes, which are intentionally unequal even when computation is exactly
reproducible).

## 2. Acceptance criteria

1. U0@N and R1@N produce an identical canonical computational-state digest
   (parameters, optimizer slots, scheduler state, counters, data cursor,
   isolated next-item cursor probe, isolated next-random-sample probe from a
   cloned RNG bundle, validation loss, generated sample).
2. Final fixed-validation loss improves over the initial value by at least the
   predeclared `evaluation.loss_improvement_threshold` (amendment J).
3. R0 publishes a truthful interrupted/partial manifest and never appears
   completed.
4. Every required terminal artifact is published in the correct per-attempt
   order, with the manifest finalized last; the comparison report is published
   as an R1 artifact before R1's manifest.
5. The permanent smoke CI job runs the locked smoke tier and the one-command
   gate.

## 3. Topology and configuration

- **U0** — `AllocationMode.INDEPENDENT`, completed. Train 0..K → checkpoint@K →
  continue K..N → validate → generate → final checkpoint.
- **R0** — `AllocationMode.INDEPENDENT` (same specification fingerprint, distinct
  run identity), interrupted at K. Train 0..K → parent checkpoint@K → terminate.
  No validate/generate after K.
- **R1** — `AllocationMode.RESUME` (retains R0's `run_id`, new `attempt_id`,
  same specification fingerprint, fully-qualified native lineage naming R0's K
  checkpoint). Authoritative load → full `RestoreTransaction` (RNG last) →
  continue K..N → validate → generate → final checkpoint.

Frozen gate configuration (`configs/m0-smoke-gate.yaml`):

| Field | Value |
|-------|-------|
| model.dim / n_layers / n_heads / ffn_dim | 16 / 1 / 1 / 32 |
| training.seed | 7 |
| training.batch_size / seq_len | 2 / 8 |
| training.tokens (N) | 128 |
| checkpointing.interval_tokens (K) | 32 |
| training.lr | 0.1 |
| evaluation.loss_improvement_threshold | 0.1 |

Update-boundary semantics (review correction #4):

```
tokens_per_update = batch_size * seq_len = 2 * 8 = 16
N_updates         = 128 / 16 = 8
K_updates         = 32  / 16 = 2
0 < K_updates(2) < N_updates(8)
```

State invariants held at every K/N checkpoint:
`global_update == completed_microsteps`, `accumulation_position == 0`,
`processed_tokens == global_update * tokens_per_update`.

The toy model is a pure-NumPy causal decoder (token embedding + a stack of
`n_layers` causal self-attention blocks factored into `n_heads` heads + ordinary
ReLU FFN + output projection). The frozen gate config declares
`n_layers=1, n_heads=1`, but the implementation **genuinely honors every declared
dimension** (finding F9): `n_heads` controls the attention head factorization and
`n_layers` controls the block-stack depth (verified against finite differences at
multi-head/multi-layer configs: dim=16/n=2×2, dim=24/n=2×3). With the frozen
config it is **2560 exact parameters**, float32, no dropout, no nondeterminism,
analytical backward verified against finite differences. RoPE/RMSNorm/SwiGLU are
intentionally omitted. This is an infrastructure gate toy model under
`expertforge.smoke`, **not** a reusable training framework or D0 precursor.

## 4. Designated evidence run

- **Evidence source commit E:** `991b50a0f1566ac7dcda83452007f0e86162e760`
  (clean working tree; the source snapshot enters the specification fingerprint,
  so the evidence run was executed against a clean checkout of E, after stashing
  the report/state changes that belong to commit H).
- **Report committed at final review head H:** see the PR; all validation and
  the permanent CI re-run on H.
- **Command (run from commit E):**
  ```
  uv run --locked python -m expertforge.smoke.run \
      --artifact-root runs/m0-smoke-evidence \
      --config configs/m0-smoke-gate.yaml --repo .
  ```
- **Exit code:** 0 (`SMOKE GATE: PASS`)

## 5. Observed results (designated evidence run)

| Measurement | Value |
|-------------|-------|
| `passed` | `true` |
| `computational_equal` | `true` |
| `comparison_mismatches` | `[]` |
| initial validation loss | 2.8403357104342373 |
| U0 final validation loss | 2.4023344710684476 |
| R1 final validation loss | 2.4023344710684476 |
| loss improvement (initial − final) | 0.4380012393657897 |
| predeclared threshold | 0.1 |
| loss-movement criterion met | yes |
| U0 computational digest | `925fbd9610757842e3159ef3e55202a4ef10c3c62a5547ad9135c1b7dbd5eab7` |
| R1 computational digest | `925fbd9610757842e3159ef3e55202a4ef10c3c62a5547ad9135c1b7dbd5eab7` |

U0 and R1 final validation losses are bit-identical; the computational digests
are bit-identical. Validation uses a held-out corpus suffix disjoint from the
training prefix (F2). The digests are derived only from computational state (the
run/attempt/timestamp/provenance/artifact/manifest/tar-byte containers are
deliberately excluded and are intentionally unequal).

## 6. Artifact and manifest identities (designated evidence run)

Specification fingerprint (shared by U0/R0/R1):
`spec-v1-sha256-2909813727ee706d9af7a5eb9592d5db5c9568b2f81f851934a61f05a43ca901`

| Attempt | Run/attempt (distinct) | Terminal manifest artifact ID |
|---------|------------------------|-------------------------------|
| U0 (completed) | `run-…efac6103…` / `attempt-…9abbd14d…` | `artifact-v1-sha256-47db07773d9c2505dabd1c820c6f0dd0a9fc46265f8f8a434c883abf0882a9ba` |
| R0 (interrupted) | `run-…ea252e6b…` / `attempt-…435b206e…` | `artifact-v1-sha256-006bd818e995d52c301423228c4d113218af123d3e7164bb499b01b6d3554acf` |
| R1 (completed, resumed) | `run-…ea252e6b…` / `attempt-…8162e2ba…` | `artifact-v1-sha256-af8cf076fbe86f93ae67490fb9bf6a3fd38fd1250f2cc9a7e51525e4b05cf2b1` |

R0's K checkpoint (R1's resume parent):
`artifact-v1-sha256-22ad4b7f260ecfb238fa390b8c75f8b276e52a66705162974bc99b60bf7ac132`

R0's manifest is `status="interrupted"`, `outcome_diagnostic="handled_interruption"`,
with truthful partial evidence (`evaluation_summary_missing`,
`generated_output_artifact_missing`). U0 and R1 manifests are `completed` with
complete core evidence.

## 7. Determinism claim and bounds (amendment I)

Exact-byte reproducibility is claimed **only** for the same locked
ExpertForge/Python/NumPy environment on the supported CPU platform. The
designated evidence environment:

- Python 3.11.14
- NumPy 2.4.6
- Platform: Windows / AMD64 (local evidence); the permanent Ubuntu 24.04 CPU CI
  job is authoritative for repository acceptance.

No cross-platform or cross-BLAS bitwise identity is claimed unless separately
demonstrated. Operations and tensor sizes are small and deterministic; no
backend-sensitive parallel reductions are used.

## 8. Resource bounds (measured)

The full U0/R0/R1 gate (3 attempts) was measured on the designated evidence
environment (Python 3.11.14, NumPy 2.4.6, Windows/AMD64) using the committed
helper `scripts/measure_smoke_gate.py` (runs the gate in-process under
`tracemalloc`):

```
uv run --locked python scripts/measure_smoke_gate.py \
    --artifact-root runs/m0-smoke-measure --config configs/m0-smoke-gate.yaml
```

| Measurement | Value |
|-------------|-------|
| wall-clock time (full gate, in-process) | **3.89 s** |
| peak Python-allocated memory (`tracemalloc`) | **1.7 MiB** |

The `tracemalloc` figure counts only Python-allocated heap (toy model parameters
+ optimizer state + RNG bundle + artifacts); process RSS is higher due to the
interpreter and NumPy runtime overhead (typically tens of MiB for this process).
The model is 2560 parameters over an 800-byte corpus. The permanent smoke CI job
is bounded at 20 minutes; the measured wall-clock leaves wide margin.

## 9. Substrate defect found and fixed during the gate

The gate was the first realistic capture to exceed 10 tensors (6 model
parameters + 12 AdamW slots + 1 scheduler = 19 tensors). This exposed a latent
defect in `CheckpointManifest`'s `tensor_members` ordering check: it sorted
member names lexicographically, but unpadded decimal indices sort incorrectly
at ≥11 entries (`tensors/10.bin` before `tensors/2.bin`). The check now sorts by
ascending numeric index (the authoritative invariant), preserving the
contiguous-from-0 contract. The existing Issue #11 regression suite continues to
pass (it never exceeded 10 tensors).

## 9b. Review-finding corrections (blocking review `4835261100`)

Each finding was corrected and given a dedicated regression test in
`tests/integration/test_smoke_gate.py` (and `tests/test_smoke_components.py`):

- **F1 (R1 trains before restore):** R1 now authoritatively loads + restores
  through the full transaction (RNG last) BEFORE any state-advancing work; no
  pre-restore training occurs. Regression: `TestNewFindingRegressions.
  test_r1_restores_before_any_state_advancing_work`.
- **F2 (validation not held out):** validation reads a held-out corpus suffix
  disjoint from the consumed training prefix (`validation_tokens()`).
  Regression: `test_validation_uses_disjoint_held_out_window`.
- **F3 (failure reports crash):** comparison mismatch diagnostics are
  canonicalized (closed domain, sorted, deduplicated) before constructing the
  report. Regression: `test_failing_comparison_reports_failure_without_crashing`.
- **F4 (config/provenance published late):** resolved configuration + provenance
  are published FIRST, before any training/checkpoint artifacts.
- **F5 (reimplemented checkpoint internals):** the smoke package now uses the
  public checkpoint SafeValue API (`safe_value_from_native` /
  `safe_value_to_native`, newly exported) instead of importing private `_Safe*`
  models.
- **F6 (required telemetry events absent):** the workflow now emits
  `checkpoint.saved`/`loaded`/`restored`, `validation.completed`,
  `generation.completed`, and `checkpoint.restore_failed`. Regression:
  `test_required_telemetry_events_are_emitted`.
- **F7 (incomplete failure matrix):** added compat-mismatch →
  `RestoreIncompatibleError`, missing manifest evidence → `ManifestBindingError`,
  wrong parent fingerprint → `ProvenanceOrchestrationError`, and the canonical
  failing-comparison path.
- **F8 (bypassable terminal guard):** `SmokeAttemptResult` no longer exposes a
  publish-capable `ArtifactStore`; the attempt-scoped guard is the sole
  publication path. Regression: `test_result_does_not_expose_publish_capable_store`.
- **F9 (untruthful dimensions):** the model genuinely factors attention over
  `n_heads` and stacks `n_layers` blocks (verified vs finite differences at
  multi-head/multi-layer configs). Regression:
  `test_analytical_backward_matches_finite_differences` (now covers 1×1, 2×2, 2×3).

Re-review (`4835379984` and `4835574289`) corrections:

- **B1 (bypassable terminal guard → race-safe, crash-safe, artifact-bound,
  opt-in sealing via seal-intent):** `publish(seal=True)` writes a durable
  ``sealing`` intent marker — bound to the intended terminal artifact_id — under
  the attempt lock BEFORE the terminal commit. The pre-lock fast-path rejects
  only if ``sealed`` exists; the in-lock context-aware check
  (``_assert_not_sealed_publish``) rejects if ``sealed`` exists, or if
  ``sealing`` exists unless ``seal=True`` AND the intent binds the exact
  artifact_id being published (authorized crash-recovery). Recovery is
  artifact-bound: the intent's content is the artifact_id; completion verifies
  the binding before removing the intent. A crash between intent-write and
  seal-completion leaves the intent, which blocks every mutating path; only a
  retry of the SAME artifact with ``seal=True`` recovers. Three crash-state
  regressions (``after_intent``, ``after_rename``, ``after_append``) inject a
  controlled crash at each point, assert an unrelated mutation is rejected, retry
  the terminal artifact, and verify it is the final registry publication + the
  attempt is sealed. ``is_sealed`` is fail-closed. Opt-in preserves Issue #12.
  Regressions: ``test_durable_seal_blocks_reconstructed_store_publication``,
  ``test_concurrent_publish_after_seal_is_rejected``,
  ``test_crash_window_intervening_publish_rejected_retry_completes_seal``,
  ``test_crash_injection_recovery_artifact_bound[after_intent/after_rename/after_append]``.

- **B2 (vacuous failure matrix + truthful mismatch lifecycle):** the missing-
  evidence regression references a CONCRETE UNREGISTERED checkpoint artifact and
  asserts exactly `ManifestBindingError` at the real `generate()`→`publish()`
  boundary. The canonical failing-comparison path is driven by `--force-mismatch`
  (a real computational divergence): R1 closes telemetry as `failed`
  (`checkpoint_failure`), publishes a `status="failed"` terminal manifest
  (`outcome_diagnostic="compatibility_mismatch"`), publishes a `decision="fail"`
  comparison report, and the CLI exits 1. An injected failure can never be
  mistaken for a successful canonical run. Regressions:
  `test_missing_evidence_artifact_raises_manifest_binding_error`,
  `test_cli_exits_nonzero_on_computational_mismatch`,
  `test_incomplete_vs_corrupt_telemetry_diagnostics`.
- **B3 (stale project evidence):** `PROJECT_STATE.md` latest-experiment now cites
  commit E `991b50a` and digest `925fbd96…`; the superseded
  `58acd64`/`b2abe623…`, `f0730e1`, and `fc043c1` evidence runs are marked
  invalidated.
- **B4 (resource measurements):** §8 records measured wall-clock and peak memory
  via the committed `scripts/measure_smoke_gate.py` (reproducible; see §8 for the
  exact command and current values).
- **B4-r1 (missing-parent typed layer):** a resume naming a never-registered
  parent checkpoint is rejected at authoritative load with
  `CheckpointLineageError` (the binding amendment's required typed layer).
  Regression: `test_missing_parent_reference_raises_checkpoint_lineage_error`.

## 10. Limitations

- This is an infrastructure sanity check, not a model-quality claim. The toy
  model is intentionally minimal; its loss movement confirms the training loop
  is wired correctly, not that the model is useful.
- The exact-byte contract is single-environment (§7).
- The smoke state provider/factory are scoped to `expertforge.smoke`; they are
  not a production D0 training adapter.

## 11. Validation (re-run on the final review head H)

The complete canonical validation passes on H:

```
uv sync --locked
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src tests
uv run --locked pytest
uv run --locked python -c "import expertforge"
uv run --locked expertforge-config configs/smoke.yaml > /dev/null
uv run --locked expertforge-config configs/m0-smoke-gate.yaml > /dev/null
uv run --locked python scripts/validate_repository.py all
uv run --locked pytest -m "smoke and not accelerator"
uv run --locked python -m expertforge.smoke.run --artifact-root runs/m0-smoke
uv lock --check
```

Full suite on H: 1382 passed, 20 skipped (platform/optional-dependency skips
only). The permanent CI smoke job additionally runs the locked smoke tier and
the one-command gate on Ubuntu 24.04.

## 12. Milestone 0 decision

**PASS.** All eleven Milestone 0 child issues (#4–#14) are delivered; the
experimental substrate composes end-to-end with exact computational-state
reproducibility across an interrupt/resume boundary. D0 (dense model training)
planning may begin.
