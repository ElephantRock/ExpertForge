# Milestone 0 End-to-End Smoke & Recovery Gate — Acceptance Report

**Issue:** #14 (closes #13 and #14 together)  
**Maturity stage:** Milestone 0 (experimental substrate)  
**Classification:** `smoke_test`  
**Decision:** **PASS**  
**Normative references:** design `5151917939`, amendment `5152078541` (A–P), authorization/clarifications `5152259715`, and the blocking-review sequence through `4838854599`.

## 1. Objective

Prove that the complete Milestone 0 substrate composes through a deterministic train → checkpoint → interrupt → restore → continue → validate → generate → compare → terminal-manifest workflow.

The acceptance comparison is **U0@N versus R1@N** over computational state. Run IDs, attempt IDs, timestamps, provenance containers, artifact IDs, manifest bytes, and tar containers are intentionally excluded from computational equality.

## 2. Required lifecycle

Each attempt publishes evidence in this order:

1. Resolved configuration.
2. Source and environment provenance.
3. Checkpoints and generated output as applicable.
4. Closed and registered telemetry.
5. Comparison or acceptance report.
6. Experiment manifest as the final registry publication.
7. Durable terminal seal.

The terminal seal is artifact-bound and crash-safe. A durable `sealing` intent containing the exact terminal artifact ID is committed before terminal publication. While that intent exists, every unrelated mutation is rejected; only the exact intended terminal artifact may retry and complete the seal.

## 3. Acceptance topology

- **U0:** uninterrupted execution from initialization through update N.
- **R0:** execution through update K, checkpoint publication, and truthful interruption.
- **R1:** a new attempt retaining R0's run lineage, authoritatively loading R0's checkpoint, restoring all state with RNG restored last, and continuing from K through N.

Frozen gate values:

| Field | Value |
|---|---:|
| model dimension / layers / heads / FFN | 16 / 1 / 1 / 32 |
| seed | 7 |
| batch size / sequence length | 2 / 8 |
| tokens per update | 16 |
| N updates | 8 |
| K updates | 2 |
| loss-improvement threshold | 0.1 |

## 4. Final sealing correction

The final TOCTOU correction is implemented by the public `ArtifactStore` in `expertforge.artifacts.secure_store`.

On POSIX:

- The sealing-intent path is opened first with no-follow, nonblocking, close-on-exec flags where available.
- The opened descriptor is `fstat`-verified as a regular file.
- The descriptor remains open while the pathname is re-inspected.
- Path and descriptor identities are compared while the original inode is pinned by the open descriptor, so unlink/recreate cannot be hidden by immediate inode reuse.
- The artifact ID is read only from the verified descriptor, with a 256-byte bound, strict UTF-8 decoding, and closed artifact-ID validation.

On Windows:

- A pre/open/post identity tuple is compared.
- The tuple includes device, inode, kernel-managed change time, size, and file type.
- Missing or zero identity fields fail closed with `ArtifactStoreError`.

The existing deterministic regression uses `sealing_read_inject` to replace the marker during the verification interval. The locked smoke tier now proves that this replacement is rejected through a typed error.

## 5. Designated clean evidence

**Evidence source commit E:** `42c7de1adbc07a6645cc01abc80c9a8db2c3b91f`

E contains all evidence-bearing implementation changes and no report/state synchronization changes after it. The designated evidence is the exact-E pull-request CI run:

- **Workflow run:** `30753390845`
- **Platform:** Ubuntu 24.04 / Linux x86_64
- **Python:** 3.11.15
- **NumPy:** 2.4.6
- **Locked environment:** 22 packages

Permanent CI results on E:

| Tier | Result |
|---|---:|
| formatting / Ruff / mypy / import / config / lock / policy | pass |
| fast CPU suite | 1248 passed |
| portable CPU integration suite | 114 passed |
| locked smoke tier | 39 passed |
| one-command smoke gate | `SMOKE GATE: PASS` |

The repository contains 1402 collected tests. The three permanent non-accelerator partitions cover 1401 tests; the remaining test is accelerator-scoped.

## 6. Observed gate result

| Measurement | Value |
|---|---|
| `passed` | `true` |
| `computational_equal` | `true` |
| mismatches | `[]` |
| initial validation loss | 2.8403357104342373 |
| U0 final validation loss | 2.4023344710684476 |
| R1 final validation loss | 2.4023344710684476 |
| loss improvement | 0.4380012393657897 |
| threshold | 0.1 |
| U0 computational digest | `925fbd9610757842e3159ef3e55202a4ef10c3c62a5547ad9135c1b7dbd5eab7` |
| R1 computational digest | `925fbd9610757842e3159ef3e55202a4ef10c3c62a5547ad9135c1b7dbd5eab7` |
| deterministic generated-sample digest | `f1c18145be41f0ed2b50b33db1bfe6fbddb11b748a7211fb5d327494f5772801` |

U0 and R1 validation losses and computational digests are identical. Loss movement exceeds the predeclared threshold.

## 7. Evidence identities

Specification fingerprint:

`spec-v1-sha256-4c80700fbf8120e4735d59643868572d70d822b5439368b3772890e2c28fe791`

| Attempt | Run ID | Attempt ID | Terminal manifest artifact ID |
|---|---|---|---|
| U0 | `run-20260802t150224z-4c80700fbf81-af3358229dca2b25399c` | `attempt-20260802t150224z-53695a721583258de126` | `artifact-v1-sha256-7abdefec85b4fc1de1d271bc5b2249ccbb10214d1757ddbb8d73014654bed4a4` |
| R0 | `run-20260802t150224z-4c80700fbf81-6b86f10107c1342279bb` | `attempt-20260802t150224z-ecfe46afff4474030c32` | `artifact-v1-sha256-b27925d17a19755d77c1654e4d0d7d75d6c8bab5af361909310e63ed8dcdc3b0` |
| R1 | `run-20260802t150224z-4c80700fbf81-6b86f10107c1342279bb` | `attempt-20260802t150224z-03dedafada84acbaf512` | `artifact-v1-sha256-bb4a55b6f13febe324aade495e773413a8af043cd305d734a5029be6279616ce` |

R1 retains R0's run ID and has a distinct attempt ID. R0's manifest is interrupted with explicit partial-evidence diagnostics; U0 and R1 are completed with complete core evidence. The CI summary does not print the K-checkpoint artifact ID, but R1's authoritative restore and lineage checks passed and the terminal manifests bind the checkpoint evidence.

## 8. Regression coverage

The permanent suite covers:

- restoration before any state-advancing R1 work;
- held-out validation;
- canonical comparison diagnostics;
- configuration and provenance first;
- public checkpoint serialization APIs;
- required telemetry events;
- compatibility, lineage, missing-evidence, and comparison-failure paths;
- absence of a publish-capable store in `SmokeAttemptResult`;
- real multi-head and multi-layer model behavior;
- durable intent write and file/directory fsync ordering;
- exact artifact-bound recovery after crashes at `after_intent`, `after_rename`, and `after_append`;
- rejection of unrelated and different-target publications during a pending intent;
- malformed, unreadable, symlinked, oversized, invalid-UTF-8, and invalid-ID intent states;
- rejection of standalone sealing while an artifact-bound intent exists;
- final registry-sequence verification against the manifest's `initial_publication` entry;
- public-API retention-transition counterexample;
- in-lock sealing rechecks for every mutating path, including external registration;
- rejection of nonregular `sealed` paths while preserving the recovery intent;
- deterministic path replacement during intent reading.

## 9. Reproducibility scope

Exact-byte reproducibility is claimed only for the same locked ExpertForge, Python, NumPy, and supported CPU environment. Cross-platform or cross-BLAS bitwise equivalence is not claimed without separate evidence.

The toy NumPy decoder is an infrastructure proof, not a production model architecture or D0 precursor.

## 10. Milestone decision

**PASS.** The designated clean implementation commit passes the permanent validation hierarchy and demonstrates exact computational-state recovery across U0/R0/R1. The artifact lifecycle terminates in a final manifest followed by a durable seal that rejects every later mutation.

The final review head H contains only this report and project-state synchronization after E. Exact-head CI on H is required before merge.
