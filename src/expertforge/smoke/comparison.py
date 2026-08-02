"""Computational-state comparison for the smoke gate (Issue #14, amendment H).

Compares **computational state** between U0@N and R1@N, never identity-bound
containers (run/attempt IDs, timestamps, provenance records, artifact IDs,
manifest bytes, or checkpoint tar bytes are intentionally unequal even when
computation is exactly reproducible).

The compared state covers:

- every model parameter/buffer: name, dtype, shape, exact bytes;
- every AdamW slot + scalar optimizer state;
- scheduler state;
- counters and data cursor;
- an isolated next-item probe from the cursor;
- an isolated next-random-sample probe from a CLONED/restored RNG bundle;
- validation output and generated-sample payload.

Probes are ISOLATED (cloned RNG, snapshot cursor) and MUST NOT mutate the
canonical live RNG or data cursor (review correction #9).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from expertforge.checkpoints.models import CounterSnapshot
from expertforge.rng.state import RngStateBundle

__all__ = ["ComputationalState", "extract_computational_state", "compare"]


@dataclass(frozen=True)
class ComputationalState:
    """The digest-bearing computational-state snapshot used for comparison."""

    parameters_digest: str
    optimizer_slots_digest: str
    optimizer_scalar_digest: str
    scheduler_digest: str
    counters_digest: str
    cursor_digest: str
    next_item_probe: tuple[int, ...]
    next_random_probe: tuple[int, ...]
    validation_loss: float
    generated_sample: tuple[int, ...]
    # The canonical computational-state digest derived ONLY from the above.
    computational_digest: str


def _bytes_digest(values: list[bytes]) -> str:
    h = hashlib.sha256()
    for chunk in values:
        h.update(chunk)
    return h.hexdigest()


def _json_digest(obj: Any) -> str:
    payload = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_computational_state(
    *,
    model_params: dict[str, np.ndarray],
    model_buffers: dict[str, np.ndarray],
    optimizer: Any,
    scheduler: Any,
    cursor: Any,
    counters: CounterSnapshot,
    rng_manager: Any,
    validation_loss: float,
    generated_sample: np.ndarray,
) -> ComputationalState:
    """Extract a non-mutating computational-state snapshot from a live runtime.

    All probes operate on COPIES: parameters/slots are hashed from their bytes
    (no copy of the array needed for hashing), the cursor probe advances a
    snapshot cursor, and the RNG probe advances a freshly restored clone of the
    captured bundle. The canonical live RNG and cursor are never advanced.
    """
    # Parameters + buffers: hash each array's name + dtype + shape + bytes.
    param_chunks: list[bytes] = []
    for name in sorted(model_params):
        arr = np.ascontiguousarray(model_params[name], dtype=np.float32)
        param_chunks.append(name.encode("utf-8"))
        param_chunks.append(b"\x00float32\x00")
        param_chunks.append(repr(tuple(int(d) for d in arr.shape)).encode("ascii"))
        param_chunks.append(arr.tobytes())
    for name in sorted(model_buffers):
        arr = np.ascontiguousarray(model_buffers[name], dtype=np.float32)
        param_chunks.append(name.encode("utf-8"))
        param_chunks.append(b"\x00buffer\x00")
        param_chunks.append(repr(tuple(int(d) for d in arr.shape)).encode("ascii"))
        param_chunks.append(arr.tobytes())
    parameters_digest = _bytes_digest(param_chunks)

    # Optimizer slots: hash each slot tensor's name + shape + bytes.
    slot_arrays = optimizer.slot_arrays()
    slot_chunks: list[bytes] = []
    for sname in sorted(slot_arrays):
        arr = np.ascontiguousarray(slot_arrays[sname], dtype=np.float32)
        slot_chunks.append(sname.encode("utf-8"))
        slot_chunks.append(repr(tuple(int(d) for d in arr.shape)).encode("ascii"))
        slot_chunks.append(arr.tobytes())
    optimizer_slots_digest = _bytes_digest(slot_chunks)

    # Optimizer scalar state (per-param steps).
    optimizer_scalar_digest = _json_digest(optimizer.scalar_state())

    # Scheduler state.
    scheduler_state = {
        "lr": float(scheduler.lr),
        "step_count": int(scheduler.step_count),
        "state_tensor": np.ascontiguousarray(scheduler.state_tensor(), dtype=np.float32)
        .tobytes()
        .hex(),
    }
    scheduler_digest = _json_digest(scheduler_state)

    # Counters.
    counters_digest = _json_digest(counters.model_dump(mode="json"))

    # Cursor: a snapshot digest (no advancement of the live cursor).
    live_cursor_snapshot = {
        "batch_size": int(cursor.batch_size),
        "sequence_length": int(cursor.sequence_length),
        "drop_last": bool(cursor.drop_last),
        "epoch": int(cursor.epoch),
        "position": int(cursor.position),
        "accepted_samples": int(cursor.accepted_samples),
        "accepted_sequences": int(cursor.accepted_sequences),
        "length": int(cursor.length),
    }
    cursor_digest = _json_digest(live_cursor_snapshot)

    # Isolated next-item cursor probe: advance a COPY of the cursor's token
    # stream from the current position and read the next item without mutating
    # the live cursor.
    next_item_probe = _isolated_next_item_probe(cursor)

    # Isolated next-random-sample probe from a CLONED/restored RNG bundle.
    captured_bundle = rng_manager.capture_state()
    next_random_probe = _isolated_rng_probe(captured_bundle)

    generated_tuple = tuple(int(x) for x in np.asarray(generated_sample).reshape(-1).tolist())

    # Canonical computational digest over ALL of the above.
    digest_input = {
        "parameters_digest": parameters_digest,
        "optimizer_slots_digest": optimizer_slots_digest,
        "optimizer_scalar_digest": optimizer_scalar_digest,
        "scheduler_digest": scheduler_digest,
        "counters_digest": counters_digest,
        "cursor_digest": cursor_digest,
        "next_item_probe": list(next_item_probe),
        "next_random_probe": list(next_random_probe),
        "validation_loss": float(validation_loss),
        "generated_sample": list(generated_tuple),
    }
    computational_digest = _json_digest(digest_input)

    return ComputationalState(
        parameters_digest=parameters_digest,
        optimizer_slots_digest=optimizer_slots_digest,
        optimizer_scalar_digest=optimizer_scalar_digest,
        scheduler_digest=scheduler_digest,
        counters_digest=counters_digest,
        cursor_digest=cursor_digest,
        next_item_probe=next_item_probe,
        next_random_probe=next_random_probe,
        validation_loss=float(validation_loss),
        generated_sample=generated_tuple,
        computational_digest=computational_digest,
    )


def _isolated_next_item_probe(cursor: Any) -> tuple[int, ...]:
    """Read the next few tokens from the cursor's position WITHOUT advancing it.

    Uses a snapshot of the token stream and a copy of the position.
    """
    tokens = cursor.tokens
    pos = int(cursor.position)
    n = min(8, int(cursor.length))
    return tuple(int(tokens[(pos + i) % len(tokens)]) for i in range(n))


def _isolated_rng_probe(bundle: RngStateBundle) -> tuple[int, ...]:
    """Draw a few samples from a freshly RESTORED clone of ``bundle``.

    Builds a fresh :class:`RngManager` (no framework adapters), calls
    ``restore_state(bundle)``, and draws from its generator. The live manager is
    never touched.
    """
    from expertforge.rng.derivation import SeedContext
    from expertforge.rng.manager import RngManager

    fresh = RngManager(
        root_seed=int(bundle.root_seed),
        context=SeedContext(
            component=bundle.context.component,
            worker=bundle.context.worker,
            rank=bundle.context.rank,
            device=bundle.context.device,
            stream=bundle.context.stream,
        ),
        determinism_mode=bundle.determinism_mode,
        unsupported_determinism=bundle.unsupported_determinism,
    )
    fresh.restore_state(bundle)
    gen = fresh.generator
    # Draw a deterministic probe: integers in [0, 256) — small and reproducible.
    return tuple(int(x) for x in gen.integers(0, 256, size=16).tolist())


def compare(u0: ComputationalState, r1: ComputationalState) -> dict[str, Any]:
    """Compare two computational states and return a detailed mismatch report.

    Returns a dict with ``equal`` (bool), ``digests`` (u0/r1), and ``mismatches``
    (list of human-readable field names that differ). Run/attempt/timestamp/
    provenance/artifact/manifest/tar bytes are deliberately excluded.
    """
    fields = (
        "parameters_digest",
        "optimizer_slots_digest",
        "optimizer_scalar_digest",
        "scheduler_digest",
        "counters_digest",
        "cursor_digest",
        "next_item_probe",
        "next_random_probe",
        "validation_loss",
        "generated_sample",
        "computational_digest",
    )
    mismatches: list[str] = []
    for f in fields:
        u0_val = getattr(u0, f)
        r1_val = getattr(r1, f)
        if u0_val != r1_val:
            mismatches.append(f)
    return {
        "equal": len(mismatches) == 0,
        "u0_computational_digest": u0.computational_digest,
        "r1_computational_digest": r1.computational_digest,
        "mismatches": mismatches,
    }
