"""Optional hardware/topology providers with typed models (Issue #7 review item 4).

Returns frozen typed models (``AcceleratorInfo``, ``TopologyInfo``,
``DeviceInfo``, ``HardwareAggregate``). Device ordinals are stable; memory is
typed numeric. Accelerator-framework package versions are identified. No
serials/MAC/UUIDs/IPs.
"""

from __future__ import annotations

import subprocess
from enum import StrEnum
from typing import Literal

from expertforge.provenance.record import (
    AcceleratorInfo,
    DeviceInfo,
    HardwareAggregate,
    TopologyInfo,
)

__all__ = [
    "FieldStatus",
    "capture_accelerator",
    "capture_hardware",
    "capture_topology",
]


class FieldStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"
    REDACTED = "redacted"


def _stable_reason(
    exc: BaseException,
) -> Literal["io_error", "timeout", "decode_error", "duplicate_device_ordinals", "not_found"]:
    """Map a capture exception to a stable accelerator error-reason code.

    The result MUST be one of the closed Literal domain codes on
    :class:`AcceleratorInfo.reason`. An unrecognized exception type degrades to
    ``"io_error"`` (a generic capture-failure code) rather than emitting a
    free-text code that would be rejected by the Literal-validated field.
    """
    if isinstance(exc, FileNotFoundError):
        return "not_found"
    if isinstance(exc, subprocess.TimeoutExpired):
        return "timeout"
    return "io_error"


# Common CUDA runtime distribution names published on PyPI. The first one
# found via importlib.metadata wins; the rest are not queried.
_CUDA_RUNTIME_PACKAGES = (
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-runtime-cu11",
    "nvidia-cuda-runtime-cu13",
)


def _detect_runtime_version() -> str | None:
    """Best-effort CUDA runtime distribution version via importlib.metadata.

    Probes common ``nvidia-cuda-runtime-*`` distributions. This is the CUDA
    *runtime* (not a deep-learning framework) — the naming reflects that.
    """
    try:
        from importlib import metadata as md
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib
        return None
    for pkg in _CUDA_RUNTIME_PACKAGES:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            continue
        except Exception:
            return None
    return None


def capture_accelerator(*, nvidia_smi: str = "nvidia-smi", timeout: float = 5.0) -> AcceleratorInfo:
    """Best-effort accelerator discovery via ``nvidia-smi``.

    Queries per-device index/name/memory/driver and derives count from validated
    rows. Devices are normalized by ordinal. No serials/MAC/UUIDs.
    """
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return AcceleratorInfo(status=FieldStatus.UNAVAILABLE.value)
    except (subprocess.SubprocessError, OSError) as e:
        return AcceleratorInfo(status=FieldStatus.ERROR.value, reason=_stable_reason(e))

    if result.returncode != 0:
        return AcceleratorInfo(status=FieldStatus.UNAVAILABLE.value)

    try:
        lines = result.stdout.decode("utf-8", errors="replace").strip().splitlines()
    except (UnicodeDecodeError, OSError):
        return AcceleratorInfo(status=FieldStatus.ERROR.value, reason="decode_error")
    if not lines:
        return AcceleratorInfo(status=FieldStatus.UNAVAILABLE.value)

    devices: list[DeviceInfo] = []
    for line in lines:
        # Parse each line independently. A single malformed nvidia-smi row
        # (truncated output, non-numeric index/memory, etc.) must not abort
        # the whole capture — skip it and continue with the valid rows.
        try:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            ordinal = int(parts[0])
            try:
                mem = int(parts[2])
            except ValueError:
                mem = None
            devices.append(
                DeviceInfo(
                    ordinal=ordinal,
                    model=parts[1],
                    memory_total_mib=mem,
                    driver_version=parts[3],
                )
            )
        except (ValueError, IndexError):
            continue
    if not devices:
        return AcceleratorInfo(status=FieldStatus.UNAVAILABLE.value)

    # Sort by ordinal for deterministic ordering.
    devices.sort(key=lambda d: d.ordinal)
    runtime_version = _detect_runtime_version()
    try:
        return AcceleratorInfo(
            status=FieldStatus.AVAILABLE.value,
            framework="cuda",
            # CUDA is not a deep-learning framework (Torch/JAX/TF are); only the
            # runtime distribution version is recorded under runtime_version.
            framework_version=None,
            runtime_version=runtime_version,
            # The binding decision says unavailable capabilities must be recorded
            # as 'unavailable', not 'not_applicable' — we did not probe precision.
            precision_status=FieldStatus.UNAVAILABLE.value,
            device_count=len(devices),
            devices=tuple(devices),
        )
    except ValueError:
        # The AcceleratorInfo model_validator rejects duplicate ordinals. A
        # malformed nvidia-smi output that yields two rows with the same ordinal
        # must degrade to error rather than abort the whole capture.
        return AcceleratorInfo(
            status=FieldStatus.ERROR.value,
            reason="duplicate_device_ordinals",
        )


# Topology environment allowlist — a narrow set of env vars that are safe to
# read for rank/world_size detection.
_TOPO_ENV_ALLOWLIST = {
    "RANK": int,
    "WORLD_SIZE": int,
    "LOCAL_RANK": int,
    "LOCAL_WORLD_SIZE": int,
    "NODE_RANK": int,
    "NNODES": int,
}


def _detect_topology_from_env() -> tuple[dict[str, int], tuple[str, ...]]:
    """Read a narrow allowlist of topology env vars (no coordinator addresses).

    Returns ``(detected, warnings)``. Values that are present but cannot be
    parsed as the expected integer type are NOT silently discarded — they are
    recorded as typed warnings so a misconfigured launcher is visible rather
    than silently dropping the rank/world_size.
    """
    import os

    detected: dict[str, int] = {}
    warnings: list[str] = []
    for key, caster in _TOPO_ENV_ALLOWLIST.items():
        val = os.environ.get(key)
        if val is not None:
            try:
                detected[key] = caster(val)
            except ValueError:
                warnings.append(f"invalid_topology_env_value:{key}")
    return detected, tuple(sorted(set(warnings)))


def capture_topology(
    *,
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
    node_count: int | None = None,
    backend: str | None = None,
) -> TopologyInfo:
    """Capture distributed topology.

    Explicit input takes precedence; then the narrow environment allowlist is
    consulted. No coordinator addresses or IPs are recorded.

    Note: invalid numeric topology env values are surfaced via
    :func:`capture_hardware` (as ``topology_warnings``) rather than here, since
    a frozen ``TopologyInfo`` cannot carry ad-hoc warning text without a new
    field. ``capture_hardware`` is the recommended aggregator when env-warning
    visibility is required.
    """
    env, _warnings = _detect_topology_from_env()

    resolved_rank = rank if rank is not None else env.get("RANK")
    resolved_world = world_size if world_size is not None else env.get("WORLD_SIZE")
    resolved_local_rank = local_rank if local_rank is not None else env.get("LOCAL_RANK")
    resolved_nodes = node_count if node_count is not None else env.get("NNODES")

    if resolved_rank is None and resolved_world is None and not env:
        # Check if any partial explicit input was supplied (local_rank, node_count, backend).
        # If so, it's an error — partial topology without rank/world_size.
        has_partial_input = any(v is not None for v in (local_rank, node_count, backend))
        if has_partial_input:
            return TopologyInfo(
                status=FieldStatus.ERROR.value,
                reason="partial_explicit_input_without_rank_world_size",
            )
        return TopologyInfo(status=FieldStatus.NOT_APPLICABLE.value)

    # rank and world_size must both be present or both absent.
    if (resolved_rank is not None) != (resolved_world is not None):
        return TopologyInfo(
            status=FieldStatus.ERROR.value,
            reason="rank_and_world_size_required_together",
        )

    # If only partial env data (LOCAL_RANK etc.) without RANK/WORLD_SIZE → error.
    if resolved_rank is None and resolved_world is None and env:
        return TopologyInfo(
            status=FieldStatus.ERROR.value,
            reason="partial_topology_env_without_rank_world_size",
        )

    # Cross-field validation: rank < world_size.
    if resolved_rank is not None and resolved_world is not None:
        if resolved_rank >= resolved_world:
            return TopologyInfo(
                status=FieldStatus.ERROR.value,
                reason="rank_must_be_less_than_world_size",
            )

    # local_rank < world_size when both are present.
    if resolved_local_rank is not None and resolved_world is not None:
        if resolved_local_rank >= resolved_world:
            return TopologyInfo(
                status=FieldStatus.ERROR.value,
                reason="invalid_local_rank",
            )

    # An explicitly-supplied node_count of 0 (or a parsed env NNODES=0) is
    # invalid — at least one node is required for a real run. Do NOT silently
    # coerce to 1; surface an error.
    if resolved_nodes is not None and resolved_nodes < 1:
        return TopologyInfo(
            status=FieldStatus.ERROR.value,
            reason="node_count_must_be_positive",
        )

    return TopologyInfo(
        status=FieldStatus.AVAILABLE.value,
        rank=resolved_rank,
        local_rank=resolved_local_rank,
        world_size=resolved_world,
        node_count=resolved_nodes,
        backend=backend,
    )


def capture_hardware(
    *,
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
    node_count: int | None = None,
    backend: str | None = None,
    nvidia_smi: str = "nvidia-smi",
) -> HardwareAggregate:
    """Aggregate hardware + topology capture as a typed frozen model.

    Returns a :class:`HardwareAggregate` (not a bare dict) so the combined
    result is validated and round-trips through JSON. ``topology_warnings``
    surfaces invalid numeric topology env values observed but not parseable.
    """
    _env, topology_warnings = _detect_topology_from_env()
    return HardwareAggregate(
        accelerator=capture_accelerator(nvidia_smi=nvidia_smi),
        topology=capture_topology(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            node_count=node_count,
            backend=backend,
        ),
        topology_warnings=topology_warnings,
    )
