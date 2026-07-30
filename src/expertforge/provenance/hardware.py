"""Optional hardware/topology providers with typed models (Issue #7 review item 4).

Returns frozen typed models (``AcceleratorInfo``, ``TopologyInfo``,
``DeviceInfo``). Device ordinals are stable; memory is typed numeric.
Accelerator-framework package versions are identified. No serials/MAC/UUIDs/IPs.
"""

from __future__ import annotations

import subprocess
from enum import StrEnum
from typing import Any

from expertforge.provenance.record import (
    AcceleratorInfo,
    DeviceInfo,
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


def _stable_reason(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return "not_found"
    if isinstance(exc, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(exc, (OSError, subprocess.SubprocessError)):
        return "io_error"
    return type(exc).__name__.lower()


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
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            ordinal = int(parts[0])
        except ValueError:
            continue
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
    if not devices:
        return AcceleratorInfo(status=FieldStatus.UNAVAILABLE.value)

    # Sort by ordinal for deterministic ordering.
    devices.sort(key=lambda d: d.ordinal)
    runtime_version = _detect_runtime_version()
    return AcceleratorInfo(
        status=FieldStatus.AVAILABLE.value,
        framework="cuda",
        framework_version=runtime_version,
        runtime_version=runtime_version,
        precision_status=FieldStatus.NOT_APPLICABLE.value,
        device_count=len(devices),
        devices=tuple(devices),
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


def _detect_topology_from_env() -> dict[str, int]:
    """Read a narrow allowlist of topology env vars (no coordinator addresses)."""
    detected: dict[str, int] = {}
    import os

    for key, caster in _TOPO_ENV_ALLOWLIST.items():
        val = os.environ.get(key)
        if val is not None:
            try:
                detected[key] = caster(val)
            except ValueError:
                pass
    return detected


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
    """
    env = _detect_topology_from_env()

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

    return TopologyInfo(
        status=FieldStatus.AVAILABLE.value,
        rank=resolved_rank,
        local_rank=resolved_local_rank,
        world_size=resolved_world,
        node_count=resolved_nodes or 1,
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
) -> dict[str, Any]:
    """Aggregate hardware + topology capture."""
    return {
        "accelerator": capture_accelerator(nvidia_smi=nvidia_smi),
        "topology": capture_topology(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            node_count=node_count,
            backend=backend,
        ),
    }
