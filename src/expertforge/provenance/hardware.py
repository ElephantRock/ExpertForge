"""Optional hardware/topology providers (Issue #7 decision: explicit degradation).

Optional fields use typed statuses: ``available``, ``unavailable``,
``not_applicable``, ``error``, ``redacted``. A missing accelerator detector
must not break CPU execution. Detector errors use stable sanitized reason codes;
raw exception messages and command output are not recorded. V1 adds no mandatory
runtime dependency; ``nvidia-smi`` is an optional external detector. Precision
support is recorded only when a trusted detector reports it — never inferred
from a GPU model name.

No hardware serials, MAC addresses, device UUIDs, or coordinator addresses/IPs
are ever recorded.
"""

from __future__ import annotations

import subprocess
from enum import StrEnum
from typing import Any

__all__ = [
    "FieldStatus",
    "capture_accelerator",
    "capture_hardware",
    "capture_topology",
]


class FieldStatus(StrEnum):
    """Typed status for optional provenance fields."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"
    REDACTED = "redacted"


def _stable_reason(exc: BaseException) -> str:
    """Reduce an exception to a stable, non-leaking reason code."""
    if isinstance(exc, FileNotFoundError):
        return "not_found"
    if isinstance(exc, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(exc, (OSError, subprocess.SubprocessError)):
        return "io_error"
    return type(exc).__name__.lower()


def capture_accelerator(*, nvidia_smi: str = "nvidia-smi", timeout: float = 5.0) -> dict[str, Any]:
    """Best-effort accelerator discovery via ``nvidia-smi``.

    Returns ``{"status": "unavailable"}`` when the detector is absent or the
    host is CPU-only. Detector failures return ``{"status": "error", "reason":
    <stable code>}``. Records only: device count, model name, memory total, and
    driver/CUDA versions — never serials, MACs, UUIDs, or raw output.
    """
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=count,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {"status": FieldStatus.UNAVAILABLE.value}
    except (subprocess.SubprocessError, OSError) as e:
        return {"status": FieldStatus.ERROR.value, "reason": _stable_reason(e)}

    if result.returncode != 0:
        return {"status": FieldStatus.UNAVAILABLE.value}

    # Parse CSV lines: each GPU is "count,name,memory,driver". Take the first as
    # representative; record the device count separately.
    try:
        lines = result.stdout.decode("utf-8", errors="replace").strip().splitlines()
    except (UnicodeDecodeError, OSError):
        return {"status": FieldStatus.ERROR.value, "reason": "decode_error"}
    if not lines:
        return {"status": FieldStatus.UNAVAILABLE.value}

    devices: list[dict[str, Any]] = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        devices.append(
            {
                # Model name only — no serial/UUID/MAC. Precision is NOT inferred
                # from the name; it requires a trusted detector (deferred).
                "model": parts[1],
                "memory_total_mib": parts[2],
                "driver_version": parts[3],
            }
        )
    if not devices:
        return {"status": FieldStatus.UNAVAILABLE.value}
    # CUDA/runtime version: query separately; degrade if unavailable.
    cuda = _query_cuda_version(nvidia_smi, timeout)
    return {
        "status": FieldStatus.AVAILABLE.value,
        "framework": "cuda",
        "device_count": len(devices),
        "devices": devices,
        "cuda_version": cuda,
    }


def _query_cuda_version(nvidia_smi: str, timeout: float) -> dict[str, Any]:
    """Best-effort CUDA version query; degrades to unavailable."""
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return {"status": FieldStatus.UNAVAILABLE.value}
    if result.returncode != 0:
        return {"status": FieldStatus.UNAVAILABLE.value}
    # nvidia-smi does not directly report CUDA version in query-gpu; rely on
    # the driver version already captured. Mark CUDA as not directly reported.
    return {"status": FieldStatus.NOT_APPLICABLE.value, "reason": "not_directly_reported"}


def capture_topology(
    *,
    rank: int | None = None,
    world_size: int | None = None,
) -> dict[str, Any]:
    """Capture distributed topology.

    With no explicit input, returns ``not_applicable`` (single-process). Explicit
    ``rank``/``world_size`` are recorded; coordinator addresses/IPs never are.
    """
    if rank is None and world_size is None:
        return {"status": FieldStatus.NOT_APPLICABLE.value}
    if rank is None or world_size is None:
        return {
            "status": FieldStatus.ERROR.value,
            "reason": "rank_and_world_size_required_together",
        }
    return {
        "status": FieldStatus.AVAILABLE.value,
        "rank": rank,
        "world_size": world_size,
    }


def capture_hardware(
    *,
    rank: int | None = None,
    world_size: int | None = None,
    nvidia_smi: str = "nvidia-smi",
) -> dict[str, Any]:
    """Aggregate hardware + topology capture."""
    return {
        "accelerator": capture_accelerator(nvidia_smi=nvidia_smi),
        "topology": capture_topology(rank=rank, world_size=world_size),
    }
