"""Frozen D0 precision preflight (fail-closed environment verification).

Verifies the runtime environment against the typed ``D0PrecisionConfig``
before any training-path work may advance:

- a CUDA device is present and supports native BF16;
- device, host, and free-storage memory meet the frozen minimums;
- the dtype strategy is exactly bf16-autocast compute over float32 master
  parameters and float32 gradient/loss/optimizer state — bf16 needs no
  gradient scaler, so none is used (the checkpoint schema's scaler slot
  stays empty);
- fp16 is never authorized (the typed config pins ``fp16_authorized=False``).

Checks that cannot be measured on the current platform are recorded as
**failed** with an explanatory detail — the preflight is fail-closed, and a
host that cannot demonstrate the frozen minimums must not start a training
attempt. The canonical profile additionally pins
``canonical_fallback=none_fail_closed``: there is no degraded mode. The
qualification profile's float32 fallback is a *distinct non-equivalent
attempt* (it can never authorize D0.6), recorded here as a fallback note.

Deferred (contract flags A5/A7/A8): stream-edge semantics, non-data RNG
substreams, and generation-seed mapping belong to later tranches.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from expertforge.d0.errors import PrecisionPreflightError
from expertforge.d0.training._torch import require_torch

if TYPE_CHECKING:
    from expertforge.config.d0_models import D0PrecisionConfig

__all__ = [
    "DTYPE_STRATEGY",
    "PrecisionCheck",
    "PrecisionPreflightReport",
    "run_precision_preflight",
]

DTYPE_STRATEGY = "bf16_autocast_compute_fp32_master_params_fp32_optimizer_state_no_scaler"


@dataclass(frozen=True, slots=True)
class PrecisionCheck:
    """One named preflight check with its outcome."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PrecisionPreflightReport:
    """The complete, truthful preflight observation of one environment."""

    torch_version: str
    cuda_available: bool
    device_name: str | None
    native_bf16_supported: bool
    device_memory_bytes: int | None
    host_memory_bytes: int | None
    free_storage_bytes: int | None
    dtype_strategy: str
    canonical_fallback: str
    qualification_fallback: str
    checks: tuple[PrecisionCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def raise_if_failed(self) -> None:
        """Raise :class:`PrecisionPreflightError` if any check failed."""

        if self.passed:
            return
        failed = [f"{c.name}: {c.detail}" for c in self.checks if not c.passed]
        raise PrecisionPreflightError("D0 precision preflight failed: " + "; ".join(failed))


def _host_memory_bytes() -> int | None:
    """Total physical host memory, when the platform exposes it."""

    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1] == "kB":
                try:
                    return int(fields[0]) * 1024
                except ValueError:
                    return None
    return None


def run_precision_preflight(
    config: D0PrecisionConfig, *, artifact_root: Path
) -> PrecisionPreflightReport:
    """Observe the environment and evaluate every frozen precision check."""

    torch_module = require_torch()
    cuda_available = bool(torch_module.cuda.is_available())
    device_name: str | None = None
    device_memory: int | None = None
    native_bf16 = False
    if cuda_available:
        properties = torch_module.cuda.get_device_properties(0)
        device_name = str(properties.name)
        device_memory = int(properties.total_memory)
        native_bf16 = bool(torch_module.cuda.is_bf16_supported())

    host_memory = _host_memory_bytes()
    try:
        free_storage = int(shutil.disk_usage(artifact_root).free)
    except OSError:
        free_storage = None

    checks: list[PrecisionCheck] = [
        PrecisionCheck(
            name="cuda_device_present",
            passed=cuda_available,
            detail="CUDA device present" if cuda_available else "no CUDA device visible",
        ),
        PrecisionCheck(
            name="native_bf16",
            passed=native_bf16,
            detail=(
                "device reports native BF16 support"
                if native_bf16
                else "native BF16 support not available"
            ),
        ),
    ]
    if device_memory is not None:
        checks.append(
            PrecisionCheck(
                name="minimum_device_memory",
                passed=device_memory >= config.minimum_device_memory_bytes,
                detail=(
                    f"device memory {device_memory} bytes vs required "
                    f"{config.minimum_device_memory_bytes}"
                ),
            )
        )
    else:
        checks.append(
            PrecisionCheck(
                name="minimum_device_memory",
                passed=False,
                detail="unmeasurable without a CUDA device",
            )
        )
    if host_memory is not None:
        checks.append(
            PrecisionCheck(
                name="minimum_host_memory",
                passed=host_memory >= config.minimum_host_memory_bytes,
                detail=(
                    f"host memory {host_memory} bytes vs required "
                    f"{config.minimum_host_memory_bytes}"
                ),
            )
        )
    else:
        checks.append(
            PrecisionCheck(
                name="minimum_host_memory",
                passed=False,
                detail="unmeasurable on this platform (no /proc/meminfo)",
            )
        )
    if free_storage is not None:
        checks.append(
            PrecisionCheck(
                name="minimum_free_storage",
                passed=free_storage >= config.minimum_free_local_storage_bytes,
                detail=(
                    f"free storage {free_storage} bytes vs required "
                    f"{config.minimum_free_local_storage_bytes}"
                ),
            )
        )
    else:
        checks.append(
            PrecisionCheck(
                name="minimum_free_storage",
                passed=False,
                detail="unmeasurable (artifact root not usable)",
            )
        )
    checks.append(
        PrecisionCheck(
            name="fp16_not_authorized",
            passed=config.fp16_authorized is False,
            detail="fp16 is never authorized for D0",
        )
    )

    return PrecisionPreflightReport(
        torch_version=str(torch_module.__version__),
        cuda_available=cuda_available,
        device_name=device_name,
        native_bf16_supported=native_bf16,
        device_memory_bytes=device_memory,
        host_memory_bytes=host_memory,
        free_storage_bytes=free_storage,
        dtype_strategy=DTYPE_STRATEGY,
        canonical_fallback=str(config.canonical_fallback),
        qualification_fallback=str(config.qualification_fallback),
        checks=tuple(checks),
    )
