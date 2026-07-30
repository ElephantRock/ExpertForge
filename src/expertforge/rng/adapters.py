"""Explicit tensor-framework RNG adapter contract and lazy PyTorch adapter."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from expertforge.rng.derivation import DerivedSeed
from expertforge.rng.state import (
    DeterminismMode,
    FrameworkRngState,
    RngWarningCode,
    UnsupportedDeterminismPolicy,
)

__all__ = [
    "DeterminismUnavailableError",
    "FrameworkAdapterError",
    "FrameworkRngAdapter",
    "TorchRngAdapter",
]

FrameworkErrorCode = Literal[
    "framework_not_installed",
    "framework_api_unavailable",
    "framework_state_invalid",
    "framework_device_set_mismatch",
    "framework_restore_failed",
]


class FrameworkAdapterError(RuntimeError):
    """Typed framework-adapter failure with a stable non-secret code."""

    def __init__(self, code: FrameworkErrorCode) -> None:
        self.code = code
        super().__init__(code)


class DeterminismUnavailableError(RuntimeError):
    """Raised when reproducible mode cannot be enforced under error policy."""

    def __init__(self, code: Literal["framework_determinism_unavailable"]) -> None:
        self.code = code
        super().__init__(code)


@runtime_checkable
class FrameworkRngAdapter(Protocol):
    """Minimal explicit contract required by :class:`RngManager`."""

    @property
    def provider(self) -> str: ...

    def configure(
        self,
        mode: DeterminismMode,
        unsupported_policy: UnsupportedDeterminismPolicy,
    ) -> tuple[RngWarningCode, ...]: ...

    def seed(self, seed: DerivedSeed) -> tuple[RngWarningCode, ...]: ...

    def capture(self) -> tuple[FrameworkRngState, ...]: ...

    def validate_restore(self, states: Sequence[FrameworkRngState]) -> None: ...

    def restore(self, states: Sequence[FrameworkRngState]) -> None: ...


class TorchRngAdapter:
    """Lazily imported PyTorch CPU/CUDA RNG adapter.

    ``torch_module`` is injectable for deterministic, dependency-free contract
    tests. Production callers normally omit it, in which case PyTorch is imported
    only when this adapter is constructed.
    """

    def __init__(self, torch_module: Any | None = None) -> None:
        if torch_module is None:
            try:
                torch_module = importlib.import_module("torch")
            except ModuleNotFoundError as exc:
                raise FrameworkAdapterError("framework_not_installed") from exc
        self._torch = torch_module

    @property
    def provider(self) -> str:
        return "torch"

    def configure(
        self,
        mode: DeterminismMode,
        unsupported_policy: UnsupportedDeterminismPolicy,
    ) -> tuple[RngWarningCode, ...]:
        deterministic_api = getattr(self._torch, "use_deterministic_algorithms", None)
        if mode == "reproducible":
            if not callable(deterministic_api):
                if unsupported_policy == "error":
                    raise DeterminismUnavailableError("framework_determinism_unavailable")
                return ("framework_determinism_unavailable",)
            deterministic_api(True, warn_only=unsupported_policy == "warn")
            self._set_cudnn_flags(deterministic=True, benchmark=False)
            return ()

        if callable(deterministic_api):
            deterministic_api(False)
        self._set_cudnn_flags(deterministic=False, benchmark=True)
        return ()

    def seed(self, seed: DerivedSeed) -> tuple[RngWarningCode, ...]:
        manual_seed = getattr(self._torch, "manual_seed", None)
        if not callable(manual_seed):
            raise FrameworkAdapterError("framework_api_unavailable")
        manual_seed(seed.seed_u64)

        cuda = getattr(self._torch, "cuda", None)
        if cuda is None or not self._cuda_is_available(cuda):
            return ("accelerator_unavailable",)
        manual_seed_all = getattr(cuda, "manual_seed_all", None)
        if not callable(manual_seed_all):
            raise FrameworkAdapterError("framework_api_unavailable")
        manual_seed_all(seed.seed_u64)
        return ()

    def capture(self) -> tuple[FrameworkRngState, ...]:
        get_cpu_state = getattr(self._torch, "get_rng_state", None)
        if not callable(get_cpu_state):
            raise FrameworkAdapterError("framework_api_unavailable")
        states = [
            FrameworkRngState.from_bytes(
                provider=self.provider,
                device="cpu",
                payload=self._tensor_to_bytes(get_cpu_state()),
            )
        ]

        cuda = getattr(self._torch, "cuda", None)
        if cuda is not None and self._cuda_is_available(cuda):
            get_all = getattr(cuda, "get_rng_state_all", None)
            if not callable(get_all):
                raise FrameworkAdapterError("framework_api_unavailable")
            for ordinal, state in enumerate(get_all()):
                states.append(
                    FrameworkRngState.from_bytes(
                        provider=self.provider,
                        device=f"cuda:{ordinal}",
                        payload=self._tensor_to_bytes(state),
                    )
                )
        return tuple(sorted(states, key=lambda item: (item.provider, item.device)))

    def validate_restore(self, states: Sequence[FrameworkRngState]) -> None:
        if not states or any(state.provider != self.provider for state in states):
            raise FrameworkAdapterError("framework_state_invalid")
        devices = [state.device for state in states]
        if devices.count("cpu") != 1 or len(set(devices)) != len(devices):
            raise FrameworkAdapterError("framework_state_invalid")

        cuda_states = [state for state in states if state.device.startswith("cuda:")]
        expected_cuda = 0
        cuda = getattr(self._torch, "cuda", None)
        if cuda is not None and self._cuda_is_available(cuda):
            device_count = getattr(cuda, "device_count", None)
            if not callable(device_count):
                raise FrameworkAdapterError("framework_api_unavailable")
            expected_cuda = int(device_count())
        expected_devices = {"cpu", *(f"cuda:{index}" for index in range(expected_cuda))}
        if set(devices) != expected_devices or len(cuda_states) != expected_cuda:
            raise FrameworkAdapterError("framework_device_set_mismatch")

        for state in states:
            state.payload_bytes()

    def restore(self, states: Sequence[FrameworkRngState]) -> None:
        self.validate_restore(states)
        by_device = {state.device: state for state in states}
        set_cpu_state = getattr(self._torch, "set_rng_state", None)
        if not callable(set_cpu_state):
            raise FrameworkAdapterError("framework_api_unavailable")
        try:
            set_cpu_state(self._bytes_to_tensor(by_device["cpu"].payload_bytes()))
            cuda_devices = sorted(
                (device for device in by_device if device.startswith("cuda:")),
                key=lambda value: int(value.split(":", 1)[1]),
            )
            if cuda_devices:
                cuda = getattr(self._torch, "cuda", None)
                set_all = getattr(cuda, "set_rng_state_all", None)
                if not callable(set_all):
                    raise FrameworkAdapterError("framework_api_unavailable")
                set_all(
                    [
                        self._bytes_to_tensor(by_device[device].payload_bytes())
                        for device in cuda_devices
                    ]
                )
        except FrameworkAdapterError:
            raise
        except Exception as exc:
            raise FrameworkAdapterError("framework_restore_failed") from exc

    def _set_cudnn_flags(self, *, deterministic: bool, benchmark: bool) -> None:
        backends = getattr(self._torch, "backends", None)
        cudnn = getattr(backends, "cudnn", None) if backends is not None else None
        if cudnn is not None:
            if hasattr(cudnn, "deterministic"):
                cudnn.deterministic = deterministic
            if hasattr(cudnn, "benchmark"):
                cudnn.benchmark = benchmark

    @staticmethod
    def _cuda_is_available(cuda: Any) -> bool:
        is_available = getattr(cuda, "is_available", None)
        return bool(callable(is_available) and is_available())

    @staticmethod
    def _tensor_to_bytes(state: Any) -> bytes:
        value = state
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        tolist = getattr(value, "tolist", None)
        if not callable(tolist):
            raise FrameworkAdapterError("framework_state_invalid")
        try:
            return bytes(int(item) for item in tolist())
        except (TypeError, ValueError, OverflowError) as exc:
            raise FrameworkAdapterError("framework_state_invalid") from exc

    def _bytes_to_tensor(self, payload: bytes) -> Any:
        tensor = getattr(self._torch, "tensor", None)
        uint8 = getattr(self._torch, "uint8", None)
        if not callable(tensor) or uint8 is None:
            raise FrameworkAdapterError("framework_api_unavailable")
        return tensor(list(payload), dtype=uint8)
