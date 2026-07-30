"""Explicit tensor-framework RNG adapter contract and lazy PyTorch adapter."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast, runtime_checkable

from expertforge.rng.derivation import DerivedSeed, SeedContext, derive_substream_seed
from expertforge.rng.state import (
    DeterminismMode,
    FrameworkRngState,
    RngWarningCode,
    UnsupportedDeterminismPolicy,
    framework_state_sort_key,
)

__all__ = [
    "DeterminismUnavailableError",
    "FrameworkAdapterError",
    "FrameworkRngAdapter",
    "FrameworkSeedResult",
    "TorchRngAdapter",
]

FrameworkErrorCode = Literal[
    "framework_not_installed",
    "framework_api_unavailable",
    "framework_probe_failed",
    "framework_configure_failed",
    "framework_seed_failed",
    "framework_capture_failed",
    "framework_state_invalid",
    "framework_device_set_mismatch",
    "framework_restore_failed",
    "framework_configuration_restore_failed",
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


@dataclass(frozen=True)
class FrameworkSeedResult:
    """Exact provider streams and stable warnings produced by one seed operation."""

    derived_seeds: tuple[DerivedSeed, ...]
    warning_codes: tuple[RngWarningCode, ...] = ()

    def __post_init__(self) -> None:
        if not self.derived_seeds:
            raise ValueError("framework seed result must contain at least one derived seed")
        contexts = tuple(seed.context for seed in self.derived_seeds)
        if len(set(contexts)) != len(contexts):
            raise ValueError("framework seed result contains duplicate stream contexts")
        if self.warning_codes != tuple(sorted(self.warning_codes)):
            raise ValueError("framework seed result warning codes must be sorted")
        if len(set(self.warning_codes)) != len(self.warning_codes):
            raise ValueError("framework seed result warning codes must be unique")


@runtime_checkable
class FrameworkRngAdapter(Protocol):
    """Minimal explicit contract required by :class:`RngManager`."""

    @property
    def provider(self) -> str: ...

    def capture_configuration(self) -> object: ...

    def restore_configuration(self, snapshot: object) -> None: ...

    def configure(
        self,
        mode: DeterminismMode,
        unsupported_policy: UnsupportedDeterminismPolicy,
    ) -> tuple[RngWarningCode, ...]: ...

    def derive_seeds(self, root_seed: int, context: SeedContext) -> tuple[DerivedSeed, ...]: ...

    def seed(self, root_seed: int, context: SeedContext) -> FrameworkSeedResult: ...

    def capture(self) -> tuple[FrameworkRngState, ...]: ...

    def validate_restore(self, states: Sequence[FrameworkRngState]) -> None: ...

    def restore(self, states: Sequence[FrameworkRngState]) -> None: ...


@dataclass(frozen=True)
class _TorchConfigurationSnapshot:
    deterministic_algorithms: bool | None
    deterministic_warn_only: bool | None
    cudnn_deterministic: bool | None
    cudnn_benchmark: bool | None


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

    def capture_configuration(self) -> object:
        try:
            enabled_api = getattr(self._torch, "are_deterministic_algorithms_enabled", None)
            warn_api = getattr(
                self._torch,
                "is_deterministic_algorithms_warn_only_enabled",
                None,
            )
            enabled = bool(enabled_api()) if callable(enabled_api) else None
            warn_only = bool(warn_api()) if callable(warn_api) else None
            cudnn = self._cudnn_backend()
            cudnn_deterministic = (
                bool(cudnn.deterministic)
                if cudnn is not None and hasattr(cudnn, "deterministic")
                else None
            )
            cudnn_benchmark = (
                bool(cudnn.benchmark) if cudnn is not None and hasattr(cudnn, "benchmark") else None
            )
            return _TorchConfigurationSnapshot(
                deterministic_algorithms=enabled,
                deterministic_warn_only=warn_only,
                cudnn_deterministic=cudnn_deterministic,
                cudnn_benchmark=cudnn_benchmark,
            )
        except Exception as exc:
            raise FrameworkAdapterError("framework_capture_failed") from exc

    def restore_configuration(self, snapshot: object) -> None:
        if not isinstance(snapshot, _TorchConfigurationSnapshot):
            raise FrameworkAdapterError("framework_state_invalid")
        try:
            deterministic_api = getattr(self._torch, "use_deterministic_algorithms", None)
            if snapshot.deterministic_algorithms is not None:
                if not callable(deterministic_api):
                    raise FrameworkAdapterError("framework_api_unavailable")
                deterministic_api(
                    snapshot.deterministic_algorithms,
                    warn_only=bool(snapshot.deterministic_warn_only),
                )
            cudnn = self._cudnn_backend()
            if cudnn is not None:
                if snapshot.cudnn_deterministic is not None:
                    cudnn.deterministic = snapshot.cudnn_deterministic
                if snapshot.cudnn_benchmark is not None:
                    cudnn.benchmark = snapshot.cudnn_benchmark
        except FrameworkAdapterError:
            raise
        except Exception as exc:
            raise FrameworkAdapterError("framework_configuration_restore_failed") from exc

    def configure(
        self,
        mode: DeterminismMode,
        unsupported_policy: UnsupportedDeterminismPolicy,
    ) -> tuple[RngWarningCode, ...]:
        deterministic_api = getattr(self._torch, "use_deterministic_algorithms", None)
        if mode == "reproducible" and not callable(deterministic_api):
            if unsupported_policy == "error":
                raise DeterminismUnavailableError("framework_determinism_unavailable")
            return ("framework_determinism_unavailable",)
        try:
            if mode == "reproducible":
                deterministic_call = cast(Callable[..., Any], deterministic_api)
                deterministic_call(True, warn_only=unsupported_policy == "warn")
                self._set_cudnn_flags(deterministic=True, benchmark=False)
            else:
                if callable(deterministic_api):
                    deterministic_api(False)
                self._set_cudnn_flags(deterministic=False, benchmark=True)
        except Exception as exc:
            raise FrameworkAdapterError("framework_configure_failed") from exc
        return ()

    def derive_seeds(self, root_seed: int, context: SeedContext) -> tuple[DerivedSeed, ...]:
        _, device_count = self._cuda_runtime()
        cpu_seed = derive_substream_seed(root_seed, context, f"{self.provider}.cpu")
        cuda_seeds = tuple(
            derive_substream_seed(
                root_seed,
                context,
                f"{self.provider}.cuda",
                device=ordinal,
            )
            for ordinal in range(device_count)
        )
        return (cpu_seed, *cuda_seeds)

    def seed(self, root_seed: int, context: SeedContext) -> FrameworkSeedResult:
        manual_seed = getattr(self._torch, "manual_seed", None)
        if not callable(manual_seed):
            raise FrameworkAdapterError("framework_api_unavailable")
        cuda, device_count = self._cuda_runtime()
        cuda_manual_seed: Any | None = None
        cuda_device: Any | None = None
        if device_count:
            cuda_manual_seed = getattr(cuda, "manual_seed", None)
            cuda_device = getattr(cuda, "device", None)
            if not callable(cuda_manual_seed) or not callable(cuda_device):
                raise FrameworkAdapterError("framework_api_unavailable")
        cuda_manual_seed_call = cast(Callable[[int], Any], cuda_manual_seed)
        cuda_device_call = cast(Callable[[int], Any], cuda_device)

        derived_seeds = self.derive_seeds(root_seed, context)
        cpu_seed = derived_seeds[0]
        cuda_seeds = derived_seeds[1:]
        try:
            manual_seed(cpu_seed.seed_u64)
            for ordinal, seed in enumerate(cuda_seeds):
                with cuda_device_call(ordinal):
                    cuda_manual_seed_call(seed.seed_u64)
        except Exception as exc:
            raise FrameworkAdapterError("framework_seed_failed") from exc

        warnings: tuple[RngWarningCode, ...] = ()
        if device_count == 0:
            warnings = ("accelerator_unavailable",)
        return FrameworkSeedResult(
            derived_seeds=derived_seeds,
            warning_codes=warnings,
        )

    def capture(self) -> tuple[FrameworkRngState, ...]:
        get_cpu_state = getattr(self._torch, "get_rng_state", None)
        if not callable(get_cpu_state):
            raise FrameworkAdapterError("framework_api_unavailable")
        cuda, device_count = self._cuda_runtime()
        try:
            states = [
                FrameworkRngState.from_bytes(
                    provider=self.provider,
                    device="cpu",
                    payload=self._tensor_to_bytes(get_cpu_state()),
                )
            ]
            if device_count:
                get_all = getattr(cuda, "get_rng_state_all", None)
                if not callable(get_all):
                    raise FrameworkAdapterError("framework_api_unavailable")
                raw_states = tuple(get_all())
                if len(raw_states) != device_count:
                    raise FrameworkAdapterError("framework_state_invalid")
                for ordinal, state in enumerate(raw_states):
                    states.append(
                        FrameworkRngState.from_bytes(
                            provider=self.provider,
                            device=f"cuda:{ordinal}",
                            payload=self._tensor_to_bytes(state),
                        )
                    )
            return tuple(sorted(states, key=framework_state_sort_key))
        except FrameworkAdapterError:
            raise
        except Exception as exc:
            raise FrameworkAdapterError("framework_capture_failed") from exc

    def validate_restore(self, states: Sequence[FrameworkRngState]) -> None:
        if not states or any(state.provider != self.provider for state in states):
            raise FrameworkAdapterError("framework_state_invalid")
        devices = [state.device for state in states]
        if devices.count("cpu") != 1 or len(set(devices)) != len(devices):
            raise FrameworkAdapterError("framework_state_invalid")

        _, expected_cuda = self._cuda_runtime()
        expected_devices = {"cpu", *(f"cuda:{index}" for index in range(expected_cuda))}
        if set(devices) != expected_devices:
            raise FrameworkAdapterError("framework_device_set_mismatch")
        if tuple(states) != tuple(sorted(states, key=framework_state_sort_key)):
            raise FrameworkAdapterError("framework_state_invalid")
        for state in states:
            state.payload_bytes()

    def restore(self, states: Sequence[FrameworkRngState]) -> None:
        self.validate_restore(states)
        by_device = {state.device: state for state in states}
        set_cpu_state = getattr(self._torch, "set_rng_state", None)
        if not callable(set_cpu_state):
            raise FrameworkAdapterError("framework_api_unavailable")
        cuda_devices = [state.device for state in states if state.device.startswith("cuda:")]
        cuda_tensors: list[Any] = []
        set_all_call: Callable[[list[Any]], Any] | None = None
        if cuda_devices:
            cuda, _ = self._cuda_runtime()
            set_all = getattr(cuda, "set_rng_state_all", None)
            if not callable(set_all):
                raise FrameworkAdapterError("framework_api_unavailable")
            set_all_call = cast(Callable[[list[Any]], Any], set_all)
            cuda_tensors = [
                self._bytes_to_tensor(by_device[device].payload_bytes()) for device in cuda_devices
            ]
        cpu_tensor = self._bytes_to_tensor(by_device["cpu"].payload_bytes())
        try:
            set_cpu_state(cpu_tensor)
            if set_all_call is not None:
                set_all_call(cuda_tensors)
        except Exception as exc:
            raise FrameworkAdapterError("framework_restore_failed") from exc

    def _cuda_runtime(self) -> tuple[Any | None, int]:
        cuda = getattr(self._torch, "cuda", None)
        if cuda is None:
            return None, 0
        is_available = getattr(cuda, "is_available", None)
        try:
            if not callable(is_available) or not bool(is_available()):
                return cuda, 0
            device_count_api = getattr(cuda, "device_count", None)
            if not callable(device_count_api):
                raise FrameworkAdapterError("framework_api_unavailable")
            device_count = int(device_count_api())
            if device_count < 0:
                raise FrameworkAdapterError("framework_state_invalid")
            return cuda, device_count
        except FrameworkAdapterError:
            raise
        except Exception as exc:
            raise FrameworkAdapterError("framework_probe_failed") from exc

    def _cudnn_backend(self) -> Any | None:
        backends = getattr(self._torch, "backends", None)
        return getattr(backends, "cudnn", None) if backends is not None else None

    def _set_cudnn_flags(self, *, deterministic: bool, benchmark: bool) -> None:
        cudnn = self._cudnn_backend()
        if cudnn is not None:
            if hasattr(cudnn, "deterministic"):
                cudnn.deterministic = deterministic
            if hasattr(cudnn, "benchmark"):
                cudnn.benchmark = benchmark

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
        try:
            return tensor(list(payload), dtype=uint8)
        except Exception as exc:
            raise FrameworkAdapterError("framework_state_invalid") from exc
