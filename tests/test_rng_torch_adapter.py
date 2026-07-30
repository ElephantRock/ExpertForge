"""Dependency-free contract tests for the lazy PyTorch RNG adapter."""

from __future__ import annotations

from types import TracebackType
from typing import Any

import pytest

from expertforge.rng.adapters import (
    DeterminismUnavailableError,
    FrameworkAdapterError,
    TorchRngAdapter,
)
from expertforge.rng.derivation import SeedContext


class FakeTensor:
    def __init__(self, values: list[int]) -> None:
        self.values = list(values)

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list[int]:
        return list(self.values)


class FakeCudnn:
    deterministic = False
    benchmark = False


class FakeBackends:
    def __init__(self) -> None:
        self.cudnn = FakeCudnn()


class FakeDeviceContext:
    def __init__(self, cuda: FakeCuda, ordinal: int) -> None:
        self.cuda = cuda
        self.ordinal = ordinal
        self.previous = cuda.current_device

    def __enter__(self) -> None:
        self.cuda.current_device = self.ordinal

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.cuda.current_device = self.previous


class FakeCuda:
    def __init__(self, *, available: bool, device_count: int = 0) -> None:
        self.available = available
        self._device_count = device_count
        self.states = [FakeTensor([index + 10]) for index in range(device_count)]
        self.seed_values: dict[int, int] = {}
        self.current_device = 0

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self._device_count

    def device(self, ordinal: int) -> FakeDeviceContext:
        if ordinal < 0 or ordinal >= self._device_count:
            raise ValueError("invalid device")
        return FakeDeviceContext(self, ordinal)

    def manual_seed(self, seed: int) -> None:
        self.seed_values[self.current_device] = seed
        self.states[self.current_device] = FakeTensor(
            [seed % 256, (seed >> 8) % 256]
        )

    def get_rng_state_all(self) -> list[FakeTensor]:
        return list(self.states)

    def set_rng_state_all(self, states: list[FakeTensor]) -> None:
        self.states = list(states)


class FakeTorch:
    uint8 = object()

    def __init__(
        self,
        *,
        cuda_available: bool = True,
        cuda_devices: int = 2,
        deterministic_api: bool = True,
    ) -> None:
        self.backends = FakeBackends()
        self.cuda = FakeCuda(available=cuda_available, device_count=cuda_devices)
        self.cpu_state = FakeTensor([1, 2, 3])
        self.seed_value: int | None = None
        self.deterministic_enabled = False
        self.deterministic_warn_only = False
        self.deterministic_calls: list[tuple[bool, bool | None]] = []
        if not deterministic_api:
            self.use_deterministic_algorithms = None  # type: ignore[assignment]

    def use_deterministic_algorithms(self, enabled: bool, warn_only: bool = False) -> None:
        self.deterministic_enabled = enabled
        self.deterministic_warn_only = warn_only
        self.deterministic_calls.append((enabled, warn_only))

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.deterministic_enabled

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self.deterministic_warn_only

    def manual_seed(self, seed: int) -> None:
        self.seed_value = seed
        self.cpu_state = FakeTensor([seed % 256, (seed >> 8) % 256])

    def get_rng_state(self) -> FakeTensor:
        return self.cpu_state

    def set_rng_state(self, state: FakeTensor) -> None:
        self.cpu_state = state

    def tensor(self, values: list[int], *, dtype: object) -> FakeTensor:
        assert dtype is self.uint8
        return FakeTensor(values)


def test_reproducible_mode_configures_deterministic_backends() -> None:
    torch = FakeTorch()
    adapter = TorchRngAdapter(torch)

    warnings = adapter.configure("reproducible", "error")

    assert warnings == ()
    assert torch.deterministic_calls == [(True, False)]
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_performance_mode_configures_performance_backends() -> None:
    torch = FakeTorch()
    adapter = TorchRngAdapter(torch)

    adapter.configure("performance", "error")

    assert torch.deterministic_calls == [(False, False)]
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_configuration_snapshot_round_trip() -> None:
    torch = FakeTorch()
    adapter = TorchRngAdapter(torch)
    snapshot = adapter.capture_configuration()
    adapter.configure("reproducible", "warn")

    adapter.restore_configuration(snapshot)

    assert torch.deterministic_enabled is False
    assert torch.deterministic_warn_only is False
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is False


def test_missing_deterministic_api_obeys_error_and_warn_policy() -> None:
    torch = FakeTorch(deterministic_api=False)
    adapter = TorchRngAdapter(torch)

    with pytest.raises(DeterminismUnavailableError, match="framework_determinism_unavailable"):
        adapter.configure("reproducible", "error")

    assert adapter.configure("reproducible", "warn") == (
        "framework_determinism_unavailable",
    )


def test_seed_uses_distinct_cpu_and_cuda_streams() -> None:
    torch = FakeTorch(cuda_available=True, cuda_devices=2)
    adapter = TorchRngAdapter(torch)
    context = SeedContext(component="run")

    result = adapter.seed(7, context)

    assert result.warning_codes == ()
    assert len(result.derived_seeds) == 3
    seed_values = tuple(seed.seed_u64 for seed in result.derived_seeds)
    assert len(set(seed_values)) == 3
    assert torch.seed_value == seed_values[0]
    assert torch.cuda.seed_values == {0: seed_values[1], 1: seed_values[2]}
    assert result.derived_seeds[1].context.device == 0
    assert result.derived_seeds[2].context.device == 1


def test_cpu_only_seed_records_accelerator_unavailable() -> None:
    torch = FakeTorch(cuda_available=False, cuda_devices=0)
    adapter = TorchRngAdapter(torch)

    result = adapter.seed(7, SeedContext(component="run"))

    assert result.warning_codes == ("accelerator_unavailable",)
    assert len(result.derived_seeds) == 1


def test_capture_and_restore_round_trip_cpu_and_cuda_states() -> None:
    torch = FakeTorch(cuda_available=True, cuda_devices=2)
    adapter = TorchRngAdapter(torch)
    adapter.seed(7, SeedContext(component="run"))
    captured = adapter.capture()
    expected = tuple((state.device, state.payload_bytes()) for state in captured)

    torch.cpu_state = FakeTensor([99])
    torch.cuda.states = [FakeTensor([98]), FakeTensor([97])]
    adapter.restore(captured)

    restored = adapter.capture()
    assert tuple((state.device, state.payload_bytes()) for state in restored) == expected


def test_restore_rejects_device_set_mismatch_before_mutation() -> None:
    source = TorchRngAdapter(FakeTorch(cuda_available=True, cuda_devices=2))
    states = source.capture()
    target_torch = FakeTorch(cuda_available=True, cuda_devices=1)
    target = TorchRngAdapter(target_torch)
    before = target_torch.cpu_state.tolist()

    with pytest.raises(FrameworkAdapterError, match="framework_device_set_mismatch"):
        target.restore(states)

    assert target_torch.cpu_state.tolist() == before


def test_capture_orders_device_states_by_numeric_ordinal() -> None:
    adapter = TorchRngAdapter(FakeTorch(cuda_available=True, cuda_devices=12))

    devices = tuple(state.device for state in adapter.capture())

    assert devices == ("cpu", *(f"cuda:{ordinal}" for ordinal in range(12)))


def test_capture_rejects_wrong_cuda_state_count() -> None:
    torch = FakeTorch(cuda_available=True, cuda_devices=2)
    torch.cuda.states.pop()
    adapter = TorchRngAdapter(torch)

    with pytest.raises(FrameworkAdapterError, match="framework_state_invalid"):
        adapter.capture()


def test_adapter_rejects_missing_cpu_state() -> None:
    adapter = TorchRngAdapter(FakeTorch(cuda_available=False, cuda_devices=0))

    with pytest.raises(FrameworkAdapterError, match="framework_state_invalid"):
        adapter.validate_restore(())


def test_tensor_state_must_be_byte_convertible() -> None:
    class InvalidTorch(FakeTorch):
        def get_rng_state(self) -> Any:
            return object()

    adapter = TorchRngAdapter(InvalidTorch(cuda_available=False, cuda_devices=0))

    with pytest.raises(FrameworkAdapterError, match="framework_state_invalid"):
        adapter.capture()


def test_provider_probe_failure_is_typed() -> None:
    class FailingCuda(FakeCuda):
        def is_available(self) -> bool:
            raise OSError("private provider detail")

    torch = FakeTorch()
    torch.cuda = FailingCuda(available=True, device_count=1)
    adapter = TorchRngAdapter(torch)

    with pytest.raises(FrameworkAdapterError, match="framework_probe_failed"):
        adapter.capture()


@pytest.mark.accelerator
def test_real_torch_cuda_state_round_trip_when_available() -> None:
    torch = pytest.importorskip("torch", reason="optional PyTorch is not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime/device unavailable")

    adapter = TorchRngAdapter(torch)
    adapter.configure("reproducible", "warn")
    adapter.seed(7, SeedContext(component="run"))
    state = adapter.capture()
    adapter.restore(state)

    assert any(item.device.startswith("cuda:") for item in state)
