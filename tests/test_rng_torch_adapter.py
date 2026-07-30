"""Dependency-free contract tests for the lazy PyTorch RNG adapter."""

from __future__ import annotations

from typing import Any

import pytest

from expertforge.rng.adapters import (
    DeterminismUnavailableError,
    FrameworkAdapterError,
    TorchRngAdapter,
)
from expertforge.rng.derivation import SeedContext, derive_seed


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


class FakeCuda:
    def __init__(self, *, available: bool, device_count: int = 0) -> None:
        self.available = available
        self._device_count = device_count
        self.states = [FakeTensor([index + 10]) for index in range(device_count)]
        self.seed_value: int | None = None

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self._device_count

    def manual_seed_all(self, seed: int) -> None:
        self.seed_value = seed
        self.states = [FakeTensor([(seed + index) % 256]) for index in range(self._device_count)]

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
        self.deterministic_calls: list[tuple[bool, bool | None]] = []
        if not deterministic_api:
            self.use_deterministic_algorithms = None  # type: ignore[assignment]

    def use_deterministic_algorithms(self, enabled: bool, warn_only: bool = False) -> None:
        self.deterministic_calls.append((enabled, warn_only))

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


def test_missing_deterministic_api_obeys_error_and_warn_policy() -> None:
    torch = FakeTorch(deterministic_api=False)
    adapter = TorchRngAdapter(torch)

    with pytest.raises(DeterminismUnavailableError, match="framework_determinism_unavailable"):
        adapter.configure("reproducible", "error")

    assert adapter.configure("reproducible", "warn") == (
        "framework_determinism_unavailable",
    )


def test_seed_covers_cpu_and_all_available_cuda_devices() -> None:
    torch = FakeTorch(cuda_available=True, cuda_devices=2)
    adapter = TorchRngAdapter(torch)
    seed = derive_seed(7, SeedContext(component="run.torch"))

    warnings = adapter.seed(seed)

    assert warnings == ()
    assert torch.seed_value == seed.seed_u64
    assert torch.cuda.seed_value == seed.seed_u64


def test_cpu_only_seed_records_accelerator_unavailable() -> None:
    torch = FakeTorch(cuda_available=False, cuda_devices=0)
    adapter = TorchRngAdapter(torch)

    assert adapter.seed(derive_seed(7, SeedContext(component="run.torch"))) == (
        "accelerator_unavailable",
    )


def test_capture_and_restore_round_trip_cpu_and_cuda_states() -> None:
    torch = FakeTorch(cuda_available=True, cuda_devices=2)
    adapter = TorchRngAdapter(torch)
    adapter.seed(derive_seed(7, SeedContext(component="run.torch")))
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


def test_capture_orders_device_states_canonically() -> None:
    adapter = TorchRngAdapter(FakeTorch(cuda_available=True, cuda_devices=12))

    devices = tuple(state.device for state in adapter.capture())

    assert devices == tuple(sorted(devices))


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


@pytest.mark.accelerator
def test_real_torch_cuda_state_round_trip_when_available() -> None:
    torch = pytest.importorskip("torch", reason="optional PyTorch is not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime/device unavailable")

    adapter = TorchRngAdapter(torch)
    seed = derive_seed(7, SeedContext(component="run.torch"))
    adapter.configure("reproducible", "warn")
    adapter.seed(seed)
    state = adapter.capture()
    adapter.restore(state)

    assert any(item.device.startswith("cuda:") for item in state)
