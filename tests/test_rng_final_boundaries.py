"""Final authoritative-boundary regressions for the Issue #8 RNG contract."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from expertforge.rng import RngManager, RngManagerError, SeedContext, TorchRngAdapter
from expertforge.rng.adapters import FrameworkAdapterError
from expertforge.rng.state import FrameworkRngState, RngStateBundle
from tests.test_rng_manager import FakeFrameworkAdapter
from tests.test_rng_state import _bundle, _cpu_framework_state
from tests.test_rng_torch_adapter import FakeTorch


def test_framework_warning_codes_match_policy_and_device_facts() -> None:
    cpu = _cpu_framework_state()
    cuda = FrameworkRngState.from_bytes(
        provider="torch",
        device="cuda:0",
        payload=b"cuda",
    )

    with pytest.raises(ValidationError, match="reproducible/warn"):
        _bundle(
            framework_states=(cpu,),
            warning_codes=("framework_determinism_unavailable",),
        )

    warning_data = _bundle(framework_states=(cpu,)).model_dump(mode="json")
    warning_data["unsupported_determinism"] = "warn"
    warning_data["warning_codes"] = ["framework_determinism_unavailable"]
    warning_bundle = RngStateBundle.model_validate(warning_data, strict=False)
    assert warning_bundle.warning_codes == ("framework_determinism_unavailable",)

    with pytest.raises(ValidationError, match="CPU-only provider"):
        _bundle(
            framework_states=(cpu, cuda),
            warning_codes=("accelerator_unavailable",),
        )

    assert _bundle(
        framework_states=(cpu,),
        warning_codes=("accelerator_unavailable",),
    ).warning_codes == ("accelerator_unavailable",)


def test_state_json_rejects_duplicate_object_keys() -> None:
    payload = _bundle().to_deterministic_json().decode("utf-8")
    duplicate = payload.replace('"root_seed":7', '"root_seed":7,"root_seed":8')

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        RngStateBundle.from_json_bytes(duplicate.encode("utf-8"))


def test_broken_optional_torch_import_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str) -> Any:
        assert name == "torch"
        raise OSError("private loader detail")

    monkeypatch.setattr("expertforge.rng.adapters.importlib.import_module", fail_import)

    with pytest.raises(FrameworkAdapterError, match="framework_import_failed"):
        TorchRngAdapter()


def test_missing_cuda_availability_api_is_not_misclassified() -> None:
    torch = FakeTorch()
    setattr(torch.cuda, "is_available", None)
    adapter = TorchRngAdapter(torch)

    with pytest.raises(FrameworkAdapterError, match="framework_api_unavailable"):
        adapter.capture()


def test_configuration_snapshot_requires_determinism_getters() -> None:
    torch = FakeTorch()
    setattr(torch, "are_deterministic_algorithms_enabled", None)
    adapter = TorchRngAdapter(torch)

    with pytest.raises(FrameworkAdapterError, match="framework_api_unavailable"):
        adapter.capture_configuration()


def test_restore_rejects_changed_determinism_capability() -> None:
    source_adapter = FakeFrameworkAdapter(deterministic_supported=False)
    source = RngManager(
        root_seed=7,
        context=SeedContext(component="run"),
        unsupported_determinism="warn",
        adapters=(source_adapter,),
    )
    source.initialize()
    bundle = source.capture_state()

    target_adapter = FakeFrameworkAdapter(deterministic_supported=True)
    target = RngManager(
        root_seed=7,
        context=SeedContext(component="run"),
        unsupported_determinism="warn",
        adapters=(target_adapter,),
    )

    with pytest.raises(RngManagerError, match="state_contract_mismatch"):
        target.restore_state(bundle)

    assert target_adapter.mode is None
    with pytest.raises(RngManagerError, match="not_initialized"):
        _ = target.generator
