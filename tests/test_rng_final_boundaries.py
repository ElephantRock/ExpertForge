"""Final authoritative-boundary regressions for the Issue #8 RNG contract."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from expertforge.rng import (
    FrameworkSeedResult,
    RngManager,
    RngManagerError,
    SeedContext,
    TorchRngAdapter,
    derive_substream_seed,
)
from expertforge.rng.adapters import FrameworkAdapterError
from expertforge.rng.derivation import DerivedSeed
from expertforge.rng.state import (
    DeterminismMode,
    FrameworkRngState,
    RngStateBundle,
    RngWarningCode,
    UnsupportedDeterminismPolicy,
)


class _BoundaryAdapter:
    provider = "boundary"

    def __init__(self, *, deterministic_supported: bool) -> None:
        self.deterministic_supported = deterministic_supported
        self.mode: DeterminismMode | None = None
        self._state = 0

    def capture_configuration(self) -> object:
        return self.mode

    def restore_configuration(self, snapshot: object) -> None:
        self.mode = cast(DeterminismMode | None, snapshot)

    def configure(
        self,
        mode: DeterminismMode,
        unsupported_policy: UnsupportedDeterminismPolicy,
    ) -> tuple[RngWarningCode, ...]:
        self.mode = mode
        if mode == "reproducible" and not self.deterministic_supported:
            if unsupported_policy == "error":
                raise FrameworkAdapterError("framework_configure_failed")
            return ("framework_determinism_unavailable",)
        return ()

    def derive_seeds(self, root_seed: int, context: SeedContext) -> tuple[DerivedSeed, ...]:
        return (derive_substream_seed(root_seed, context, "boundary.cpu"),)

    def seed(self, root_seed: int, context: SeedContext) -> FrameworkSeedResult:
        seeds = self.derive_seeds(root_seed, context)
        self._state = seeds[0].seed_u64
        return FrameworkSeedResult(
            derived_seeds=seeds,
            warning_codes=("accelerator_unavailable",),
        )

    def capture(self) -> tuple[FrameworkRngState, ...]:
        return (
            FrameworkRngState.from_bytes(
                provider=self.provider,
                device="cpu",
                payload=self._state.to_bytes(8, "big"),
            ),
        )

    def validate_restore(self, states: Sequence[FrameworkRngState]) -> None:
        if len(states) != 1 or states[0].provider != self.provider:
            raise FrameworkAdapterError("framework_state_invalid")

    def restore(self, states: Sequence[FrameworkRngState]) -> None:
        self.validate_restore(states)
        self._state = int.from_bytes(states[0].payload_bytes(), "big")


def _base_bundle_data() -> dict[str, Any]:
    manager = RngManager(root_seed=7, context=SeedContext(component="boundary"))
    manager.initialize()
    return manager.capture_state().model_dump(mode="json")


def _bundle(
    *,
    framework_states: tuple[FrameworkRngState, ...] = (),
    warning_codes: tuple[str, ...] = (),
    unsupported_determinism: str = "error",
) -> RngStateBundle:
    data = _base_bundle_data()
    data["framework_states"] = [state.model_dump(mode="json") for state in framework_states]
    data["warning_codes"] = list(warning_codes)
    data["unsupported_determinism"] = unsupported_determinism
    return RngStateBundle.model_validate(data, strict=False)


def test_framework_warning_codes_match_policy_and_device_facts() -> None:
    cpu = FrameworkRngState.from_bytes(
        provider="torch",
        device="cpu",
        payload=b"cpu",
    )
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

    warning_bundle = _bundle(
        framework_states=(cpu,),
        warning_codes=("framework_determinism_unavailable",),
        unsupported_determinism="warn",
    )
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
    torch = SimpleNamespace(
        cuda=SimpleNamespace(),
        get_rng_state=lambda: object(),
    )
    adapter = TorchRngAdapter(torch)

    with pytest.raises(FrameworkAdapterError, match="framework_api_unavailable"):
        adapter.capture()


def test_configuration_snapshot_requires_determinism_getters() -> None:
    torch = SimpleNamespace(
        use_deterministic_algorithms=lambda enabled, warn_only=False: None,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
    )
    adapter = TorchRngAdapter(torch)

    with pytest.raises(FrameworkAdapterError, match="framework_api_unavailable"):
        adapter.capture_configuration()


def test_restore_rejects_changed_determinism_capability() -> None:
    source_adapter = _BoundaryAdapter(deterministic_supported=False)
    source = RngManager(
        root_seed=7,
        context=SeedContext(component="run"),
        unsupported_determinism="warn",
        adapters=(source_adapter,),
    )
    source.initialize()
    bundle = source.capture_state()

    target_adapter = _BoundaryAdapter(deterministic_supported=True)
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
