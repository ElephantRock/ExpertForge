"""Regression tests for explicit RNG initialization and restoration."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

from expertforge.config.resolve import resolve_config
from expertforge.rng.adapters import (
    DeterminismUnavailableError,
    FrameworkAdapterError,
)
from expertforge.rng.derivation import DerivedSeed, SeedContext
from expertforge.rng.manager import RngManager, RngManagerError
from expertforge.rng.state import (
    DeterminismMode,
    FrameworkRngState,
    RngWarningCode,
    UnsupportedDeterminismPolicy,
)


class FakeFrameworkAdapter:
    provider = "fake"

    def __init__(self, *, deterministic_supported: bool = True, accelerator: bool = False) -> None:
        self.deterministic_supported = deterministic_supported
        self.accelerator = accelerator
        self.mode: DeterminismMode | None = None
        self._state = 0

    def configure(
        self,
        mode: DeterminismMode,
        unsupported_policy: UnsupportedDeterminismPolicy,
    ) -> tuple[RngWarningCode, ...]:
        self.mode = mode
        if mode == "reproducible" and not self.deterministic_supported:
            if unsupported_policy == "error":
                raise DeterminismUnavailableError("framework_determinism_unavailable")
            return ("framework_determinism_unavailable",)
        return ()

    def seed(self, seed: DerivedSeed) -> tuple[RngWarningCode, ...]:
        self._state = seed.seed_u64
        return () if self.accelerator else ("accelerator_unavailable",)

    def sample(self) -> int:
        self._state = (6364136223846793005 * self._state + 1442695040888963407) % 2**64
        return self._state

    def capture(self) -> tuple[FrameworkRngState, ...]:
        return (
            FrameworkRngState.from_bytes(
                provider=self.provider,
                device="cpu",
                payload=self._state.to_bytes(8, "big"),
            ),
        )

    def validate_restore(self, states: Sequence[FrameworkRngState]) -> None:
        if len(states) != 1 or states[0].provider != self.provider or states[0].device != "cpu":
            raise FrameworkAdapterError("framework_state_invalid")
        if len(states[0].payload_bytes()) != 8:
            raise FrameworkAdapterError("framework_state_invalid")

    def restore(self, states: Sequence[FrameworkRngState]) -> None:
        self.validate_restore(states)
        self._state = int.from_bytes(states[0].payload_bytes(), "big")


def _samples(manager: RngManager, count: int = 4) -> tuple[object, ...]:
    return (
        tuple(random.random() for _ in range(count)),
        tuple(float(value) for value in np.random.random(count)),
        tuple(float(value) for value in manager.generator.random(count)),
    )


def test_same_seed_and_context_reproduce_python_and_numpy_outputs() -> None:
    first = RngManager(root_seed=7, context=SeedContext(component="run"))
    first.initialize()
    first_samples = _samples(first)

    second = RngManager(root_seed=7, context=SeedContext(component="run"))
    second.initialize()

    assert _samples(second) == first_samples


def test_different_root_seed_changes_outputs() -> None:
    first = RngManager(root_seed=7, context=SeedContext(component="run"))
    first.initialize()
    first_samples = _samples(first)

    second = RngManager(root_seed=8, context=SeedContext(component="run"))
    second.initialize()

    assert _samples(second) != first_samples


def test_worker_rank_and_component_separate_streams() -> None:
    contexts = (
        SeedContext(component="data", worker=0, rank=0),
        SeedContext(component="data", worker=1, rank=0),
        SeedContext(component="data", worker=0, rank=1),
        SeedContext(component="model", worker=0, rank=0),
    )
    outputs = []
    for context in contexts:
        manager = RngManager(root_seed=11, context=context)
        manager.initialize()
        outputs.append(_samples(manager, count=1))

    assert len(set(outputs)) == len(contexts)


def test_capture_restore_resumes_exact_next_samples() -> None:
    adapter = FakeFrameworkAdapter()
    manager = RngManager(
        root_seed=19,
        context=SeedContext(component="train", worker=2, rank=1),
        adapters=(adapter,),
    )
    manager.initialize()
    _samples(manager, count=5)
    adapter.sample()
    bundle = manager.capture_state()

    expected = (*_samples(manager, count=6), adapter.sample())

    restored_adapter = FakeFrameworkAdapter()
    restored = RngManager(
        root_seed=19,
        context=SeedContext(component="train", worker=2, rank=1),
        adapters=(restored_adapter,),
    )
    restored.restore_state(bundle)

    assert (*_samples(restored, count=6), restored_adapter.sample()) == expected


def test_state_json_round_trip_restores_exact_next_samples() -> None:
    manager = RngManager(root_seed=23, context=SeedContext(component="sampling"))
    manager.initialize()
    _samples(manager, count=3)
    payload = manager.capture_state().to_deterministic_json()
    expected = _samples(manager, count=3)

    restored = RngManager(root_seed=23, context=SeedContext(component="sampling"))
    from expertforge.rng.state import RngStateBundle

    restored.restore_state(RngStateBundle.from_json_bytes(payload))

    assert _samples(restored, count=3) == expected


def test_restore_rejects_contract_mismatch_before_mutating_globals() -> None:
    source = RngManager(root_seed=7, context=SeedContext(component="run"))
    source.initialize()
    bundle = source.capture_state()

    random_before = random.getstate()
    numpy_before = cast(tuple[Any, ...], np.random.get_state(legacy=True))
    target = RngManager(root_seed=8, context=SeedContext(component="run"))

    with pytest.raises(RngManagerError, match="state_contract_mismatch"):
        target.restore_state(bundle)

    assert random.getstate() == random_before
    numpy_after = cast(tuple[Any, ...], np.random.get_state(legacy=True))
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]


def test_restore_rejects_framework_provider_set_mismatch() -> None:
    adapter = FakeFrameworkAdapter()
    source = RngManager(
        root_seed=7,
        context=SeedContext(component="run"),
        adapters=(adapter,),
    )
    source.initialize()
    bundle = source.capture_state()

    target = RngManager(root_seed=7, context=SeedContext(component="run"))
    with pytest.raises(RngManagerError, match="framework_provider_set_mismatch"):
        target.restore_state(bundle)


def test_manager_requires_explicit_initialization() -> None:
    manager = RngManager(root_seed=7, context=SeedContext(component="run"))

    with pytest.raises(RngManagerError, match="not_initialized"):
        _ = manager.generator
    with pytest.raises(RngManagerError, match="not_initialized"):
        manager.capture_state()


def test_manager_rejects_double_initialization_and_restore() -> None:
    manager = RngManager(root_seed=7, context=SeedContext(component="run"))
    manager.initialize()
    bundle = manager.capture_state()

    with pytest.raises(RngManagerError, match="already_initialized"):
        manager.initialize()
    with pytest.raises(RngManagerError, match="already_initialized"):
        manager.restore_state(bundle)


def test_duplicate_framework_providers_are_rejected() -> None:
    with pytest.raises(RngManagerError, match="duplicate_framework_provider"):
        RngManager(
            root_seed=7,
            context=SeedContext(component="run"),
            adapters=(FakeFrameworkAdapter(), FakeFrameworkAdapter()),
        )


def test_unsupported_determinism_error_policy_fails_before_seeding() -> None:
    adapter = FakeFrameworkAdapter(deterministic_supported=False)
    manager = RngManager(
        root_seed=7,
        context=SeedContext(component="run"),
        unsupported_determinism="error",
        adapters=(adapter,),
    )
    random_before = random.getstate()

    with pytest.raises(DeterminismUnavailableError, match="framework_determinism_unavailable"):
        manager.initialize()

    assert random.getstate() == random_before


def test_unsupported_determinism_warn_policy_is_durable() -> None:
    adapter = FakeFrameworkAdapter(deterministic_supported=False)
    manager = RngManager(
        root_seed=7,
        context=SeedContext(component="run"),
        unsupported_determinism="warn",
        adapters=(adapter,),
    )

    initialization = manager.initialize()
    bundle = manager.capture_state()

    assert "framework_determinism_unavailable" in initialization.warning_codes
    assert bundle.warning_codes == initialization.warning_codes


def test_performance_mode_is_explicitly_marked() -> None:
    adapter = FakeFrameworkAdapter()
    manager = RngManager(
        root_seed=7,
        context=SeedContext(component="run"),
        determinism_mode="performance",
        adapters=(adapter,),
    )

    initialization = manager.initialize()

    assert adapter.mode == "performance"
    assert "performance_mode_enabled" in initialization.warning_codes


def test_manager_from_resolved_config_uses_root_policy() -> None:
    config = resolve_config("configs/smoke.yaml").config

    manager = RngManager.from_config(config, component="data.loader", worker=3, rank=2)

    assert manager.root_seed == config.training.seed
    assert manager.determinism_mode == "reproducible"
    assert manager.unsupported_determinism == "error"
    assert manager.context == SeedContext(component="data.loader", worker=3, rank=2)


def test_root_seed_rejects_bool() -> None:
    with pytest.raises(TypeError, match="root_seed"):
        RngManager(root_seed=True, context=SeedContext(component="run"))
