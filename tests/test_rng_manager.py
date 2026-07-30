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
    FrameworkSeedResult,
)
from expertforge.rng.derivation import DerivedSeed, SeedContext, derive_substream_seed
from expertforge.rng.manager import RngManager, RngManagerError
from expertforge.rng.state import (
    DeterminismMode,
    FrameworkRngState,
    RngWarningCode,
    UnsupportedDeterminismPolicy,
)


class FakeFrameworkAdapter:
    provider = "fake"

    def __init__(
        self,
        *,
        deterministic_supported: bool = True,
        accelerator: bool = False,
        fail_seed: bool = False,
        fail_next_restore: bool = False,
    ) -> None:
        self.deterministic_supported = deterministic_supported
        self.accelerator = accelerator
        self.fail_seed = fail_seed
        self.fail_next_restore = fail_next_restore
        self.mode: DeterminismMode | None = None
        self._state = 0

    @property
    def state(self) -> int:
        return self._state

    def capture_configuration(self) -> object:
        return self.mode

    def restore_configuration(self, snapshot: object) -> None:
        if snapshot is not None and snapshot not in {"reproducible", "performance"}:
            raise FrameworkAdapterError("framework_state_invalid")
        self.mode = snapshot

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

    def derive_seeds(self, root_seed: int, context: SeedContext) -> tuple[DerivedSeed, ...]:
        return (derive_substream_seed(root_seed, context, "fake.cpu"),)

    def seed(self, root_seed: int, context: SeedContext) -> FrameworkSeedResult:
        derived_seeds = self.derive_seeds(root_seed, context)
        self._state = derived_seeds[0].seed_u64
        if self.fail_seed:
            raise FrameworkAdapterError("framework_seed_failed")
        warnings: tuple[RngWarningCode, ...] = ()
        if not self.accelerator:
            warnings = ("accelerator_unavailable",)
        return FrameworkSeedResult(derived_seeds=derived_seeds, warning_codes=warnings)

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
        if self.fail_next_restore:
            self.fail_next_restore = False
            self._state ^= 1
            raise FrameworkAdapterError("framework_restore_failed")


def _samples(manager: RngManager, count: int = 4) -> tuple[object, ...]:
    return (
        tuple(random.random() for _ in range(count)),
        tuple(float(value) for value in np.random.random(count)),
        tuple(float(value) for value in manager.generator.random(count)),
    )


def _numpy_state() -> tuple[Any, ...]:
    return cast(tuple[Any, ...], np.random.get_state(legacy=True))


def _assert_numpy_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> None:
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


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


def test_long_valid_component_remains_usable() -> None:
    manager = RngManager(root_seed=7, context=SeedContext(component="a" * 128))

    initialization = manager.initialize()

    assert len(initialization.derived_seeds) == 3
    assert all(len(seed.context.component) <= 128 for seed in initialization.derived_seeds)


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
    numpy_before = _numpy_state()
    target = RngManager(root_seed=8, context=SeedContext(component="run"))

    with pytest.raises(RngManagerError, match="state_contract_mismatch"):
        target.restore_state(bundle)

    assert random.getstate() == random_before
    _assert_numpy_state_equal(_numpy_state(), numpy_before)


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


def test_failed_adapter_seed_rolls_back_all_process_and_provider_state() -> None:
    random.seed(101)
    np.random.seed(202)
    random_before = random.getstate()
    numpy_before = _numpy_state()
    adapter = FakeFrameworkAdapter(fail_seed=True)
    adapter_before = adapter.state
    mode_before = adapter.mode
    manager = RngManager(
        root_seed=7,
        context=SeedContext(component="run"),
        adapters=(adapter,),
    )

    with pytest.raises(FrameworkAdapterError, match="framework_seed_failed"):
        manager.initialize()

    assert random.getstate() == random_before
    _assert_numpy_state_equal(_numpy_state(), numpy_before)
    assert adapter.state == adapter_before
    assert adapter.mode == mode_before
    with pytest.raises(RngManagerError, match="not_initialized"):
        _ = manager.generator


def test_failed_adapter_restore_rolls_back_all_process_and_provider_state() -> None:
    source_adapter = FakeFrameworkAdapter()
    source = RngManager(
        root_seed=7,
        context=SeedContext(component="run"),
        adapters=(source_adapter,),
    )
    source.initialize()
    source_adapter.sample()
    bundle = source.capture_state()

    random.seed(303)
    np.random.seed(404)
    random_before = random.getstate()
    numpy_before = _numpy_state()
    target_adapter = FakeFrameworkAdapter(fail_next_restore=True)
    target_adapter._state = 12345
    adapter_before = target_adapter.state
    mode_before = target_adapter.mode
    target = RngManager(
        root_seed=7,
        context=SeedContext(component="run"),
        adapters=(target_adapter,),
    )

    with pytest.raises(FrameworkAdapterError, match="framework_restore_failed"):
        target.restore_state(bundle)

    assert random.getstate() == random_before
    _assert_numpy_state_equal(_numpy_state(), numpy_before)
    assert target_adapter.state == adapter_before
    assert target_adapter.mode == mode_before
    with pytest.raises(RngManagerError, match="not_initialized"):
        _ = target.generator


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
    numpy_before = _numpy_state()

    with pytest.raises(DeterminismUnavailableError, match="framework_determinism_unavailable"):
        manager.initialize()

    assert random.getstate() == random_before
    _assert_numpy_state_equal(_numpy_state(), numpy_before)
    assert adapter.mode is None


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
