"""Explicit RNG initialization, capture, and checkpoint-style restoration."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from expertforge.config.models import ConfigRoot
from expertforge.rng.adapters import FrameworkRngAdapter
from expertforge.rng.derivation import DerivedSeed, SeedContext, derive_substream_seed
from expertforge.rng.state import (
    RNG_STATE_SCHEMA_VERSION,
    DeterminismMode,
    FrameworkRngState,
    NumpyGeneratorState,
    NumpyLegacyState,
    PythonRandomState,
    RngStateBundle,
    RngWarningCode,
    UnsupportedDeterminismPolicy,
    framework_state_sort_key,
)

__all__ = ["RngInitialization", "RngManager", "RngManagerError"]

RngManagerErrorCode = Literal[
    "already_initialized",
    "not_initialized",
    "duplicate_framework_provider",
    "invalid_determinism_mode",
    "invalid_unsupported_determinism_policy",
    "state_contract_mismatch",
    "framework_provider_set_mismatch",
    "framework_seed_plan_mismatch",
    "initialization_rollback_failed",
    "restore_rollback_failed",
    "python_state_invalid",
    "numpy_state_invalid",
]
RollbackErrorCode = Literal[
    "initialization_rollback_failed",
    "restore_rollback_failed",
]


class RngManagerError(RuntimeError):
    """Typed RNG-manager failure with a stable non-secret code."""

    def __init__(self, code: RngManagerErrorCode) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class RngInitialization:
    """Immutable evidence returned by explicit RNG initialization."""

    derived_seeds: tuple[DerivedSeed, ...]
    warning_codes: tuple[RngWarningCode, ...]
    active_frameworks: tuple[str, ...]


@dataclass(frozen=True)
class _AdapterSnapshot:
    adapter: FrameworkRngAdapter
    configuration: object
    states: tuple[FrameworkRngState, ...]


class RngManager:
    """Own deterministic seed derivation and active process RNG state.

    Constructing a manager has no side effects. :meth:`initialize` or
    :meth:`restore_state` must be called explicitly before sampling through the
    owned NumPy generator or capturing state.
    """

    def __init__(
        self,
        *,
        root_seed: int,
        context: SeedContext,
        determinism_mode: DeterminismMode = "reproducible",
        unsupported_determinism: UnsupportedDeterminismPolicy = "error",
        adapters: Iterable[FrameworkRngAdapter] = (),
    ) -> None:
        if isinstance(root_seed, bool) or not isinstance(root_seed, int):
            raise TypeError("root_seed must be an integer, not a bool or coercible value")
        if determinism_mode not in {"reproducible", "performance"}:
            raise RngManagerError("invalid_determinism_mode")
        if unsupported_determinism not in {"error", "warn"}:
            raise RngManagerError("invalid_unsupported_determinism_policy")
        self.root_seed = root_seed
        self.context = context
        self.determinism_mode = determinism_mode
        self.unsupported_determinism = unsupported_determinism
        self._adapters = tuple(sorted(adapters, key=lambda adapter: adapter.provider))
        providers = tuple(adapter.provider for adapter in self._adapters)
        if len(set(providers)) != len(providers):
            raise RngManagerError("duplicate_framework_provider")
        self._generator: np.random.Generator | None = None
        self._initialization: RngInitialization | None = None

    @classmethod
    def from_config(
        cls,
        config: ConfigRoot,
        *,
        component: str = "run",
        worker: int = 0,
        rank: int = 0,
        device: int = 0,
        stream: int = 0,
        adapters: Iterable[FrameworkRngAdapter] = (),
    ) -> RngManager:
        """Construct an uninitialized manager from the resolved configuration."""

        return cls(
            root_seed=config.training.seed,
            context=SeedContext(
                component=component,
                worker=worker,
                rank=rank,
                device=device,
                stream=stream,
            ),
            determinism_mode=config.training.determinism_mode,
            unsupported_determinism=config.training.unsupported_determinism,
            adapters=adapters,
        )

    @property
    def generator(self) -> np.random.Generator:
        if self._generator is None:
            raise RngManagerError("not_initialized")
        return self._generator

    @property
    def initialization(self) -> RngInitialization:
        if self._initialization is None:
            raise RngManagerError("not_initialized")
        return self._initialization

    def initialize(self) -> RngInitialization:
        """Explicitly seed Python, NumPy, and supplied framework adapters."""

        if self._initialization is not None:
            raise RngManagerError("already_initialized")

        python_seed = self._derive_for("python")
        numpy_legacy_seed = self._derive_for("numpy-legacy")
        numpy_generator_seed = self._derive_for("numpy-generator")
        adapter_plans = tuple(
            (adapter, adapter.derive_seeds(self.root_seed, self.context))
            for adapter in self._adapters
        )
        staged_generator = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(self._seed_words(numpy_generator_seed)))
        )
        python_before = random.getstate()
        numpy_before = np.random.get_state(legacy=True)
        adapter_before = self._capture_adapter_snapshots()

        warnings: set[RngWarningCode] = set()
        if self.determinism_mode == "performance":
            warnings.add("performance_mode_enabled")
        adapter_results: list[tuple[DerivedSeed, ...]] = []
        try:
            for adapter in self._adapters:
                warnings.update(
                    adapter.configure(self.determinism_mode, self.unsupported_determinism)
                )
            random.seed(python_seed.seed_u64, version=2)
            np.random.seed(numpy_legacy_seed.seed_u32)
            for adapter, expected_seeds in adapter_plans:
                result = adapter.seed(self.root_seed, self.context)
                if result.derived_seeds != expected_seeds:
                    raise RngManagerError("framework_seed_plan_mismatch")
                adapter_results.append(result.derived_seeds)
                warnings.update(result.warning_codes)
        except Exception:
            self._rollback(
                python_before,
                numpy_before,
                adapter_before,
                code="initialization_rollback_failed",
            )
            raise

        derived = self._sorted_derived_seeds(
            python_seed,
            numpy_legacy_seed,
            numpy_generator_seed,
            *(seed for result in adapter_results for seed in result),
        )
        initialization = RngInitialization(
            derived_seeds=derived,
            warning_codes=tuple(sorted(warnings)),
            active_frameworks=tuple(adapter.provider for adapter in self._adapters),
        )
        self._generator = staged_generator
        self._initialization = initialization
        return initialization

    def capture_state(self) -> RngStateBundle:
        """Capture the next-sample position of every active RNG."""

        initialization = self.initialization
        framework_states = tuple(
            sorted(
                (state for adapter in self._adapters for state in adapter.capture()),
                key=framework_state_sort_key,
            )
        )
        return RngStateBundle(
            rng_state_schema_version=RNG_STATE_SCHEMA_VERSION,
            root_seed=self.root_seed,
            context=self.context,
            determinism_mode=self.determinism_mode,
            unsupported_determinism=self.unsupported_determinism,
            python=self._python_state_from_runtime(random.getstate()),
            numpy_legacy=self._numpy_legacy_from_runtime(np.random.get_state(legacy=True)),
            numpy_generator=self._numpy_generator_from_runtime(self.generator.bit_generator.state),
            framework_states=framework_states,
            warning_codes=initialization.warning_codes,
        )

    def restore_state(self, bundle: RngStateBundle) -> RngInitialization:
        """Validate then restore a checkpoint-facing RNG state bundle."""

        if self._initialization is not None:
            raise RngManagerError("already_initialized")
        if (
            bundle.root_seed != self.root_seed
            or bundle.context != self.context
            or bundle.determinism_mode != self.determinism_mode
            or bundle.unsupported_determinism != self.unsupported_determinism
        ):
            raise RngManagerError("state_contract_mismatch")

        states_by_provider: dict[str, list[FrameworkRngState]] = defaultdict(list)
        for state in bundle.framework_states:
            states_by_provider[state.provider].append(state)
        adapter_providers = {adapter.provider for adapter in self._adapters}
        if set(states_by_provider) != adapter_providers:
            raise RngManagerError("framework_provider_set_mismatch")
        for adapter in self._adapters:
            adapter.validate_restore(states_by_provider[adapter.provider])

        staged_generator = self._generator_from_state(bundle.numpy_generator)
        planned_adapter_seeds = tuple(
            seed
            for adapter in self._adapters
            for seed in adapter.derive_seeds(self.root_seed, self.context)
        )
        python_before = random.getstate()
        numpy_before = np.random.get_state(legacy=True)
        adapter_before = self._capture_adapter_snapshots()

        configure_warnings: set[RngWarningCode] = set()
        try:
            for adapter in self._adapters:
                configure_warnings.update(
                    adapter.configure(self.determinism_mode, self.unsupported_determinism)
                )
            runtime_warning = "framework_determinism_unavailable" in configure_warnings
            stored_warning = "framework_determinism_unavailable" in bundle.warning_codes
            if runtime_warning != stored_warning:
                raise RngManagerError("state_contract_mismatch")
            random.setstate(
                (
                    bundle.python.version,
                    tuple(bundle.python.internal_state),
                    bundle.python.gauss_next,
                )
            )
            np.random.set_state(cast(Any, self._numpy_legacy_to_runtime(bundle.numpy_legacy)))
            for adapter in self._adapters:
                adapter.restore(states_by_provider[adapter.provider])
        except Exception:
            self._rollback(
                python_before,
                numpy_before,
                adapter_before,
                code="restore_rollback_failed",
            )
            raise

        initialization = RngInitialization(
            derived_seeds=self._sorted_derived_seeds(
                self._derive_for("python"),
                self._derive_for("numpy-legacy"),
                self._derive_for("numpy-generator"),
                *planned_adapter_seeds,
            ),
            warning_codes=bundle.warning_codes,
            active_frameworks=tuple(adapter.provider for adapter in self._adapters),
        )
        self._generator = staged_generator
        self._initialization = initialization
        return initialization

    def _capture_adapter_snapshots(self) -> tuple[_AdapterSnapshot, ...]:
        return tuple(
            _AdapterSnapshot(
                adapter=adapter,
                configuration=adapter.capture_configuration(),
                states=adapter.capture(),
            )
            for adapter in self._adapters
        )

    def _rollback(
        self,
        python_state: object,
        numpy_state: object,
        adapter_snapshots: tuple[_AdapterSnapshot, ...],
        *,
        code: RollbackErrorCode,
    ) -> None:
        first_error: Exception | None = None
        try:
            random.setstate(cast(tuple[Any, ...], python_state))
        except Exception as exc:
            first_error = exc
        try:
            np.random.set_state(cast(Any, numpy_state))
        except Exception as exc:
            first_error = first_error or exc
        for snapshot in reversed(adapter_snapshots):
            try:
                snapshot.adapter.restore(snapshot.states)
            except Exception as exc:
                first_error = first_error or exc
            try:
                snapshot.adapter.restore_configuration(snapshot.configuration)
            except Exception as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise RngManagerError(code) from first_error

    def _derive_for(self, subsystem: str) -> DerivedSeed:
        return derive_substream_seed(self.root_seed, self.context, subsystem)

    @staticmethod
    def _sorted_derived_seeds(*seeds: DerivedSeed) -> tuple[DerivedSeed, ...]:
        return tuple(
            sorted(
                seeds,
                key=lambda item: (
                    item.context.component,
                    item.context.worker,
                    item.context.rank,
                    item.context.device,
                    item.context.stream,
                ),
            )
        )

    @staticmethod
    def _seed_words(seed: DerivedSeed) -> tuple[int, ...]:
        digest = bytes.fromhex(seed.digest)
        return tuple(
            int.from_bytes(digest[offset : offset + 4], "big", signed=False)
            for offset in range(0, len(digest), 4)
        )

    @staticmethod
    def _generator_from_state(state: NumpyGeneratorState) -> np.random.Generator:
        generator = np.random.Generator(np.random.PCG64())
        generator.bit_generator.state = {
            "bit_generator": state.bit_generator,
            "state": {
                "state": state.state,
                "inc": state.increment,
            },
            "has_uint32": state.has_uint32,
            "uinteger": state.uinteger,
        }
        return generator

    @staticmethod
    def _numpy_legacy_to_runtime(state: NumpyLegacyState) -> tuple[object, ...]:
        return (
            state.algorithm,
            np.asarray(state.keys, dtype=np.uint32),
            state.position,
            state.has_gauss,
            state.cached_gaussian,
        )

    @staticmethod
    def _python_state_from_runtime(state: object) -> PythonRandomState:
        if not isinstance(state, tuple) or len(state) != 3:
            raise RngManagerError("python_state_invalid")
        version, internal, gauss_next = state
        if version != 3 or not isinstance(internal, tuple):
            raise RngManagerError("python_state_invalid")
        try:
            internal_state = tuple(int(value) for value in internal)
            normalized_gauss = None if gauss_next is None else float(gauss_next)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RngManagerError("python_state_invalid") from exc
        return PythonRandomState(
            version=3,
            internal_state=internal_state,
            gauss_next=normalized_gauss,
        )

    @staticmethod
    def _numpy_legacy_from_runtime(state: object) -> NumpyLegacyState:
        if not isinstance(state, tuple) or len(state) != 5:
            raise RngManagerError("numpy_state_invalid")
        algorithm, keys, position, has_gauss, cached_gaussian = state
        if algorithm != "MT19937" or has_gauss not in {0, 1}:
            raise RngManagerError("numpy_state_invalid")
        try:
            key_tuple = tuple(int(value) for value in np.asarray(keys, dtype=np.uint32).tolist())
            normalized_position = int(cast(Any, position))
            normalized_cached = float(cast(Any, cached_gaussian))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RngManagerError("numpy_state_invalid") from exc
        return NumpyLegacyState(
            algorithm="MT19937",
            keys=key_tuple,
            position=normalized_position,
            has_gauss=has_gauss,
            cached_gaussian=normalized_cached,
        )

    @staticmethod
    def _numpy_generator_from_runtime(state: Mapping[str, Any]) -> NumpyGeneratorState:
        try:
            bit_generator = state["bit_generator"]
            nested = state["state"]
            has_uint32 = state["has_uint32"]
            if (
                bit_generator != "PCG64"
                or not isinstance(nested, Mapping)
                or has_uint32 not in {0, 1}
            ):
                raise TypeError
            return NumpyGeneratorState(
                bit_generator="PCG64",
                state=int(nested["state"]),
                increment=int(nested["inc"]),
                has_uint32=has_uint32,
                uinteger=int(state["uinteger"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RngManagerError("numpy_state_invalid") from exc
