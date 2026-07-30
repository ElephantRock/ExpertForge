"""Explicit RNG initialization, capture, and checkpoint-style restoration."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from expertforge.config.models import ConfigRoot
from expertforge.rng.adapters import FrameworkAdapterError, FrameworkRngAdapter
from expertforge.rng.derivation import DerivedSeed, SeedContext, derive_seed
from expertforge.rng.state import (
    DeterminismMode,
    FrameworkRngState,
    NumpyGeneratorState,
    NumpyLegacyState,
    PythonRandomState,
    RNG_STATE_SCHEMA_VERSION,
    RngStateBundle,
    RngWarningCode,
    UnsupportedDeterminismPolicy,
)

__all__ = ["RngInitialization", "RngManager", "RngManagerError"]

RngManagerErrorCode = Literal[
    "already_initialized",
    "not_initialized",
    "duplicate_framework_provider",
    "state_contract_mismatch",
    "framework_provider_set_mismatch",
    "numpy_state_invalid",
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
        adapter_seeds = tuple(
            (adapter, self._derive_for(adapter.provider)) for adapter in self._adapters
        )

        warnings: set[RngWarningCode] = set()
        if self.determinism_mode == "performance":
            warnings.add("performance_mode_enabled")
        for adapter, _ in adapter_seeds:
            warnings.update(
                adapter.configure(self.determinism_mode, self.unsupported_determinism)
            )

        random.seed(python_seed.seed_u64, version=2)
        np.random.seed(numpy_legacy_seed.seed_u32)
        self._generator = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(self._seed_words(numpy_generator_seed)))
        )

        for adapter, seed in adapter_seeds:
            warnings.update(adapter.seed(seed))

        derived = tuple(
            sorted(
                (python_seed, numpy_legacy_seed, numpy_generator_seed, *(seed for _, seed in adapter_seeds)),
                key=lambda item: item.context.component,
            )
        )
        self._initialization = RngInitialization(
            derived_seeds=derived,
            warning_codes=tuple(sorted(warnings)),
            active_frameworks=tuple(adapter.provider for adapter in self._adapters),
        )
        return self._initialization

    def capture_state(self) -> RngStateBundle:
        """Capture the next-sample position of every active RNG."""

        initialization = self.initialization
        python_state = random.getstate()
        numpy_legacy = np.random.get_state()
        generator_state = self.generator.bit_generator.state

        framework_states = tuple(
            sorted(
                (state for adapter in self._adapters for state in adapter.capture()),
                key=lambda item: (item.provider, item.device),
            )
        )
        return RngStateBundle(
            rng_state_schema_version=RNG_STATE_SCHEMA_VERSION,
            root_seed=self.root_seed,
            context=self.context,
            determinism_mode=self.determinism_mode,
            unsupported_determinism=self.unsupported_determinism,
            python=self._python_state_from_runtime(python_state),
            numpy_legacy=self._numpy_legacy_from_runtime(numpy_legacy),
            numpy_generator=self._numpy_generator_from_runtime(generator_state),
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

        configure_warnings: set[RngWarningCode] = set()
        for adapter in self._adapters:
            configure_warnings.update(
                adapter.configure(self.determinism_mode, self.unsupported_determinism)
            )
        if not configure_warnings.issubset(set(bundle.warning_codes)):
            raise RngManagerError("state_contract_mismatch")

        random.setstate(
            (
                bundle.python.version,
                tuple(bundle.python.internal_state),
                bundle.python.gauss_next,
            )
        )
        np.random.set_state(
            (
                bundle.numpy_legacy.algorithm,
                np.asarray(bundle.numpy_legacy.keys, dtype=np.uint32),
                bundle.numpy_legacy.position,
                bundle.numpy_legacy.has_gauss,
                bundle.numpy_legacy.cached_gaussian,
            )
        )
        generator = np.random.Generator(np.random.PCG64())
        generator.bit_generator.state = {
            "bit_generator": bundle.numpy_generator.bit_generator,
            "state": {
                "state": bundle.numpy_generator.state,
                "inc": bundle.numpy_generator.increment,
            },
            "has_uint32": bundle.numpy_generator.has_uint32,
            "uinteger": bundle.numpy_generator.uinteger,
        }
        self._generator = generator

        for adapter in self._adapters:
            adapter.restore(states_by_provider[adapter.provider])

        derived = tuple(
            sorted(
                (
                    self._derive_for("python"),
                    self._derive_for("numpy-legacy"),
                    self._derive_for("numpy-generator"),
                    *(self._derive_for(adapter.provider) for adapter in self._adapters),
                ),
                key=lambda item: item.context.component,
            )
        )
        self._initialization = RngInitialization(
            derived_seeds=derived,
            warning_codes=bundle.warning_codes,
            active_frameworks=tuple(adapter.provider for adapter in self._adapters),
        )
        return self._initialization

    def _derive_for(self, subsystem: str) -> DerivedSeed:
        return derive_seed(
            self.root_seed,
            self.context.child(component=f"{self.context.component}.{subsystem}"),
        )

    @staticmethod
    def _seed_words(seed: DerivedSeed) -> tuple[int, ...]:
        digest = bytes.fromhex(seed.digest)
        return tuple(
            int.from_bytes(digest[offset : offset + 4], "big", signed=False)
            for offset in range(0, len(digest), 4)
        )

    @staticmethod
    def _python_state_from_runtime(state: object) -> PythonRandomState:
        if not isinstance(state, tuple) or len(state) != 3:
            raise RngManagerError("numpy_state_invalid")
        version, internal, gauss_next = state
        if not isinstance(version, int) or not isinstance(internal, tuple):
            raise RngManagerError("numpy_state_invalid")
        return PythonRandomState(
            version=version,
            internal_state=tuple(int(value) for value in internal),
            gauss_next=None if gauss_next is None else float(gauss_next),
        )

    @staticmethod
    def _numpy_legacy_from_runtime(state: tuple[object, ...]) -> NumpyLegacyState:
        if len(state) != 5:
            raise RngManagerError("numpy_state_invalid")
        algorithm, keys, position, has_gauss, cached_gaussian = state
        if str(algorithm) != "MT19937":
            raise RngManagerError("numpy_state_invalid")
        try:
            key_tuple = tuple(int(value) for value in np.asarray(keys, dtype=np.uint32).tolist())
        except (TypeError, ValueError, OverflowError) as exc:
            raise RngManagerError("numpy_state_invalid") from exc
        return NumpyLegacyState(
            algorithm="MT19937",
            keys=key_tuple,
            position=int(position),
            has_gauss=int(has_gauss),
            cached_gaussian=float(cached_gaussian),
        )

    @staticmethod
    def _numpy_generator_from_runtime(state: dict[str, object]) -> NumpyGeneratorState:
        try:
            nested = state["state"]
            if not isinstance(nested, dict):
                raise TypeError
            return NumpyGeneratorState(
                bit_generator=str(state["bit_generator"]),
                state=int(nested["state"]),
                increment=int(nested["inc"]),
                has_uint32=int(state["has_uint32"]),
                uinteger=int(state["uinteger"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RngManagerError("numpy_state_invalid") from exc
