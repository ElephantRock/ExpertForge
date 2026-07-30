"""Regression tests for the checkpoint-facing RNG state schema."""

from __future__ import annotations

import json
import random
from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from expertforge.rng.derivation import SeedContext
from expertforge.rng.state import (
    FrameworkRngState,
    NumpyGeneratorState,
    NumpyLegacyState,
    PythonRandomState,
    RngStateBundle,
)


def _python_state() -> PythonRandomState:
    version, internal, gauss_next = random.Random(7).getstate()
    assert version == 3
    return PythonRandomState(
        version=3,
        internal_state=tuple(int(value) for value in internal),
        gauss_next=gauss_next,
    )


def _legacy_state() -> NumpyLegacyState:
    state = cast(tuple[Any, ...], np.random.RandomState(7).get_state(legacy=True))
    assert state[0] == "MT19937"
    assert state[3] in {0, 1}
    return NumpyLegacyState(
        algorithm="MT19937",
        keys=tuple(int(value) for value in np.asarray(state[1], dtype=np.uint32).tolist()),
        position=int(state[2]),
        has_gauss=state[3],
        cached_gaussian=float(state[4]),
    )


def _generator_state() -> NumpyGeneratorState:
    state = cast(dict[str, Any], np.random.Generator(np.random.PCG64(7)).bit_generator.state)
    nested = cast(dict[str, Any], state["state"])
    assert state["bit_generator"] == "PCG64"
    assert state["has_uint32"] in {0, 1}
    return NumpyGeneratorState(
        bit_generator="PCG64",
        state=int(nested["state"]),
        increment=int(nested["inc"]),
        has_uint32=state["has_uint32"],
        uinteger=int(state["uinteger"]),
    )


def _bundle(
    *,
    framework_states: tuple[FrameworkRngState, ...] = (),
    warning_codes: tuple[str, ...] = (),
    determinism_mode: str = "reproducible",
) -> RngStateBundle:
    return RngStateBundle.model_validate(
        {
            "root_seed": 7,
            "context": {"component": "run"},
            "determinism_mode": determinism_mode,
            "unsupported_determinism": "error",
            "python": _python_state().model_dump(mode="json"),
            "numpy_legacy": _legacy_state().model_dump(mode="json"),
            "numpy_generator": _generator_state().model_dump(mode="json"),
            "framework_states": [state.model_dump(mode="json") for state in framework_states],
            "warning_codes": list(warning_codes),
        },
        strict=False,
    )


def _cpu_framework_state() -> FrameworkRngState:
    return FrameworkRngState.from_bytes(provider="torch", device="cpu", payload=b"cpu")


def test_framework_state_round_trip_and_digest() -> None:
    state = FrameworkRngState.from_bytes(provider="torch", device="cpu", payload=b"\x00\x01")

    assert state.payload_bytes() == b"\x00\x01"
    assert FrameworkRngState.model_validate(state.model_dump(mode="json"), strict=False) == state


def test_framework_state_rejects_payload_tampering() -> None:
    state = FrameworkRngState.from_bytes(provider="torch", device="cpu", payload=b"original")
    data = state.model_dump(mode="json")
    data["payload"] = "dGFtcGVyZWQ="

    with pytest.raises(ValidationError, match="digest mismatch"):
        FrameworkRngState.model_validate(data, strict=False)


def test_framework_state_rejects_empty_or_noncanonical_base64() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        FrameworkRngState.from_bytes(provider="torch", device="cpu", payload=b"")

    state = FrameworkRngState.from_bytes(provider="torch", device="cpu", payload=b"x")
    data = state.model_dump(mode="json")
    data["payload"] = "eA"
    with pytest.raises(ValidationError, match="base64"):
        FrameworkRngState.model_validate(data, strict=False)


@pytest.mark.parametrize(
    "device",
    ["", "cuda", "cuda:-1", "cuda:00", "CPU", "../cpu"],
)
def test_framework_state_rejects_noncanonical_device(device: str) -> None:
    with pytest.raises(ValidationError, match="device"):
        FrameworkRngState.from_bytes(provider="torch", device=device, payload=b"x")


def test_state_bundle_deterministic_json_round_trip() -> None:
    states = (
        _cpu_framework_state(),
        FrameworkRngState.from_bytes(provider="torch", device="cuda:0", payload=b"cuda"),
    )
    bundle = _bundle(framework_states=states)

    payload = bundle.to_deterministic_json()
    loaded = RngStateBundle.from_json_bytes(payload)

    assert loaded == bundle
    assert payload == loaded.to_deterministic_json()
    assert (
        payload
        == json.dumps(
            bundle.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def test_state_bundle_is_deeply_immutable() -> None:
    bundle = _bundle()

    with pytest.raises(ValidationError):
        bundle.root_seed = 9  # type: ignore[misc]
    assert isinstance(bundle.python.internal_state, tuple)
    assert isinstance(bundle.numpy_legacy.keys, tuple)
    assert isinstance(bundle.framework_states, tuple)


@pytest.mark.parametrize("field", ["rng_state_schema_version", "derivation_version"])
def test_state_bundle_rejects_unknown_versions(field: str) -> None:
    data = _bundle().model_dump(mode="json")
    data[field] = 2

    with pytest.raises(ValueError, match="Unsupported"):
        RngStateBundle.from_mapping(data)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("root_seed",), "7"),
        (("context", "worker"), "0"),
        (("numpy_generator", "has_uint32"), False),
        (("python", "internal_state", 0), "1"),
    ],
)
def test_state_bundle_rejects_coercible_scalar_types(
    path: tuple[str | int, ...], value: object
) -> None:
    data: Any = _bundle().model_dump(mode="json")
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        RngStateBundle.from_mapping(data)


def test_state_bundle_rejects_duplicate_framework_device() -> None:
    state = _cpu_framework_state()

    with pytest.raises(ValidationError, match="duplicate"):
        _bundle(framework_states=(state, state))


def test_state_bundle_rejects_noncanonical_framework_order() -> None:
    states = (
        FrameworkRngState.from_bytes(provider="torch", device="cuda:0", payload=b"cuda"),
        _cpu_framework_state(),
    )

    with pytest.raises(ValidationError, match="canonical"):
        _bundle(framework_states=states)


def test_state_bundle_requires_numeric_device_order() -> None:
    states = (
        _cpu_framework_state(),
        FrameworkRngState.from_bytes(provider="torch", device="cuda:10", payload=b"ten"),
        FrameworkRngState.from_bytes(provider="torch", device="cuda:2", payload=b"two"),
    )

    with pytest.raises(ValidationError, match="canonical"):
        _bundle(framework_states=states)

    canonical = (states[0], states[2], states[1])
    assert _bundle(framework_states=canonical).framework_states == canonical


def test_performance_mode_requires_warning() -> None:
    with pytest.raises(ValidationError, match="performance_mode_enabled"):
        _bundle(determinism_mode="performance")

    assert (
        _bundle(
            determinism_mode="performance", warning_codes=("performance_mode_enabled",)
        ).determinism_mode
        == "performance"
    )


def test_reproducible_mode_forbids_performance_warning() -> None:
    with pytest.raises(ValidationError, match="cannot record"):
        _bundle(warning_codes=("performance_mode_enabled",))


def test_warning_codes_must_be_sorted_unique_and_bound_to_framework_state() -> None:
    state = _cpu_framework_state()
    with pytest.raises(ValidationError, match="sorted"):
        _bundle(
            framework_states=(state,),
            warning_codes=("framework_determinism_unavailable", "accelerator_unavailable"),
        )
    with pytest.raises(ValidationError, match="unique"):
        _bundle(
            framework_states=(state,),
            warning_codes=("accelerator_unavailable", "accelerator_unavailable"),
        )
    with pytest.raises(ValidationError, match="require persisted framework states"):
        _bundle(warning_codes=("accelerator_unavailable",))


def test_python_state_rejects_wrong_shape() -> None:
    with pytest.raises(ValidationError, match="625"):
        PythonRandomState(internal_state=(1, 2, 3))


def test_numpy_legacy_state_rejects_noncanonical_gaussian_cache() -> None:
    with pytest.raises(ValidationError, match="must store 0.0"):
        NumpyLegacyState(
            keys=tuple(0 for _ in range(624)),
            position=0,
            has_gauss=0,
            cached_gaussian=1.0,
        )


def test_numpy_generator_rejects_invalid_increment_and_cache() -> None:
    with pytest.raises(ValidationError, match="increment must be odd"):
        NumpyGeneratorState(
            state=0,
            increment=2,
            has_uint32=0,
            uinteger=0,
        )
    with pytest.raises(ValidationError, match="must store 0"):
        NumpyGeneratorState(
            state=0,
            increment=1,
            has_uint32=0,
            uinteger=1,
        )


def test_numpy_states_reject_wrong_algorithms() -> None:
    with pytest.raises(ValidationError):
        NumpyLegacyState(
            algorithm="PCG64",  # type: ignore[arg-type]
            keys=tuple(0 for _ in range(624)),
            position=0,
            has_gauss=0,
            cached_gaussian=0.0,
        )
    with pytest.raises(ValidationError):
        NumpyGeneratorState(
            bit_generator="MT19937",  # type: ignore[arg-type]
            state=0,
            increment=1,
            has_uint32=0,
            uinteger=0,
        )


def test_state_json_rejects_non_object_invalid_utf8_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="root must be an object"):
        RngStateBundle.from_json_bytes(b"[]")
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        RngStateBundle.from_json_bytes(b"\xff")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        RngStateBundle.from_json_bytes(b'{"rng_state_schema_version":1,"x":NaN}')


def test_context_round_trip_remains_typed() -> None:
    bundle = _bundle()

    assert bundle.context == SeedContext(component="run")
