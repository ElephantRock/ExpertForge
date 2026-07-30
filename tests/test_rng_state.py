"""Regression tests for the checkpoint-facing RNG state schema."""

from __future__ import annotations

import json
import random

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
    return PythonRandomState(
        version=version,
        internal_state=tuple(int(value) for value in internal),
        gauss_next=gauss_next,
    )


def _legacy_state() -> NumpyLegacyState:
    state = np.random.RandomState(7).get_state()
    return NumpyLegacyState(
        algorithm="MT19937",
        keys=tuple(int(value) for value in state[1].tolist()),
        position=int(state[2]),
        has_gauss=int(state[3]),
        cached_gaussian=float(state[4]),
    )


def _generator_state() -> NumpyGeneratorState:
    state = np.random.Generator(np.random.PCG64(7)).bit_generator.state
    nested = state["state"]
    return NumpyGeneratorState(
        bit_generator="PCG64",
        state=int(nested["state"]),
        increment=int(nested["inc"]),
        has_uint32=int(state["has_uint32"]),
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


def test_framework_state_rejects_noncanonical_base64() -> None:
    state = FrameworkRngState.from_bytes(provider="torch", device="cpu", payload=b"x")
    data = state.model_dump(mode="json")
    data["payload"] = "eA"

    with pytest.raises(ValidationError, match="base64"):
        FrameworkRngState.model_validate(data, strict=False)


@pytest.mark.parametrize("device", ["", "cuda", "cuda:-1", "CPU", "../cpu"])
def test_framework_state_rejects_noncanonical_device(device: str) -> None:
    with pytest.raises(ValidationError, match="device"):
        FrameworkRngState.from_bytes(provider="torch", device=device, payload=b"x")


def test_state_bundle_deterministic_json_round_trip() -> None:
    states = (
        FrameworkRngState.from_bytes(provider="torch", device="cpu", payload=b"cpu"),
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


def test_state_bundle_rejects_duplicate_framework_device() -> None:
    state = FrameworkRngState.from_bytes(provider="torch", device="cpu", payload=b"cpu")

    with pytest.raises(ValidationError, match="duplicate"):
        _bundle(framework_states=(state, state))


def test_state_bundle_rejects_noncanonical_framework_order() -> None:
    states = (
        FrameworkRngState.from_bytes(provider="torch", device="cuda:0", payload=b"cuda"),
        FrameworkRngState.from_bytes(provider="torch", device="cpu", payload=b"cpu"),
    )

    with pytest.raises(ValidationError, match="sorted"):
        _bundle(framework_states=states)


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


def test_warning_codes_must_be_sorted_unique() -> None:
    with pytest.raises(ValidationError, match="sorted"):
        _bundle(warning_codes=("framework_determinism_unavailable", "accelerator_unavailable"))
    with pytest.raises(ValidationError, match="unique"):
        _bundle(warning_codes=("accelerator_unavailable", "accelerator_unavailable"))


def test_python_state_rejects_wrong_shape() -> None:
    with pytest.raises(ValidationError, match="625"):
        PythonRandomState(internal_state=(1, 2, 3))


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


def test_state_json_rejects_non_object_and_invalid_utf8() -> None:
    with pytest.raises(ValueError, match="root must be an object"):
        RngStateBundle.from_json_bytes(b"[]")
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        RngStateBundle.from_json_bytes(b"\xff")


def test_context_round_trip_remains_typed() -> None:
    bundle = _bundle()

    assert bundle.context == SeedContext(component="run")
