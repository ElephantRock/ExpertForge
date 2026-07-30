"""Versioned, immutable, non-pickle RNG state for checkpoint embedding."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from typing import Any, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from expertforge.rng.derivation import SEED_DERIVATION_VERSION, SeedContext

__all__ = [
    "RNG_STATE_SCHEMA_VERSION",
    "DeterminismMode",
    "FrameworkRngState",
    "NumpyGeneratorState",
    "NumpyLegacyState",
    "PythonRandomState",
    "RngStateBundle",
    "RngWarningCode",
    "UnsupportedDeterminismPolicy",
    "framework_state_sort_key",
]

RNG_STATE_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PROVIDER_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_DEVICE_PATTERN = re.compile(r"^(?:cpu|[a-z][a-z0-9_-]*:[0-9]+)$")

DeterminismMode = Literal["reproducible", "performance"]
UnsupportedDeterminismPolicy = Literal["error", "warn"]
RngWarningCode = Literal[
    "performance_mode_enabled",
    "framework_determinism_unavailable",
    "accelerator_unavailable",
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True,
    )


class PythonRandomState(_FrozenModel):
    """Structural representation of ``random.getstate()`` version 3."""

    version: Literal[3] = 3
    internal_state: tuple[int, ...]
    gauss_next: float | None = None

    @model_validator(mode="after")
    def _validate_state(self) -> PythonRandomState:
        if len(self.internal_state) != 625:
            raise ValueError("Python random internal_state must contain exactly 625 integers")
        if any(value < 0 or value > 2**32 - 1 for value in self.internal_state[:-1]):
            raise ValueError("Python random state words must be unsigned 32-bit integers")
        if self.internal_state[-1] < 0 or self.internal_state[-1] > 624:
            raise ValueError("Python random state index must be in [0, 624]")
        if self.gauss_next is not None and not math.isfinite(self.gauss_next):
            raise ValueError("Python random gauss_next must be finite when present")
        return self


class NumpyLegacyState(_FrozenModel):
    """Structural representation of NumPy's legacy process-global MT19937."""

    algorithm: Literal["MT19937"] = "MT19937"
    keys: tuple[int, ...]
    position: int = Field(ge=0, le=624)
    has_gauss: Literal[0, 1]
    cached_gaussian: float

    @field_validator("has_gauss", mode="before")
    @classmethod
    def _reject_boolean_flag(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("has_gauss must be the integer 0 or 1, not a boolean")
        return value

    @model_validator(mode="after")
    def _validate_state(self) -> NumpyLegacyState:
        if len(self.keys) != 624:
            raise ValueError("NumPy legacy MT19937 state must contain exactly 624 keys")
        if any(value < 0 or value > 2**32 - 1 for value in self.keys):
            raise ValueError("NumPy legacy state keys must be unsigned 32-bit integers")
        if not math.isfinite(self.cached_gaussian):
            raise ValueError("NumPy legacy cached_gaussian must be finite")
        if self.has_gauss == 0 and self.cached_gaussian != 0.0:
            raise ValueError("NumPy legacy state without a Gaussian cache must store 0.0")
        return self


class NumpyGeneratorState(_FrozenModel):
    """Structural state for the owned NumPy PCG64 generator."""

    bit_generator: Literal["PCG64"] = "PCG64"
    state: int = Field(ge=0, lt=2**128)
    increment: int = Field(ge=0, lt=2**128)
    has_uint32: Literal[0, 1]
    uinteger: int = Field(ge=0, le=2**32 - 1)

    @field_validator("has_uint32", mode="before")
    @classmethod
    def _reject_boolean_flag(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("has_uint32 must be the integer 0 or 1, not a boolean")
        return value

    @model_validator(mode="after")
    def _validate_state(self) -> NumpyGeneratorState:
        if self.increment % 2 != 1:
            raise ValueError("NumPy PCG64 increment must be odd")
        if self.has_uint32 == 0 and self.uinteger != 0:
            raise ValueError("NumPy PCG64 state without a cached uint32 must store 0")
        return self


class FrameworkRngState(_FrozenModel):
    """One provider/device byte state with an authenticated transport payload."""

    provider: str
    device: str
    encoding: Literal["base64"] = "base64"
    payload: str
    payload_sha256: str

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        if not _PROVIDER_PATTERN.fullmatch(value):
            raise ValueError("provider must be a lowercase stable identifier")
        return value

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        if not _DEVICE_PATTERN.fullmatch(value):
            raise ValueError("device must be 'cpu' or a stable '<provider>:<ordinal>' identifier")
        if value != "cpu":
            _, ordinal_text = value.rsplit(":", 1)
            if ordinal_text != str(int(ordinal_text)):
                raise ValueError("device ordinal must use canonical base-10 encoding")
        return value

    @field_validator("payload_sha256")
    @classmethod
    def _validate_digest_shape(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("payload_sha256 must be exactly 64 lowercase hexadecimal characters")
        return value

    @model_validator(mode="after")
    def _verify_payload(self) -> FrameworkRngState:
        try:
            decoded = base64.b64decode(self.payload.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as exc:
            raise ValueError("payload must be canonical ASCII base64") from exc
        if not decoded:
            raise ValueError("framework RNG payload must not be empty")
        if base64.b64encode(decoded).decode("ascii") != self.payload:
            raise ValueError("payload must use canonical padded base64 encoding")
        if hashlib.sha256(decoded).hexdigest() != self.payload_sha256:
            raise ValueError("framework RNG payload digest mismatch")
        return self

    @classmethod
    def from_bytes(cls, *, provider: str, device: str, payload: bytes) -> FrameworkRngState:
        return cls(
            provider=provider,
            device=device,
            payload=base64.b64encode(payload).decode("ascii"),
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def payload_bytes(self) -> bytes:
        return base64.b64decode(self.payload.encode("ascii"), validate=True)


def framework_state_sort_key(state: FrameworkRngState) -> tuple[str, int, str, int]:
    """Return the one canonical provider/device ordering key.

    Provider namespaces sort first. Within a provider, CPU precedes accelerator
    devices, and accelerator ordinals sort numerically rather than lexically.
    """

    if state.device == "cpu":
        return (state.provider, 0, "", 0)
    device_kind, ordinal_text = state.device.rsplit(":", 1)
    return (state.provider, 1, device_kind, int(ordinal_text))


class RngStateBundle(_FrozenModel):
    """Checkpoint-facing snapshot of the next-sample position of active RNGs."""

    rng_state_schema_version: int = Field(default=RNG_STATE_SCHEMA_VERSION)
    derivation_version: int = Field(default=SEED_DERIVATION_VERSION)
    root_seed: int
    context: SeedContext
    determinism_mode: DeterminismMode
    unsupported_determinism: UnsupportedDeterminismPolicy
    python: PythonRandomState
    numpy_legacy: NumpyLegacyState
    numpy_generator: NumpyGeneratorState
    framework_states: tuple[FrameworkRngState, ...] = ()
    warning_codes: tuple[RngWarningCode, ...] = ()

    @field_validator("rng_state_schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != RNG_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported rng_state_schema_version {value}; expected "
                f"{RNG_STATE_SCHEMA_VERSION}."
            )
        return value

    @field_validator("derivation_version")
    @classmethod
    def _validate_derivation_version(cls, value: int) -> int:
        if value != SEED_DERIVATION_VERSION:
            raise ValueError(
                f"Unsupported derivation_version {value}; expected {SEED_DERIVATION_VERSION}."
            )
        return value

    @model_validator(mode="after")
    def _validate_canonical_order(self) -> RngStateBundle:
        state_keys = tuple((state.provider, state.device) for state in self.framework_states)
        expected_states = tuple(sorted(self.framework_states, key=framework_state_sort_key))
        if self.framework_states != expected_states:
            raise ValueError("framework_states must use canonical provider/device ordinal order")
        if len(set(state_keys)) != len(state_keys):
            raise ValueError("framework_states must not contain duplicate provider/device entries")
        if self.warning_codes != tuple(sorted(self.warning_codes)):
            raise ValueError("warning_codes must be sorted")
        if len(set(self.warning_codes)) != len(self.warning_codes):
            raise ValueError("warning_codes must be unique")
        if (
            self.determinism_mode == "performance"
            and "performance_mode_enabled" not in self.warning_codes
        ):
            raise ValueError("performance mode must record performance_mode_enabled")
        if (
            self.determinism_mode == "reproducible"
            and "performance_mode_enabled" in self.warning_codes
        ):
            raise ValueError("reproducible mode cannot record performance_mode_enabled")

        framework_warning_codes = {
            "framework_determinism_unavailable",
            "accelerator_unavailable",
        }
        if not self.framework_states and framework_warning_codes.intersection(self.warning_codes):
            raise ValueError("framework warning codes require persisted framework states")
        if "framework_determinism_unavailable" in self.warning_codes and (
            self.determinism_mode != "reproducible" or self.unsupported_determinism != "warn"
        ):
            raise ValueError("framework_determinism_unavailable requires reproducible/warn policy")
        if "accelerator_unavailable" in self.warning_codes:
            devices_by_provider: dict[str, set[str]] = {}
            for state in self.framework_states:
                devices_by_provider.setdefault(state.provider, set()).add(state.device)
            cpu_only_provider_exists = any(
                "cpu" in devices and all(device == "cpu" for device in devices)
                for devices in devices_by_provider.values()
            )
            if not cpu_only_provider_exists:
                raise ValueError("accelerator_unavailable requires at least one CPU-only provider")
        return self

    def to_deterministic_json(self) -> bytes:
        """Return compact sorted-key UTF-8 JSON suitable for checkpoint embedding."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> RngStateBundle:
        if data.get("rng_state_schema_version") != RNG_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported rng_state_schema_version "
                f"{data.get('rng_state_schema_version')!r}; expected {RNG_STATE_SCHEMA_VERSION}."
            )
        if data.get("derivation_version") != SEED_DERIVATION_VERSION:
            raise ValueError(
                f"Unsupported derivation_version {data.get('derivation_version')!r}; "
                f"expected {SEED_DERIVATION_VERSION}."
            )
        payload = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return cls.model_validate_json(payload, strict=True)

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> RngStateBundle:
        try:
            text = payload.decode("utf-8")
            data = json.loads(
                text,
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("RNG state payload must be valid UTF-8 JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("RNG state JSON root must be an object")
        return cls.from_mapping(data)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is not allowed: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")
