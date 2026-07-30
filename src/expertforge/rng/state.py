"""Versioned, immutable, non-pickle RNG state for checkpoint embedding."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from expertforge.rng.derivation import SEED_DERIVATION_VERSION, SeedContext

__all__ = [
    "RNG_STATE_SCHEMA_VERSION",
    "FrameworkRngState",
    "NumpyGeneratorState",
    "NumpyLegacyState",
    "PythonRandomState",
    "RngStateBundle",
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

    @model_validator(mode="after")
    def _validate_state(self) -> NumpyLegacyState:
        if len(self.keys) != 624:
            raise ValueError("NumPy legacy MT19937 state must contain exactly 624 keys")
        if any(value < 0 or value > 2**32 - 1 for value in self.keys):
            raise ValueError("NumPy legacy state keys must be unsigned 32-bit integers")
        if not math.isfinite(self.cached_gaussian):
            raise ValueError("NumPy legacy cached_gaussian must be finite")
        return self


class NumpyGeneratorState(_FrozenModel):
    """Structural state for the owned NumPy PCG64 generator."""

    bit_generator: Literal["PCG64"] = "PCG64"
    state: int = Field(ge=0, lt=2**128)
    increment: int = Field(ge=0, lt=2**128)
    has_uint32: Literal[0, 1]
    uinteger: int = Field(ge=0, le=2**32 - 1)


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
        if state_keys != tuple(sorted(state_keys)):
            raise ValueError("framework_states must be sorted by provider and device")
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
        return cls.model_validate(data, strict=False)

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> RngStateBundle:
        try:
            text = payload.decode("utf-8")
            data = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("RNG state payload must be valid UTF-8 JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("RNG state JSON root must be an object")
        return cls.from_mapping(data)
