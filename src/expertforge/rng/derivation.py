"""Versioned, domain-separated deterministic seed derivation.

The full SHA-256 digest identifies a logical random stream. Integer projections
exist only for provider APIs and are not collision-resistant stream identities.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "SEED_DERIVATION_SCHEMA",
    "SEED_DERIVATION_VERSION",
    "DerivedSeed",
    "SeedContext",
    "derive_seed",
    "seed_derivation_bytes",
]

SEED_DERIVATION_SCHEMA = "expertforge.seed-derivation"
SEED_DERIVATION_VERSION = 1
_COMPONENT_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_INDEX = 2**32 - 1


class SeedContext(BaseModel):
    """Stable coordinates for one logical random stream."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, validate_default=True)

    component: str
    worker: int = Field(default=0, ge=0, le=_MAX_INDEX)
    rank: int = Field(default=0, ge=0, le=_MAX_INDEX)
    device: int = Field(default=0, ge=0, le=_MAX_INDEX)
    stream: int = Field(default=0, ge=0, le=_MAX_INDEX)

    @field_validator("component")
    @classmethod
    def _validate_component(cls, value: str) -> str:
        if not _COMPONENT_PATTERN.fullmatch(value):
            raise ValueError(
                "component must be a lowercase stable identifier matching "
                "[a-z][a-z0-9._-]{0,127}"
            )
        return value

    def child(
        self,
        component: str,
        *,
        worker: int | None = None,
        rank: int | None = None,
        device: int | None = None,
        stream: int | None = None,
    ) -> SeedContext:
        """Return a context derived from this context without mutating it."""

        return SeedContext(
            component=component,
            worker=self.worker if worker is None else worker,
            rank=self.rank if rank is None else rank,
            device=self.device if device is None else device,
            stream=self.stream if stream is None else stream,
        )


class DerivedSeed(BaseModel):
    """A verified derived stream identity and fixed provider projections."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, validate_default=True)

    derivation_version: int = Field(default=SEED_DERIVATION_VERSION)
    root_seed: int
    context: SeedContext
    digest: str
    seed_u64: int = Field(ge=0, le=2**64 - 1)
    seed_u32: int = Field(ge=0, le=2**32 - 1)

    @field_validator("derivation_version")
    @classmethod
    def _validate_version(cls, value: int) -> int:
        if value != SEED_DERIVATION_VERSION:
            raise ValueError(
                f"Unsupported derivation_version {value}; expected {SEED_DERIVATION_VERSION}."
            )
        return value

    @field_validator("digest")
    @classmethod
    def _validate_digest_shape(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("digest must be exactly 64 lowercase hexadecimal characters")
        return value

    @model_validator(mode="after")
    def _verify_derived_fields(self) -> DerivedSeed:
        expected = derive_seed(self.root_seed, self.context)
        if self.digest != expected.digest:
            raise ValueError("derived seed digest does not match root seed and context")
        if self.seed_u64 != expected.seed_u64 or self.seed_u32 != expected.seed_u32:
            raise ValueError("derived seed projections do not match the digest")
        return self

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> DerivedSeed:
        version = data.get("derivation_version")
        if version != SEED_DERIVATION_VERSION:
            raise ValueError(
                f"Unsupported derivation_version {version!r}; expected "
                f"{SEED_DERIVATION_VERSION}."
            )
        return cls.model_validate(data, strict=False)


def _envelope(root_seed: int, context: SeedContext) -> dict[str, object]:
    return {
        "schema": SEED_DERIVATION_SCHEMA,
        "version": SEED_DERIVATION_VERSION,
        "root_seed": str(root_seed),
        "component": context.component,
        "worker": context.worker,
        "rank": context.rank,
        "device": context.device,
        "stream": context.stream,
    }


def seed_derivation_bytes(root_seed: int, context: SeedContext) -> bytes:
    """Return canonical UTF-8 bytes for the derivation envelope."""

    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise TypeError("root_seed must be an integer, not a bool or coercible value")
    return json.dumps(
        _envelope(root_seed, context),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def derive_seed(root_seed: int, context: SeedContext) -> DerivedSeed:
    """Derive one stable stream identity from a root seed and explicit context."""

    digest_bytes = hashlib.sha256(seed_derivation_bytes(root_seed, context)).digest()
    digest = digest_bytes.hex()
    return DerivedSeed.model_construct(
        derivation_version=SEED_DERIVATION_VERSION,
        root_seed=root_seed,
        context=context,
        digest=digest,
        seed_u64=int.from_bytes(digest_bytes[:8], "big", signed=False),
        seed_u32=int.from_bytes(digest_bytes[8:12], "big", signed=False),
    )
