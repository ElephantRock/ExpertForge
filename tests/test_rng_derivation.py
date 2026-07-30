"""Regression tests for versioned, domain-separated seed derivation."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from expertforge.rng.derivation import (
    SEED_DERIVATION_SCHEMA,
    SEED_DERIVATION_VERSION,
    DerivedSeed,
    SeedContext,
    derive_seed,
    seed_derivation_bytes,
)


def test_seed_derivation_is_stable() -> None:
    context = SeedContext(component="model.init", worker=2, rank=3, device=1, stream=4)

    first = derive_seed(-17, context)
    second = derive_seed(-17, context)

    assert first == second
    assert len(first.digest) == 64
    assert 0 <= first.seed_u64 < 2**64
    assert 0 <= first.seed_u32 < 2**32


def test_seed_derivation_canonical_envelope() -> None:
    context = SeedContext(component="data.shuffle", worker=1, rank=2, device=3, stream=4)

    payload = seed_derivation_bytes(9, context)

    assert (
        payload
        == json.dumps(
            {
                "component": "data.shuffle",
                "device": 3,
                "rank": 2,
                "root_seed": "9",
                "schema": SEED_DERIVATION_SCHEMA,
                "stream": 4,
                "version": SEED_DERIVATION_VERSION,
                "worker": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


@pytest.mark.parametrize("field", ["worker", "rank", "device", "stream"])
def test_each_coordinate_changes_stream(field: str) -> None:
    base = {"component": "sampling", "worker": 0, "rank": 0, "device": 0, "stream": 0}
    changed = dict(base)
    changed[field] = 1

    assert (
        derive_seed(7, SeedContext(**base)).digest != derive_seed(7, SeedContext(**changed)).digest
    )


def test_component_and_root_seed_change_stream() -> None:
    first = derive_seed(7, SeedContext(component="model.init"))
    assert first.digest != derive_seed(8, SeedContext(component="model.init")).digest
    assert first.digest != derive_seed(7, SeedContext(component="model.dropout")).digest


def test_worker_rank_device_grid_has_unique_full_digests() -> None:
    digests = {
        derive_seed(
            1234,
            SeedContext(
                component="data.worker",
                worker=worker,
                rank=rank,
                device=device,
                stream=stream,
            ),
        ).digest
        for worker in range(8)
        for rank in range(8)
        for device in range(2)
        for stream in range(2)
    }

    assert len(digests) == 8 * 8 * 2 * 2


def test_derived_seed_round_trip_verifies_digest_and_projections() -> None:
    seed = derive_seed(42, SeedContext(component="evaluation.sampling", stream=5))

    loaded = DerivedSeed.from_mapping(seed.model_dump(mode="json"))

    assert loaded == seed


@pytest.mark.parametrize("field", ["digest", "seed_u64", "seed_u32"])
def test_derived_seed_rejects_tampering(field: str) -> None:
    seed = derive_seed(42, SeedContext(component="evaluation.sampling"))
    data = seed.model_dump(mode="json")
    if field == "digest":
        data[field] = "0" * 64 if seed.digest != "0" * 64 else "1" * 64
    else:
        data[field] = int(data[field]) ^ 1

    with pytest.raises(ValidationError, match="does not match"):
        DerivedSeed.model_validate(data, strict=False)


def test_derived_seed_rejects_unknown_version() -> None:
    seed = derive_seed(42, SeedContext(component="evaluation.sampling"))
    data = seed.model_dump(mode="json")
    data["derivation_version"] = 2

    with pytest.raises(ValueError, match="Unsupported derivation_version"):
        DerivedSeed.from_mapping(data)


@pytest.mark.parametrize(
    "component",
    ["Uppercase", "has space", "../path", "", "a/b", "a" * 129],
)
def test_context_rejects_noncanonical_component(component: str) -> None:
    with pytest.raises(ValidationError, match="component"):
        SeedContext(component=component)


@pytest.mark.parametrize("value", [-1, 2**32])
def test_context_rejects_out_of_range_indices(value: int) -> None:
    with pytest.raises(ValidationError):
        SeedContext(component="worker", worker=value)


def test_root_seed_rejects_bool_and_coercion() -> None:
    context = SeedContext(component="run")

    with pytest.raises(TypeError, match="root_seed"):
        derive_seed(True, context)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="root_seed"):
        derive_seed("7", context)  # type: ignore[arg-type]
