from __future__ import annotations

from pathlib import Path

import pytest

from expertforge.config.resolve import ConfigResolutionError, resolve_config

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_CONFIG = ROOT / "configs/d0/qualification.yaml"


def test_nested_d0_leaf_override_traverses_optional_section() -> None:
    envelope = resolve_config(
        QUALIFICATION_CONFIG,
        override_tokens=["d0.schedule.peak_learning_rate=0.0006"],
    )
    assert envelope.config.d0 is not None
    assert envelope.config.d0.schedule.peak_learning_rate == 0.0006


def test_optional_d0_section_itself_remains_non_leaf() -> None:
    with pytest.raises(ConfigResolutionError, match="non-leaf path"):
        resolve_config(QUALIFICATION_CONFIG, override_tokens=["d0={}"])


def test_nested_d0_seed_override_is_schema_validated() -> None:
    envelope = resolve_config(
        QUALIFICATION_CONFIG,
        override_tokens=["d0.seeds.data_order_seed.data_seed_u64=4657843784274978659"],
    )
    assert envelope.config.d0 is not None
    assert envelope.config.d0.seeds.data_order_seed.data_seed_u64 == 4_657_843_784_274_978_659
