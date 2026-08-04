"""Parameter-identity tests for the D0 model.

These tests pin the parameter inventory exactly:
- tensor instance counts and total parameter elements for both profiles,
- the 11 parameter families,
- tied embedding/output projection (no separate output_head),
- absence of biases and RoPE parameters,
- exact state-dict names (expanded with layer indices).

The canonical profile is constructed on the meta device so the heavy model
never allocates real memory — only shapes and names are validated.
"""

from __future__ import annotations

import re

import torch

from expertforge.d0.model.config import canonical_config, qualification_config
from expertforge.d0.model.transformer import D0Model

EXPECTED_FAMILIES = {
    "token_embedding.weight",
    "blocks.{i}.attention_norm.weight",
    "blocks.{i}.attention.q_proj.weight",
    "blocks.{i}.attention.k_proj.weight",
    "blocks.{i}.attention.v_proj.weight",
    "blocks.{i}.attention.out_proj.weight",
    "blocks.{i}.ffn_norm.weight",
    "blocks.{i}.ffn.gate_proj.weight",
    "blocks.{i}.ffn.up_proj.weight",
    "blocks.{i}.ffn.down_proj.weight",
    "final_norm.weight",
}

# Per-family expected element counts for the qualification profile, keyed by the
# unexpanded family name. Sourced from parameter-inventory-v1.json.
QUALIFICATION_FAMILY_ELEMENTS = {
    "token_embedding.weight": 12_865_792,
    "blocks.{i}.attention_norm.weight": 2_048,
    "blocks.{i}.attention.q_proj.weight": 524_288,
    "blocks.{i}.attention.k_proj.weight": 524_288,
    "blocks.{i}.attention.v_proj.weight": 524_288,
    "blocks.{i}.attention.out_proj.weight": 524_288,
    "blocks.{i}.ffn_norm.weight": 2_048,
    "blocks.{i}.ffn.gate_proj.weight": 1_572_864,
    "blocks.{i}.ffn.up_proj.weight": 1_572_864,
    "blocks.{i}.ffn.down_proj.weight": 1_572_864,
    "final_norm.weight": 256,
}

CANONICAL_FAMILY_ELEMENTS = {
    "token_embedding.weight": 28_948_032,
    "blocks.{i}.attention_norm.weight": 6_912,
    "blocks.{i}.attention.q_proj.weight": 3_981_312,
    "blocks.{i}.attention.k_proj.weight": 3_981_312,
    "blocks.{i}.attention.v_proj.weight": 3_981_312,
    "blocks.{i}.attention.out_proj.weight": 3_981_312,
    "blocks.{i}.ffn_norm.weight": 6_912,
    "blocks.{i}.ffn.gate_proj.weight": 10_616_832,
    "blocks.{i}.ffn.up_proj.weight": 10_616_832,
    "blocks.{i}.ffn.down_proj.weight": 10_616_832,
    "final_norm.weight": 576,
}


def _family(name: str) -> str:
    return re.sub(r"blocks\.\d+\.", "blocks.{i}.", name)


def test_qualification_parameter_counts() -> None:
    model = D0Model(qualification_config())

    tensors = list(model.state_dict())
    elements = sum(p.numel() for p in model.parameters())

    assert len(tensors) == 74
    assert elements == 19_685_888


def test_qualification_family_element_counts() -> None:
    model = D0Model(qualification_config())

    by_family: dict[str, int] = {}
    for name, tensor in model.state_dict().items():
        by_family[_family(name)] = by_family.get(_family(name), 0) + int(
            torch.tensor(tensor.shape).prod().item()
        )

    for family, expected in QUALIFICATION_FAMILY_ELEMENTS.items():
        assert by_family[family] == expected, family


def test_canonical_parameter_counts_on_meta_device() -> None:
    # Meta device validates shapes/counts without allocating memory.
    with torch.device("meta"):
        model = D0Model(canonical_config())

    tensors = list(model.state_dict())
    elements = sum(p.numel() for p in model.parameters())

    assert len(tensors) == 110
    assert elements == 76_738_176


def test_canonical_family_element_counts_on_meta_device() -> None:
    with torch.device("meta"):
        model = D0Model(canonical_config())

    by_family: dict[str, int] = {}
    for name, tensor in model.state_dict().items():
        by_family[_family(name)] = by_family.get(_family(name), 0) + int(
            torch.tensor(tensor.shape).prod().item()
        )

    for family, expected in CANONICAL_FAMILY_ELEMENTS.items():
        assert by_family[family] == expected, family


def test_exactly_eleven_parameter_families() -> None:
    model = D0Model(qualification_config())

    families = {_family(name) for name in model.state_dict()}
    assert families == EXPECTED_FAMILIES
    assert len(families) == 11


def test_tied_embedding_no_separate_output_head() -> None:
    model = D0Model(qualification_config())
    names = set(model.state_dict())

    assert "token_embedding.weight" in names
    assert not any("output_head" in name for name in names)
    assert not any(name.endswith("lm_head.weight") for name in names)


def test_no_bias_parameters() -> None:
    model = D0Model(qualification_config())

    assert not any("bias" in name for name in model.state_dict())
    for module in model.modules():
        for child_name, child in module.named_children():
            if hasattr(child, "bias"):
                assert child.bias is None, child_name


def test_no_trainable_rope_parameters() -> None:
    model = D0Model(qualification_config())

    assert not any("rope" in name.lower() for name in model.state_dict())
    assert not any("inv_freq" in name for name in model.state_dict())


def test_no_persistent_rope_cache_keys() -> None:
    model = D0Model(qualification_config())

    state_dict = model.state_dict()
    for key in state_dict:
        assert "cos" not in key
        assert "sin" not in key
        assert "cached" not in key.lower()


def test_exact_state_dict_names_qualification() -> None:
    model = D0Model(qualification_config())
    names = list(model.state_dict())

    expected = ["token_embedding.weight"]
    for i in range(8):
        expected += [
            f"blocks.{i}.attention_norm.weight",
            f"blocks.{i}.attention.q_proj.weight",
            f"blocks.{i}.attention.k_proj.weight",
            f"blocks.{i}.attention.v_proj.weight",
            f"blocks.{i}.attention.out_proj.weight",
            f"blocks.{i}.ffn_norm.weight",
            f"blocks.{i}.ffn.gate_proj.weight",
            f"blocks.{i}.ffn.up_proj.weight",
            f"blocks.{i}.ffn.down_proj.weight",
        ]
    expected.append("final_norm.weight")

    assert names == expected


def test_projection_weight_shapes_match_inventory() -> None:
    model = D0Model(qualification_config())
    sd = model.state_dict()

    # [d, d] for attention projections; [f, d] for gate/up; [d, f] for down.
    d, f = 256, 768
    assert tuple(sd["blocks.0.attention.q_proj.weight"].shape) == (d, d)
    assert tuple(sd["blocks.0.attention.k_proj.weight"].shape) == (d, d)
    assert tuple(sd["blocks.0.attention.v_proj.weight"].shape) == (d, d)
    assert tuple(sd["blocks.0.attention.out_proj.weight"].shape) == (d, d)
    assert tuple(sd["blocks.0.ffn.gate_proj.weight"].shape) == (f, d)
    assert tuple(sd["blocks.0.ffn.up_proj.weight"].shape) == (f, d)
    assert tuple(sd["blocks.0.ffn.down_proj.weight"].shape) == (d, f)
    assert tuple(sd["token_embedding.weight"].shape) == (50257, d)
    assert tuple(sd["final_norm.weight"].shape) == (d,)


def test_forward_returns_logits_shape() -> None:
    model = D0Model(qualification_config()).eval()
    ids = torch.zeros((1, 4), dtype=torch.long)

    with torch.no_grad():
        logits = model(ids)

    assert tuple(logits.shape) == (1, 4, 50257)
