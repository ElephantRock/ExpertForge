"""Focused interchange-schema contract regressions for Issue #12."""

from __future__ import annotations

from typing import Any

from expertforge.experiments import experiment_manifest_json_schema


def _completed_rule(schema: dict[str, Any]) -> dict[str, Any]:
    for rule in schema["allOf"]:
        condition = rule.get("if", {})
        status = condition.get("properties", {}).get("status")
        if status == {"const": "completed"}:
            return rule
    raise AssertionError("completed-status conditional is missing")


def test_completed_schema_requires_all_core_evidence() -> None:
    schema = experiment_manifest_json_schema()
    rule = _completed_rule(schema)
    then = rule["then"]
    assert set(then["required"]) == {
        "configuration_artifact",
        "provenance_artifact",
        "dataset_identity",
        "tokenizer_identity",
        "model_descriptor",
        "training_budget",
        "evaluation_summary",
        "telemetry_artifacts",
    }
    properties = then["properties"]
    for name in (
        "configuration_artifact",
        "provenance_artifact",
        "dataset_identity",
        "tokenizer_identity",
        "model_descriptor",
        "training_budget",
        "evaluation_summary",
    ):
        assert properties[name] == {"type": "object"}
    assert properties["telemetry_artifacts"] == {"type": "array", "minItems": 1}
    assert properties["evidence"]["properties"]["status"] == {"const": "complete"}
