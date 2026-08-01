"""Stable JSON Schema document for the v1 experiment manifest."""

from __future__ import annotations

from typing import Any

from expertforge.experiments.models import ExperimentManifest

__all__ = ["experiment_manifest_json_schema"]


def experiment_manifest_json_schema() -> dict[str, Any]:
    """Return the model-derived Draft 2020-12 schema plus binding conditionals."""
    schema = ExperimentManifest.model_json_schema(by_alias=True, mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://schemas.expertforge.local/experiment-manifest-v1.json"
    schema["title"] = "ExpertForge experiment manifest v1"
    schema["allOf"] = [
        {
            "if": {"properties": {"classification": {"const": "formal_experiment"}}},
            "then": {
                "required": [
                    "hypothesis",
                    "control",
                    "fixed_constraints",
                    "independent_variable",
                    "dependent_variables",
                ],
                "properties": {
                    "hypothesis": {"type": "string", "minLength": 1},
                    "control": {"type": "string", "minLength": 1},
                    "fixed_constraints": {"type": "array", "minItems": 1},
                    "independent_variable": {"type": "string", "minLength": 1},
                    "dependent_variables": {"type": "array", "minItems": 1},
                },
            },
            "else": {
                "properties": {
                    "hypothesis": {"type": "null"},
                    "control": {"type": "null"},
                    "fixed_constraints": {"maxItems": 0},
                    "independent_variable": {"type": "null"},
                    "dependent_variables": {"maxItems": 0},
                    "minimum_useful_effect": {"type": "null"},
                    "failure_threshold": {"type": "null"},
                    "kill_criterion": {"type": "null"},
                }
            },
        },
        {
            "if": {
                "properties": {"status": {"enum": ["failed", "interrupted"]}},
                "required": ["status"],
            },
            "then": {
                "required": ["outcome_diagnostic"],
                "properties": {"outcome_diagnostic": {"type": "string"}},
            },
            "else": {"properties": {"outcome_diagnostic": {"type": "null"}}},
        },
        {
            "if": {
                "properties": {
                    "identity": {
                        "properties": {"lineage": {"type": "object"}},
                        "required": ["lineage"],
                    }
                }
            },
            "then": {
                "required": ["resume_checkpoint"],
                "properties": {"resume_checkpoint": {"type": "object"}},
            },
        },
    ]
    return schema
