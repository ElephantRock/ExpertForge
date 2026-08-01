"""Stable JSON Schema document for the v1 experiment manifest."""

from __future__ import annotations

from typing import Any

from expertforge.experiments.models import ExperimentManifest

__all__ = ["experiment_manifest_json_schema"]

_ARTIFACT_ID_PATTERN = r"^artifact-v1-sha256-[0-9a-f]{64}$"
_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_SPECIFICATION_FINGERPRINT_PATTERN = r"^spec-v1-sha256-[0-9a-f]{64}$"
_RUN_ID_PATTERN = r"^run-\d{8}t\d{6}z-[0-9a-f]{12}-[0-9a-f]{20}$"
_ATTEMPT_ID_PATTERN = r"^attempt-\d{8}t\d{6}z-[0-9a-f]{20}$"


def _patch_patterns(schema: dict[str, Any]) -> None:
    defs = schema.get("$defs", {})
    artifact = defs.get("ArtifactRecord", {}).get("properties", {})
    artifact.get("artifact_id", {})["pattern"] = _ARTIFACT_ID_PATTERN
    artifact.get("content_digest", {})["pattern"] = _SHA256_PATTERN
    artifact.get("specification_fingerprint", {})[
        "pattern"
    ] = _SPECIFICATION_FINGERPRINT_PATTERN
    artifact.get("run_id", {})["pattern"] = _RUN_ID_PATTERN
    artifact.get("attempt_id", {})["pattern"] = _ATTEMPT_ID_PATTERN

    parent = defs.get("ParentReference", {}).get("properties", {})
    parent.get("artifact_id", {})["pattern"] = _ARTIFACT_ID_PATTERN
    parent.get("run_id", {})["pattern"] = _RUN_ID_PATTERN
    parent.get("attempt_id", {})["pattern"] = _ATTEMPT_ID_PATTERN

    identity = defs.get("AttemptIdentityRecord", {}).get("properties", {})
    identity.get("run_id", {})["pattern"] = _RUN_ID_PATTERN
    identity.get("attempt_id", {})["pattern"] = _ATTEMPT_ID_PATTERN

    lineage = defs.get("ResumeLineage", {}).get("properties", {})
    lineage.get("parent_run_id", {})["pattern"] = _RUN_ID_PATTERN
    lineage.get("parent_attempt_id", {}).setdefault(
        "anyOf", [{"type": "string"}, {"type": "null"}]
    )[0]["pattern"] = _ATTEMPT_ID_PATTERN
    lineage.get("parent_checkpoint_id", {})["pattern"] = _ARTIFACT_ID_PATTERN

    fingerprint = defs.get("SpecificationFingerprintRecord", {}).get("properties", {})
    fingerprint.get("digest_str", {})[
        "pattern"
    ] = _SPECIFICATION_FINGERPRINT_PATTERN


def _classification_conditional() -> dict[str, Any]:
    formal_fields = [
        "hypothesis",
        "control",
        "fixed_constraints",
        "independent_variable",
        "dependent_variables",
        "minimum_useful_effect",
        "failure_threshold",
        "kill_criterion",
    ]
    return {
        "if": {
            "properties": {"classification": {"const": "formal_experiment"}},
            "required": ["classification"],
        },
        "then": {
            "required": formal_fields,
            "properties": {
                "hypothesis": {"type": "string", "minLength": 1},
                "control": {"type": "string", "minLength": 1},
                "fixed_constraints": {"type": "array", "minItems": 1},
                "independent_variable": {"type": "string", "minLength": 1},
                "dependent_variables": {"type": "array", "minItems": 1},
                "minimum_useful_effect": {"type": "string", "minLength": 1},
                "failure_threshold": {"type": "string", "minLength": 1},
                "kill_criterion": {"type": "string", "minLength": 1},
                "smoke_objective": {"type": "null"},
                "smoke_acceptance_criteria": {"type": "null"},
            },
        },
        "else": {
            "required": ["smoke_objective", "smoke_acceptance_criteria"],
            "properties": {
                "smoke_objective": {"type": "string", "minLength": 1},
                "smoke_acceptance_criteria": {"type": "string", "minLength": 1},
                "hypothesis": {"type": "null"},
                "control": {"type": "null"},
                "fixed_constraints": {"maxItems": 0},
                "independent_variable": {"type": "null"},
                "dependent_variables": {"maxItems": 0},
                "minimum_useful_effect": {"type": "null"},
                "failure_threshold": {"type": "null"},
                "kill_criterion": {"type": "null"},
            },
        },
    }


def _fixture_equality_conditionals() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for source, target in (
        ("dataset_identity", "tokenizer_identity"),
        ("tokenizer_identity", "dataset_identity"),
    ):
        for value in (True, False):
            rules.append(
                {
                    "if": {
                        "properties": {
                            source: {
                                "type": "object",
                                "properties": {"is_fixture": {"const": value}},
                                "required": ["is_fixture"],
                            }
                        },
                        "required": [source],
                    },
                    "then": {
                        "properties": {
                            target: {
                                "anyOf": [
                                    {"type": "null"},
                                    {
                                        "type": "object",
                                        "properties": {"is_fixture": {"const": value}},
                                        "required": ["is_fixture"],
                                    },
                                ]
                            }
                        }
                    },
                }
            )
    return rules


def experiment_manifest_json_schema() -> dict[str, Any]:
    """Return the model-derived Draft 2020-12 schema plus binding conditionals."""
    schema = ExperimentManifest.model_json_schema(by_alias=True, mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://schemas.expertforge.local/experiment-manifest-v1.json"
    schema["title"] = "ExpertForge experiment manifest v1"
    _patch_patterns(schema)
    schema["allOf"] = [
        _classification_conditional(),
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
                        "type": "object",
                        "properties": {"lineage": {"type": "object"}},
                        "required": ["lineage"],
                    }
                },
                "required": ["identity"],
            },
            "then": {
                "required": ["resume_checkpoint"],
                "properties": {"resume_checkpoint": {"type": "object"}},
            },
            "else": {"properties": {"resume_checkpoint": {"type": "null"}}},
        },
        {
            "if": {
                "properties": {
                    "evidence": {
                        "type": "object",
                        "properties": {"status": {"const": "complete"}},
                        "required": ["status"],
                    }
                },
                "required": ["evidence"],
            },
            "then": {
                "properties": {
                    "evidence": {
                        "properties": {"missing": {"maxItems": 0}},
                        "required": ["missing"],
                    }
                }
            },
            "else": {
                "properties": {
                    "evidence": {
                        "properties": {"missing": {"minItems": 1}},
                        "required": ["missing"],
                    }
                }
            },
        },
        {
            "if": {
                "properties": {"status": {"const": "completed"}},
                "required": ["status"],
            },
            "then": {
                "required": ["evaluation_summary"],
                "properties": {
                    "evaluation_summary": {"type": "object"},
                    "evidence": {
                        "properties": {"status": {"const": "complete"}},
                        "required": ["status"],
                    },
                },
            },
            "else": {
                "properties": {
                    "result": {"type": "null"},
                    "decision": {"type": "null"},
                }
            },
        },
        {
            "if": {
                "properties": {"checkpoint_evidence_required": {"const": True}},
                "required": ["checkpoint_evidence_required"],
            },
            "then": {"properties": {"checkpoint_artifacts": {"minItems": 1}}},
        },
        {
            "if": {
                "properties": {"generated_output_evidence_required": {"const": True}},
                "required": ["generated_output_evidence_required"],
            },
            "then": {"properties": {"generated_output_artifacts": {"minItems": 1}}},
        },
        *_fixture_equality_conditionals(),
    ]
    return schema
