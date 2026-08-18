from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = ROOT / "config" / "semantic_policy.json"


@dataclass(frozen=True)
class SemanticPolicy:
    relation_type: str
    require_subject_match: bool = False
    require_relation_match: bool = False
    require_answer_type_match: bool = True
    require_evidence: bool = True
    require_completeness: bool = True
    min_subject_score: float = 0.0
    min_relation_score: float = 0.0
    min_answer_type_score: float = 0.0
    min_evidence_score: float = 0.0
    min_completeness_score: float = 0.0
    allow_ranking_bypass: bool = False


@dataclass(frozen=True)
class MethodPolicy:
    method: str
    penalty: float = 1.0
    min_generation_score: float = 0.0
    min_answer_type_score: float = 0.0
    answer_type_failure_reason: str = "ANSWER_TYPE_MISMATCH"


class SemanticPolicyRegistry:
    """Versioned, auditable source of semantic gate and method policy."""

    def __init__(self, payload: dict[str, Any], source: Path):
        self.source = source
        self.version = str(payload["semantic_policy_version"])
        self.gate_order = tuple(str(item) for item in payload["gate_order"])
        self.thresholds = {
            str(name): float(value) for name, value in payload.get("thresholds", {}).items()
        }
        self.validator_thresholds = {
            str(group): {str(name): float(value) for name, value in values.items()}
            for group, values in payload.get("validator_thresholds", {}).items()
        }
        self._default_payload = dict(payload.get("default_policy", {}))
        self._policies = {
            str(name): self._build_policy(str(name), values)
            for name, values in payload.get("policies", {}).items()
        }
        self._methods = {
            str(name): MethodPolicy(method=str(name), **values)
            for name, values in payload.get("methods", {}).items()
        }
        self._validate()

    def _build_policy(self, relation_type: str, values: dict[str, Any]) -> SemanticPolicy:
        merged = {**self._default_payload, **dict(values), "relation_type": relation_type}
        allowed = {field.name for field in fields(SemanticPolicy)}
        unknown = set(merged) - allowed
        if unknown:
            raise ValueError(f"Unknown semantic policy keys for {relation_type}: {sorted(unknown)}")
        return SemanticPolicy(**merged)

    def _validate(self) -> None:
        if not self.version:
            raise ValueError("semantic_policy_version is required")
        if len(set(self.gate_order)) != len(self.gate_order):
            raise ValueError("gate_order must not contain duplicates")
        for name, value in self.thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"threshold {name} must be within 0..1")
        for group, values in self.validator_thresholds.items():
            for name, value in values.items():
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"validator threshold {group}.{name} must be within 0..1")
        for policy in self._policies.values():
            for value in (
                policy.min_subject_score,
                policy.min_relation_score,
                policy.min_answer_type_score,
                policy.min_evidence_score,
                policy.min_completeness_score,
            ):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"Policy score for {policy.relation_type} must be within 0..1")
        for method in self._methods.values():
            if not 0.0 <= method.penalty <= 1.0:
                raise ValueError(f"Method penalty for {method.method} must be within 0..1")
            if not 0.0 <= method.min_generation_score <= 1.0:
                raise ValueError(f"Method generation score for {method.method} must be within 0..1")

    def get(self, relation_type: str | None) -> SemanticPolicy:
        key = str(relation_type or "GENERAL")
        policy = self._policies.get(key)
        return policy if policy is not None else self._build_policy(key, {})

    def method(self, method: str) -> MethodPolicy:
        return self._methods.get(method, MethodPolicy(method=method))

    def threshold(self, name: str, default: float = 0.0) -> float:
        return self.thresholds.get(name, default)

    def validator_threshold(self, group: str, name: str, default: float = 0.0) -> float:
        return self.validator_thresholds.get(group, {}).get(name, default)

    @property
    def policies(self) -> tuple[SemanticPolicy, ...]:
        return tuple(self._policies.values())

    @property
    def methods(self) -> tuple[MethodPolicy, ...]:
        return tuple(self._methods.values())


def load_semantic_policy_registry(path: str | Path | None = None) -> SemanticPolicyRegistry:
    config_path = Path(path or os.getenv("QA_SEMANTIC_POLICY_CONFIG", DEFAULT_POLICY_PATH))
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SemanticPolicyRegistry(payload, config_path)


SEMANTIC_POLICIES = load_semantic_policy_registry()


__all__ = [
    "MethodPolicy",
    "SEMANTIC_POLICIES",
    "SemanticPolicy",
    "SemanticPolicyRegistry",
    "load_semantic_policy_registry",
]
