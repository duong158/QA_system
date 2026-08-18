from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from reader.question_semantics import QuestionSemantics
from reader.relation_validator import RelationValidation
from reader.semantic_policy import MethodPolicy, SemanticPolicy, SemanticPolicyRegistry
from reader.subject_consistency import SemanticStatus

if TYPE_CHECKING:
    from reader.candidates import AnswerCandidate


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    score: float | None = None
    threshold: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationContext:
    semantics: QuestionSemantics
    policy: SemanticPolicy
    method_policy: MethodPolicy
    relation: RelationValidation
    lexical_evidence: bool
    results: dict[str, GateResult] = field(default_factory=dict)


class CandidateValidator(Protocol):
    name: str

    def validate(
        self,
        candidate: AnswerCandidate,
        semantics: QuestionSemantics,
        context: ValidationContext,
    ) -> GateResult:
        ...


class SpanValidator:
    name = "span"

    def validate(self, candidate, semantics, context) -> GateResult:
        return GateResult(self.name, bool(candidate.valid_span), reason=None if candidate.valid_span else "NO_VALID_SPAN")


class BoundaryValidator:
    name = "boundary"

    def __init__(self, threshold: float):
        self.threshold = threshold

    def validate(self, candidate, semantics, context) -> GateResult:
        passed = candidate.boundary_score >= self.threshold
        return GateResult(
            self.name,
            passed,
            candidate.boundary_score,
            self.threshold,
            None if passed else "SPAN_BOUNDARY_INCOMPLETE",
        )


class CompletenessValidator:
    name = "completeness"

    def __init__(self, default_threshold: float):
        self.default_threshold = default_threshold

    def validate(self, candidate, semantics, context) -> GateResult:
        threshold = max(context.policy.min_completeness_score, self.default_threshold)
        required = context.policy.require_completeness
        passed = not required or (
            bool(candidate.relation_complete) and candidate.completeness_score >= threshold
        )
        reason = None
        if not passed:
            reason = (
                "DANGLING_CONNECTOR"
                if "DANGLING_CONNECTOR" in candidate.completeness_reasons
                else "INCOMPLETE_RELATION"
            )
        return GateResult(self.name, passed, candidate.completeness_score, threshold if required else None, reason)


class EvidenceValidator:
    name = "evidence"

    def __init__(self, default_threshold: float):
        self.default_threshold = default_threshold

    def validate(self, candidate, semantics, context) -> GateResult:
        threshold = max(context.policy.min_evidence_score, self.default_threshold)
        required = context.policy.require_evidence
        passed = not required or (context.lexical_evidence and candidate.evidence_score >= threshold)
        return GateResult(
            self.name,
            passed,
            candidate.evidence_score,
            threshold if required else None,
            None if passed else "EVIDENCE_UNSUPPORTED",
        )


class SubjectValidator:
    name = "subject"

    def validate(self, candidate, semantics, context) -> GateResult:
        required = context.policy.require_subject_match
        score = context.relation.subject_match_score
        threshold = context.policy.min_subject_score
        valid_status = context.relation.subject_status == SemanticStatus.VALID.value
        passed = not required or (valid_status and score >= threshold)
        reason = None
        if not passed:
            relation_reason = context.relation.reason
            reason = relation_reason if "SUBJECT_MISMATCH" in relation_reason else "SUBJECT_MISMATCH"
        return GateResult(self.name, passed, score, threshold if required else None, reason)


class RelationValidator:
    name = "relation"

    def validate(self, candidate, semantics, context) -> GateResult:
        required = context.policy.require_relation_match
        score = context.relation.relation_score
        threshold = context.policy.min_relation_score
        
        # The user requested RELATION_MISMATCH to only warn, not reject candidates.
        # So we always set passed to True for the relation gate.
        passed = True
        
        # We keep the actual reason (which could be RELATION_MISMATCH) as a warning.
        reason = context.relation.reason or "RELATION_UNSUPPORTED"
        if context.relation.status == SemanticStatus.VALID.value and context.relation.relation_evidence and score >= threshold:
             reason = None # Clear reason if it actually fully passed without mismatch
             
        return GateResult(self.name, passed, score, threshold if required else None, reason)


class AnswerTypeValidator:
    name = "answer_type"

    def __init__(self, default_threshold: float):
        self.default_threshold = default_threshold

    def validate(self, candidate, semantics, context) -> GateResult:
        threshold = max(
            self.default_threshold,
            context.policy.min_answer_type_score,
            context.method_policy.min_answer_type_score,
        )
        required = context.policy.require_answer_type_match
        passed = not required or candidate.answer_type_score >= threshold
        return GateResult(
            self.name,
            passed,
            candidate.answer_type_score,
            threshold if required else None,
            None if passed else context.method_policy.answer_type_failure_reason,
        )


class RankingValidator:
    name = "ranking"
    _semantic_prerequisites = ("evidence", "subject", "relation", "answer_type")

    def __init__(self, threshold: float):
        self.threshold = threshold

    def validate(self, candidate, semantics, context) -> GateResult:
        grounded = context.policy.allow_ranking_bypass and all(
            context.results[name].passed
            for name in self._semantic_prerequisites
            if name in context.results
        )
        passed = candidate.ranking_score >= self.threshold or grounded
        return GateResult(
            self.name,
            passed,
            candidate.ranking_score,
            self.threshold,
            None if passed else "LOW_RANKING_SCORE",
        )


def build_validator_registry(policies: SemanticPolicyRegistry) -> dict[str, CandidateValidator]:
    return {
        "span": SpanValidator(),
        "boundary": BoundaryValidator(policies.threshold("boundary")),
        "completeness": CompletenessValidator(policies.threshold("completeness")),
        "evidence": EvidenceValidator(policies.threshold("evidence")),
        "subject": SubjectValidator(),
        "relation": RelationValidator(),
        "answer_type": AnswerTypeValidator(policies.threshold("answer_type")),
        "ranking": RankingValidator(policies.threshold("ranking")),
    }


def evaluate_candidate_gates(
    candidate: AnswerCandidate,
    semantics: QuestionSemantics,
    context: ValidationContext,
    policies: SemanticPolicyRegistry,
) -> dict[str, GateResult]:
    validators = build_validator_registry(policies)
    for name in policies.gate_order:
        validator = validators.get(name)
        if validator is None:
            raise ValueError(f"No validator registered for gate {name!r}")
        context.results[name] = validator.validate(candidate, semantics, context)
    return context.results


def first_gate_failure(gate_results: dict[str, GateResult]) -> str | None:
    for result in gate_results.values():
        if not result.passed:
            return result.reason or f"{result.name.upper()}_GATE_FAILED"
    return None


__all__ = [
    "CandidateValidator",
    "GateResult",
    "ValidationContext",
    "build_validator_registry",
    "evaluate_candidate_gates",
    "first_gate_failure",
]
