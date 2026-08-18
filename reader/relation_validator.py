from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from reader.cause_relations import assess_cause_candidate, fold_text
from reader.question_semantics import QuestionSemantics
from reader.subject_consistency import SemanticStatus, score_subject_consistency


@dataclass(frozen=True)
class RelationValidation:
    status: str
    relation_type: str
    relation_score: float
    relation_evidence: bool
    reason: str
    subject_match_score: float
    subject_match_reason: str
    target_relation_score: float = 0.0
    cause_pattern_score: float = 0.0
    relation_method: str | None = None
    evidence_sentence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STRICT_RELATIONS = {"CAUSE", "BIRTH_TIME", "DEATH_TIME", "PURPOSE"}


def _evidence_sentence(
    context: str,
    answer: str,
    start: int,
    end: int,
    details: dict[str, Any],
) -> str:
    explicit = str(details.get("sentence_answer") or details.get("evidence_sentence") or "").strip()
    if explicit:
        return explicit
    if not context:
        return ""
    position = start if 0 <= start < len(context) else context.find(answer)
    if position < 0:
        return ""
    left = max(context.rfind(mark, 0, position) for mark in ".!?\n") + 1
    right_candidates = [context.find(mark, max(end, position)) for mark in ".!?\n"]
    right_candidates = [item for item in right_candidates if item >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(context)
    return context[left:right].strip()


def _effect_repetition(semantics: QuestionSemantics, answer: str) -> bool:
    answer_folded = " ".join(re.findall(r"[\w]+", fold_text(answer)))
    if not answer_folded:
        return False
    predicate = " ".join(re.findall(r"[\w]+", fold_text(semantics.predicate or "")))
    subject = " ".join(re.findall(r"[\w]+", fold_text(semantics.subject or "")))
    if predicate and predicate in answer_folded:
        return True
    return bool(subject and predicate and subject in answer_folded and predicate in answer_folded)


def _birth_or_death_time_validation(
    semantics: QuestionSemantics,
    evidence: str,
    answer: str,
    context: str,
) -> RelationValidation:
    subject = score_subject_consistency(semantics, evidence, answer, context)
    mismatch_reason = (
        "TIME_SUBJECT_MISMATCH"
        if subject.status == SemanticStatus.INVALID.value
        else f"{semantics.relation}_RELATION_MISMATCH"
    )
    if subject.status != SemanticStatus.VALID.value:
        return RelationValidation(
            status=SemanticStatus.INVALID.value,
            relation_type=semantics.relation,
            relation_score=0.0,
            relation_evidence=False,
            reason=mismatch_reason,
            subject_match_score=subject.score,
            subject_match_reason=subject.reason,
            evidence_sentence=evidence,
        )

    folded = fold_text(evidence)
    folded_answer = " ".join(re.findall(r"[\w]+", fold_text(answer)))
    answer_index = folded.find(fold_text(answer).strip())
    relation_pattern = (
        r"\b(?:sinh|ra doi)\b" if semantics.relation == "BIRTH_TIME"
        else r"\b(?:mat|qua doi|tu tran)\b"
    )
    explicit = bool(re.search(relation_pattern, folded))

    # Biography lead: "Subject (birth date – death date) ...". The first
    # parenthetical date is birth time and the second is death time.
    biography = False
    subject_folded = fold_text(semantics.subject or "").strip()
    subject_index = folded.find(subject_folded)
    if subject_index >= 0 and answer_index >= 0:
        open_paren = folded.find("(", subject_index + len(subject_folded))
        close_paren = folded.find(")", open_paren + 1) if open_paren >= 0 else -1
        if open_paren >= 0 and open_paren < answer_index < close_paren:
            separator_positions = [
                position
                for mark in ("–", "—", "-")
                if (position := evidence.find(mark, open_paren, close_paren)) >= 0
            ]
            separator = min(separator_positions) if separator_positions else -1
            if semantics.relation == "BIRTH_TIME":
                biography = separator < 0 or answer_index < separator
            else:
                biography = separator >= 0 and answer_index > separator

    matched = bool(folded_answer and (explicit or biography))
    score = 0.98 if explicit else (0.92 if biography else 0.0)
    return RelationValidation(
        status=SemanticStatus.VALID.value if matched else SemanticStatus.INVALID.value,
        relation_type=semantics.relation,
        relation_score=score,
        relation_evidence=matched,
        reason=(
            "DIRECT_TIME_RELATION"
            if explicit
            else "BIOGRAPHY_PARENTHETICAL_TIME"
            if biography
            else f"{semantics.relation}_RELATION_MISMATCH"
        ),
        subject_match_score=subject.score,
        subject_match_reason=subject.reason,
        target_relation_score=score,
        relation_method="time_relation_pattern" if matched else None,
        evidence_sentence=evidence,
    )


def validate_candidate_relation(
    semantics: QuestionSemantics,
    question: str,
    context: str,
    answer: str,
    start: int,
    end: int,
    candidate_method: str,
    candidate_details: dict[str, Any] | None = None,
) -> RelationValidation:
    details = candidate_details or {}
    evidence = _evidence_sentence(context, answer, start, end, details)

    if semantics.relation == "CAUSE":
        cause = assess_cause_candidate(question, context, answer, evidence or None)
        if cause.relation_evidence:
            return RelationValidation(
                status=SemanticStatus.VALID.value,
                relation_type="CAUSE",
                relation_score=cause.relation_score,
                relation_evidence=True,
                reason="DIRECT_CAUSE_PATTERN",
                subject_match_score=cause.subject_match_score,
                subject_match_reason="CAUSE_FRAME_SUBJECT_MATCH",
                target_relation_score=cause.target_relation_score,
                cause_pattern_score=cause.cause_pattern_score,
                relation_method=cause.relation_method,
                evidence_sentence=cause.evidence_sentence,
            )
        if _effect_repetition(semantics, answer):
            reason = "CAUSE_EFFECT_REPETITION"
            status = SemanticStatus.INVALID.value
        elif cause.rejection_reason in {"CAUSE_SUBJECT_MISMATCH", "CAUSE_TARGET_MISMATCH"}:
            reason = str(cause.rejection_reason)
            status = SemanticStatus.INVALID.value
        else:
            reason = "CAUSE_RELATION_NOT_FOUND"
            status = SemanticStatus.UNKNOWN.value
        subject = score_subject_consistency(semantics, evidence, answer, context)
        return RelationValidation(
            status=status,
            relation_type="CAUSE",
            relation_score=0.0,
            relation_evidence=False,
            reason=reason,
            subject_match_score=max(subject.score, cause.subject_match_score),
            subject_match_reason=subject.reason,
            target_relation_score=cause.target_relation_score,
            cause_pattern_score=cause.cause_pattern_score,
            relation_method=cause.relation_method,
            evidence_sentence=evidence,
        )

    if semantics.relation in {"BIRTH_TIME", "DEATH_TIME"}:
        return _birth_or_death_time_validation(semantics, evidence, answer, context)

    if semantics.relation == "PURPOSE":
        subject = score_subject_consistency(semantics, evidence, answer, context)
        legacy_evidence = bool(details.get("relation_evidence", False))
        legacy_score = float(details.get("relation_score", 0.0))
        if subject.status == SemanticStatus.INVALID.value:
            return RelationValidation(
                SemanticStatus.INVALID.value,
                "PURPOSE",
                0.0,
                False,
                "PURPOSE_SUBJECT_MISMATCH",
                subject.score,
                subject.reason,
                evidence_sentence=evidence,
            )
        matched = bool(legacy_evidence and subject.status == SemanticStatus.VALID.value)
        return RelationValidation(
            SemanticStatus.VALID.value if matched else SemanticStatus.UNKNOWN.value,
            "PURPOSE",
            legacy_score if matched else 0.0,
            matched,
            "DIRECT_PURPOSE_PATTERN" if matched else "PURPOSE_RELATION_NOT_FOUND",
            subject.score,
            subject.reason,
            target_relation_score=legacy_score if matched else 0.0,
            relation_method=str(details.get("fallback_method") or "purpose_relation_pattern") if matched else None,
            evidence_sentence=evidence,
        )

    subject = score_subject_consistency(semantics, evidence, answer, context)
    return RelationValidation(
        SemanticStatus.UNKNOWN.value,
        semantics.relation,
        0.0,
        False,
        "RELATION_VALIDATOR_NOT_REQUIRED",
        subject.score,
        subject.reason,
        evidence_sentence=evidence,
    )


__all__ = [
    "RelationValidation",
    "STRICT_RELATIONS",
    "validate_candidate_relation",
]
