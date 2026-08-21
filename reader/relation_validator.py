from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Callable

from reader.cause_relations import assess_cause_candidate, fold_text
from reader.fallback_extractor import (
    assess_contrast_relation,
    detect_location_relation,
    extract_alias_candidate,
    extract_location_candidate,
)
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
    subject_status: str = SemanticStatus.UNKNOWN.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STRICT_RELATIONS = {
    "CAUSE",
    "BIRTH_TIME",
    "DEATH_TIME",
    "EVENT_TIME",
    "EVENT_LOCATION",
    "OBJECT_LOCATION",
    "PROCESS_LOCATION",
    "BIRTH_LOCATION",
    "DEATH_LOCATION",
    "PURPOSE",
}


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
        subject_status=subject.status,
    )


def _validate_cause(
    semantics: QuestionSemantics,
    question: str,
    context: str,
    answer: str,
    start: int,
    end: int,
    candidate_method: str,
    candidate_details: dict[str, Any],
) -> RelationValidation:
    details = candidate_details
    evidence = _evidence_sentence(context, answer, start, end, details)
    cause = assess_cause_candidate(question, context, answer, evidence or None)
    subject = score_subject_consistency(semantics, evidence, answer, context)
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
            subject_status=SemanticStatus.VALID.value,
        )
    if subject.status == SemanticStatus.INVALID.value and subject.score == 0.0:
        reason = "CAUSE_SUBJECT_MISMATCH"
        status = SemanticStatus.INVALID.value
    elif _effect_repetition(semantics, answer):
        reason = "CAUSE_EFFECT_REPETITION"
        status = SemanticStatus.INVALID.value
    elif cause.rejection_reason in {"CAUSE_SUBJECT_MISMATCH", "CAUSE_TARGET_MISMATCH"}:
        reason = str(cause.rejection_reason)
        status = SemanticStatus.INVALID.value
    else:
        reason = "CAUSE_RELATION_NOT_FOUND"
        status = SemanticStatus.UNKNOWN.value
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
        subject_status=subject.status,
    )


def _validate_time(
    semantics: QuestionSemantics,
    question: str,
    context: str,
    answer: str,
    start: int,
    end: int,
    candidate_method: str,
    candidate_details: dict[str, Any],
) -> RelationValidation:
    evidence = _evidence_sentence(context, answer, start, end, candidate_details)
    return _birth_or_death_time_validation(semantics, evidence, answer, context)


def _validate_event_time(
    semantics: QuestionSemantics,
    question: str,
    context: str,
    answer: str,
    start: int,
    end: int,
    candidate_method: str,
    candidate_details: dict[str, Any],
) -> RelationValidation:
    evidence = _evidence_sentence(context, answer, start, end, candidate_details)
    subject = score_subject_consistency(semantics, evidence, answer, context)
    legacy_evidence = bool(candidate_details.get("relation_evidence", False))
    legacy_score = float(candidate_details.get("relation_score", 0.0))
    answer_is_time = bool(re.search(r"\b(?:nam|thang|ngay|the ky)\b|\b\d{3,4}\b", fold_text(answer)))
    matched = bool(answer_is_time and subject.status == SemanticStatus.VALID.value and (
        legacy_evidence or fold_text(answer).strip() in fold_text(evidence)
    ))
    score = legacy_score if legacy_evidence else (0.90 if matched else 0.0)
    reason = "DIRECT_TIME_RELATION" if matched else "EVENT_TIME_RELATION_MISMATCH"
    if subject.status == SemanticStatus.INVALID.value:
        reason = "TIME_SUBJECT_MISMATCH"
    return RelationValidation(
        SemanticStatus.VALID.value if matched else SemanticStatus.INVALID.value,
        semantics.relation,
        score,
        matched,
        reason,
        subject.score,
        subject.reason,
        target_relation_score=score,
        relation_method=str(candidate_details.get("fallback_method") or "time_relation_pattern") if matched else None,
        evidence_sentence=evidence,
        subject_status=subject.status,
    )


def _validate_purpose(
    semantics: QuestionSemantics,
    question: str,
    context: str,
    answer: str,
    start: int,
    end: int,
    candidate_method: str,
    candidate_details: dict[str, Any],
) -> RelationValidation:
    evidence = _evidence_sentence(context, answer, start, end, candidate_details)
    subject = score_subject_consistency(semantics, evidence, answer, context)
    legacy_evidence = bool(candidate_details.get("relation_evidence", False))
    legacy_score = float(candidate_details.get("relation_score", 0.0))
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
            subject_status=subject.status,
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
        relation_method=str(candidate_details.get("fallback_method") or "purpose_relation_pattern") if matched else None,
        evidence_sentence=evidence,
        subject_status=subject.status,
    )


def _validate_location(
    semantics: QuestionSemantics,
    question: str,
    context: str,
    answer: str,
    start: int,
    end: int,
    candidate_method: str,
    candidate_details: dict[str, Any],
) -> RelationValidation:
    evidence = _evidence_sentence(context, answer, start, end, candidate_details)
    subject = score_subject_consistency(semantics, evidence, answer, context)
    relation_type = str(candidate_details.get("relation_type") or detect_location_relation(question))
    legacy_score = float(candidate_details.get("relation_score", 0.0))
    legacy_evidence = bool(candidate_details.get("relation_evidence", False))
    extracted = extract_location_candidate(question, evidence, relation_type)
    normalized_answer = fold_text(answer).strip()
    normalized_extracted = fold_text(extracted.answer).strip()
    same_phrase = bool(normalized_answer and normalized_extracted) and (
        normalized_answer in normalized_extracted or normalized_extracted in normalized_answer
    )
    extracted_evidence = bool(extracted.relation_evidence and same_phrase)
    matched = bool(
        subject.status == SemanticStatus.VALID.value
        and (legacy_evidence or extracted_evidence)
    )
    score = max(legacy_score if legacy_evidence else 0.0, extracted.relation_score if extracted_evidence else 0.0)
    reason = "DIRECT_LOCATION_RELATION" if matched else "LOCATION_RELATION_MISMATCH"
    if subject.status == SemanticStatus.INVALID.value:
        reason = "LOCATION_SUBJECT_MISMATCH"
    return RelationValidation(
        SemanticStatus.VALID.value if matched else SemanticStatus.INVALID.value,
        semantics.relation,
        score,
        matched,
        reason,
        subject.score,
        subject.reason,
        target_relation_score=score,
        relation_method=str(candidate_details.get("fallback_method") or "location_relation_pattern") if matched else None,
        evidence_sentence=evidence,
        subject_status=subject.status,
    )


def _validate_contrast(
    semantics: QuestionSemantics,
    question: str,
    context: str,
    answer: str,
    start: int,
    end: int,
    candidate_method: str,
    candidate_details: dict[str, Any],
) -> RelationValidation:
    evidence = _evidence_sentence(context, answer, start, end, candidate_details)
    score, matched = assess_contrast_relation(answer)
    subject = score_subject_consistency(semantics, evidence, answer, context)
    return RelationValidation(
        SemanticStatus.VALID.value if matched else SemanticStatus.INVALID.value,
        semantics.relation,
        score,
        matched,
        "DIRECT_CONTRAST_PATTERN" if matched else "RELATION_MISMATCH",
        subject.score,
        subject.reason,
        relation_method="contrast_relation_pattern" if matched else None,
        evidence_sentence=evidence,
        subject_status=subject.status,
    )


def _validate_definition(
    semantics: QuestionSemantics,
    question: str,
    context: str,
    answer: str,
    start: int,
    end: int,
    candidate_method: str,
    candidate_details: dict[str, Any],
) -> RelationValidation:
    evidence = _evidence_sentence(context, answer, start, end, candidate_details)
    subject = score_subject_consistency(semantics, evidence, answer, context)
    folded = fold_text(evidence)
    subject_text = fold_text(semantics.subject or "").strip()
    answer_text = fold_text(answer).strip()
    cue = re.search(r"\b(?:la|duoc goi la|co nghia la|duoc dinh nghia)\b", folded)
    legacy_evidence = bool(candidate_details.get("relation_evidence", False))
    matched = bool(legacy_evidence or (subject_text and answer_text and cue and subject_text in folded and answer_text in folded))
    score = float(candidate_details.get("relation_score", 1.0 if matched else 0.0))
    return RelationValidation(
        SemanticStatus.VALID.value if matched else SemanticStatus.INVALID.value,
        semantics.relation,
        score,
        matched,
        "DIRECT_DEFINITION_RELATION" if matched else "RELATION_MISMATCH",
        subject.score,
        subject.reason,
        relation_method=str(candidate_details.get("fallback_method") or "definition_relation_pattern") if matched else None,
        evidence_sentence=evidence,
        subject_status=subject.status,
    )


def _validate_identity(
    semantics: QuestionSemantics,
    question: str,
    context: str,
    answer: str,
    start: int,
    end: int,
    candidate_method: str,
    candidate_details: dict[str, Any],
) -> RelationValidation:
    evidence = _evidence_sentence(context, answer, start, end, candidate_details)
    extracted = extract_alias_candidate(question, evidence)
    expected = fold_text(extracted.answer).strip() if extracted else ""
    proposed = re.sub(r"^la\s+", "", fold_text(answer)).strip()
    exact = bool(expected and proposed == expected)
    legacy_evidence = bool(candidate_details.get("relation_evidence", False))
    matched = exact or legacy_evidence
    score = 1.0 if exact else float(candidate_details.get("relation_score", 0.0))
    subject = score_subject_consistency(semantics, evidence, answer, context)
    return RelationValidation(
        SemanticStatus.VALID.value if matched else SemanticStatus.INVALID.value,
        "ALIAS",
        score,
        matched,
        "DIRECT_IDENTITY_RELATION" if matched else "RELATION_MISMATCH",
        subject.score,
        subject.reason,
        relation_method=str(candidate_details.get("fallback_method") or "alias_relation_pattern") if matched else None,
        evidence_sentence=evidence,
        subject_status=subject.status,
    )


def _validate_not_required(
    semantics: QuestionSemantics,
    question: str,
    context: str,
    answer: str,
    start: int,
    end: int,
    candidate_method: str,
    candidate_details: dict[str, Any],
) -> RelationValidation:
    evidence = _evidence_sentence(context, answer, start, end, candidate_details)
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
        subject_status=subject.status,
    )


def _validate_unsupported(
    semantics: QuestionSemantics,
    question: str,
    context: str,
    answer: str,
    start: int,
    end: int,
    candidate_method: str,
    candidate_details: dict[str, Any],
) -> RelationValidation:
    evidence = _evidence_sentence(context, answer, start, end, candidate_details)
    subject = score_subject_consistency(semantics, evidence, answer, context)
    return RelationValidation(
        SemanticStatus.UNKNOWN.value,
        semantics.relation,
        0.0,
        False,
        "RELATION_UNSUPPORTED",
        subject.score,
        subject.reason,
        evidence_sentence=evidence,
        subject_status=subject.status,
    )


RelationHandler = Callable[
    [QuestionSemantics, str, str, str, int, int, str, dict[str, Any]],
    RelationValidation,
]


RELATION_HANDLERS: dict[str, RelationHandler] = {
    "CAUSE": _validate_cause,
    "BIRTH_TIME": _validate_time,
    "DEATH_TIME": _validate_time,
    "EVENT_TIME": _validate_event_time,
    "EVENT_LOCATION": _validate_location,
    "OBJECT_LOCATION": _validate_location,
    "PROCESS_LOCATION": _validate_location,
    "BIRTH_LOCATION": _validate_location,
    "DEATH_LOCATION": _validate_location,
    "PURPOSE": _validate_purpose,
    "CONTRAST": _validate_contrast,
    "DEFINITION": _validate_definition,
    "IDENTITY": _validate_identity,
    "ATTRIBUTE": _validate_not_required,
    "GENERIC_LOCATION": _validate_not_required,
    "GENERAL": _validate_not_required,
}


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
    handler = RELATION_HANDLERS.get(semantics.relation, _validate_unsupported)
    return handler(
        semantics,
        question,
        context,
        answer,
        start,
        end,
        candidate_method,
        candidate_details or {},
    )


def relation_validation_details(validation: RelationValidation) -> dict[str, Any]:
    return {
        "relation_type": validation.relation_type,
        "relation_score": validation.relation_score,
        "phrase_quality": max(validation.cause_pattern_score, validation.target_relation_score),
        "relation_evidence": validation.relation_evidence,
        "relation_method": validation.relation_method,
        "cause_pattern_score": validation.cause_pattern_score,
        "subject_match_score": validation.subject_match_score,
        "target_relation_score": validation.target_relation_score,
        "relation_rejection_reason": validation.reason,
        "semantic_status": validation.status,
        "subject_match_reason": validation.subject_match_reason,
        "evidence_sentence": validation.evidence_sentence,
    }

__all__ = [
    "RELATION_HANDLERS",
    "RelationValidation",
    "STRICT_RELATIONS",
    "relation_validation_details",
    "validate_candidate_relation",
]
