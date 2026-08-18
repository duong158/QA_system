from __future__ import annotations

import re
from typing import Any

from backend.chunking import split_sentences
from reader.fallback_extractor import extract_fallback_answer
from reader.lexical import normalize_text, tokenize
from reader.question_type import QuestionType, detect_question_type
from reader.semantic_policy import SEMANTIC_POLICIES


ANSWER_CUE_PATTERNS = (
    "duoc chia thanh",
    "duoc chia lam",
    "chia thanh",
    "chia lam",
    "bao gom",
    "gom",
    "la",
    "duoc goi la",
    "co nghia la",
    "duoc dinh nghia",
)


def definition_subject(question: str) -> str:
    normalized = normalize_text(question).strip(" ?!.")
    match = re.match(
        r"^(?:cho biet\s+)?(.+?)\s+(?:la ai|la gi|co nghia la gi|duoc dinh nghia nhu the nao)$",
        normalized,
    )
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def relation_follows_subject(sentence: str, subject: str) -> bool:
    if not subject:
        return False
    normalized = normalize_text(sentence)
    match = re.search(rf"\b{re.escape(subject)}\b", normalized)
    if not match:
        return False
    tail = normalized[match.end() : match.end() + 180]
    tail = re.sub(r"^\s*\([^)]{0,120}\)", "", tail)
    return bool(
        re.match(
            r"^\s*,?\s*(?:(?:hien|tung|chinh|duoc xem la|duoc biet den nhu)\s+)?la\b",
            tail,
        )
    )


def sentence_fallback_predict(question: str, context: str) -> dict[str, Any]:
    """Select evidence and narrow it to a grounded fallback phrase."""

    question_tokens = set(tokenize(question))
    if not question_tokens:
        return {
            "answer": "",
            "confidence": 0.0,
            "start": -1,
            "end": -1,
            "reason": "empty_question_tokens",
        }

    normalized_question = normalize_text(question)
    question_types = detect_question_type(question)
    subject = definition_subject(question)
    subject_phrase = re.sub(
        r"\b(bao gom nhung gi|gom nhung gi|co nhung gi|duoc chia thanh|duoc chia lam|duoc chia|chia thanh|chia lam|nhu the nao|nhu nao|la gi|la ai|chia|duoc|nao|gi|ai)\b",
        " ",
        normalized_question,
    )
    subject_phrase = re.sub(r"\s+", " ", subject_phrase).strip()
    best: dict[str, Any] = {
        "answer": "",
        "confidence": 0.0,
        "start": -1,
        "end": -1,
        "reason": "no_sentence",
    }
    search_from = 0
    fallback_threshold = SEMANTIC_POLICIES.method("sentence_fallback").min_generation_score
    for sentence in split_sentences(context):
        sentence = sentence.strip()
        if not sentence:
            continue
        start = context.find(sentence, search_from)
        if start < 0:
            start = context.find(sentence)
        end = start + len(sentence) if start >= 0 else -1
        search_from = max(search_from, end)

        sentence_tokens = set(tokenize(sentence))
        overlap = len(question_tokens & sentence_tokens) / max(1, len(question_tokens))
        normalized_sentence = normalize_text(sentence)
        phrase = extract_fallback_answer(question, [qt.value for qt in question_types], sentence)
        cue_bonus = 0.0
        if any(pattern in normalized_sentence for pattern in ANSWER_CUE_PATTERNS):
            cue_bonus += 0.18
        method_bonuses = {
            "contrast_relation_pattern": 0.30,
            "role_relation_pattern": 0.40,
            "cause_clause_pattern": 0.32,
            "purpose_clause_pattern": 0.32,
            "method_clause_pattern": 0.32,
            "description_sentence_pattern": 0.24,
        }
        if phrase.relation_evidence:
            cue_bonus += method_bonuses.get(phrase.method, 0.0)
        if phrase.method == "property_description_pattern" and phrase.relation_evidence:
            cue_bonus += 0.60 if phrase.phrase_quality >= 0.95 else 0.10
        if len(subject_phrase) >= 8 and subject_phrase in normalized_sentence:
            cue_bonus += 0.22
        if len(sentence_tokens) >= 6:
            cue_bonus += 0.04
        if len(sentence) > 520:
            cue_bonus -= 0.12
        score = max(0.0, min(0.70, overlap * 0.78 + cue_bonus))
        if QuestionType.LOCATION in question_types:
            score = (
                min(0.70, score + 0.24 * phrase.relation_score)
                if phrase.relation_evidence
                else min(score, 0.68) if phrase.method == "whole_sentence" else score
            )
        relation_matched = relation_follows_subject(sentence, subject)
        if subject:
            score = (
                min(0.70, score + 0.18)
                if relation_matched
                else min(score, fallback_threshold - 0.07)
            )

        if score > best["confidence"]:
            best = {
                "answer": sentence,
                "confidence": round(score, 6),
                "start": start,
                "end": end,
                "reason": "definition_relation" if relation_matched else "sentence_overlap_cue",
                "_phrase": phrase,
            }
    if not best.get("answer"):
        return best

    supporting_sentence = str(best["answer"])
    supporting_start = int(best["start"])
    supporting_end = int(best["end"])
    phrase = best.pop("_phrase", None) or extract_fallback_answer(
        question, [qt.value for qt in question_types], supporting_sentence
    )
    if phrase.answer and phrase.start_char >= 0:
        best.update(
            {
                "answer": phrase.answer,
                "start": supporting_start + phrase.start_char,
                "end": supporting_start + phrase.end_char,
                "sentence_answer": supporting_sentence,
                "sentence_start": supporting_start,
                "sentence_end": supporting_end,
                "fallback_method": phrase.method,
                "phrase_score": phrase.score,
                "phrase_start": phrase.start_char,
                "phrase_end": phrase.end_char,
                "relation_type": phrase.relation_type,
                "relation_score": phrase.relation_score,
                "phrase_quality": phrase.phrase_quality,
                "lexical_evidence": phrase.lexical_evidence,
                "relation_evidence": phrase.relation_evidence,
                "relation_method": phrase.relation_method,
                "question_subject": phrase.question_subject,
                "question_target": phrase.question_target,
                "cause_pattern_score": phrase.cause_pattern_score,
                "subject_match_score": phrase.subject_match_score,
                "target_relation_score": phrase.target_relation_score,
                "relation_rejection_reason": phrase.relation_rejection_reason,
            }
        )
    return best


__all__ = ["definition_subject", "relation_follows_subject", "sentence_fallback_predict"]
