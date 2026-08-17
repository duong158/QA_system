from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum


class QuestionType(str, Enum):
    TIME = "TIME"
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    NUMBER = "NUMBER"
    DEFINITION = "DEFINITION"
    ENTITY = "ENTITY"
    GENERAL = "GENERAL"


@dataclass(frozen=True)
class AnswerTypeAssessment:
    expected_type: QuestionType
    score: float
    matched: bool
    reason: str


def _normalized(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text or "").casefold().replace("đ", "d"))
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return " ".join(re.findall(r"[\w%]+", value, flags=re.UNICODE))


TIME_QUESTION_PATTERNS = (
    r"\bkhi nao\b",
    r"\bbao gio\b",
    r"\b(?:nam|thang|ngay|the ky) nao\b",
    r"\btu (?:nam|thang|ngay|the ky) nao\b",
    r"\bvao (?:nam|thang|ngay|the ky) nao\b",
    r"\bthoi gian nao\b",
)
PERSON_QUESTION_PATTERNS = (r"\bai\b", r"\bnguoi nao\b", r"\bnhan vat nao\b")
LOCATION_QUESTION_PATTERNS = (
    r"\bo dau\b",
    r"\btai dau\b",
    r"\bnoi nao\b",
    r"\bdia diem nao\b",
    r"\bdia danh nao\b",
    r"\bkhu vuc nao\b",
    r"\b(?:thanh pho|tinh|quoc gia|nuoc) nao\b",
)
NUMBER_QUESTION_PATTERNS = (
    r"\bbao nhieu\b",
    r"\bmay\b",
    r"\bso luong\b.{0,40}\bbao nhieu\b",
    r"\bty le bao nhieu\b",
)
DEFINITION_QUESTION_PATTERNS = (
    r"\bla gi\b",
    r"\bco nghia la gi\b",
    r"\bdinh nghia\b",
    r"\bduoc hieu nhu the nao\b",
)
ENTITY_QUESTION_PATTERNS = (
    r"\bcai gi\b",
    r"\bdieu gi\b",
    r"\bcong trinh nao\b",
    r"\btac pham nao\b",
    r"\bto chuc nao\b",
    r"\bten goi nao\b",
    r"\bbi danh nao\b",
    r"\bten nao\b",
    r"\bloai nao\b",
    r"\bchia (?:nhu nao|nhu the nao|thanh gi|lam gi)\b",
    r"\b(?:bao gom|gom) nhung gi\b",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def detect_question_type(question: str) -> QuestionType:
    """Detect the expected answer category without extracting an answer."""

    normalized = _normalized(question)
    # TIME must precede NUMBER because years and centuries are numeric answers.
    if _matches_any(normalized, TIME_QUESTION_PATTERNS):
        return QuestionType.TIME
    if _matches_any(normalized, PERSON_QUESTION_PATTERNS):
        return QuestionType.PERSON
    if _matches_any(normalized, LOCATION_QUESTION_PATTERNS):
        return QuestionType.LOCATION
    if _matches_any(normalized, NUMBER_QUESTION_PATTERNS):
        return QuestionType.NUMBER
    if _matches_any(normalized, DEFINITION_QUESTION_PATTERNS):
        return QuestionType.DEFINITION
    if _matches_any(normalized, ENTITY_QUESTION_PATTERNS):
        return QuestionType.ENTITY
    return QuestionType.GENERAL


ROMAN = r"(?:xxi|xx|xix|xviii|xvii|xvi|xv|xiv|xiii|xii|xi|ix|viii|vii|vi|iv|iii|ii|i)"
VIETNAMESE_ORDINAL = (
    r"(?:mot|hai|ba|bon|tu|nam|sau|bay|tam|chin|muoi"
    r"|muoi\s+(?:mot|hai|ba|bon|tu|nam|sau|bay|tam|chin)"
    r"|hai\s+muoi(?:\s+mot)?)"
)
TIME_PATTERNS = (
    rf"\bthe ky\s+(?:thu\s+)?(?:\d{{1,2}}|{ROMAN}|{VIETNAMESE_ORDINAL})\b",
    r"\b(?:nam|tu nam|vao nam)\s+(?:1\d{3}|20\d{2})\b",
    r"\b(?:1\d{3}|20\d{2})\s*(?:-|–|den|toi)\s*(?:1\d{3}|20\d{2})\b",
    r"\bngay\s+\d{1,2}(?:\s+thang\s+\d{1,2})?(?:\s+nam\s+\d{3,4})?\b",
    r"\bthang\s+\d{1,2}(?:\s+nam\s+\d{3,4})?\b",
    r"\b(?:dau|giua|cuoi)\s+(?:nhung\s+nam|the ky|thap nien)\b",
    r"\b(?:thoi ky|giai doan|thap nien)\b",
)
LOCATION_TERMS = {
    "thanh pho", "tinh", "huyen", "xa", "quan", "quoc gia", "nuoc", "dao",
    "chau", "mien", "khu vuc", "thu do", "lanh tho", "thi tran", "phuong",
}
LOCATION_EVENT_TERMS = {
    "cach mang", "chien tranh", "cuoc chien", "tran", "phong trao", "khoi nghia",
    "hoi nghi", "hiep dinh", "le hoi", "su kien",
}
LOCATION_ORGANIZATION_TERMS = {
    "dai hoc", "truong dai hoc", "cong ty", "tap doan", "to chuc", "hoc vien",
    "benh vien", "vien nghien cuu",
}
PERSON_TERMS = {
    "ong", "ba", "vua", "nu hoang", "tong thong", "thu tuong", "giao su", "tac gia",
    "nha van", "nha tho", "nhac si", "tuong", "bac si",
}
NUMBER_WORDS = {
    "khong", "mot", "hai", "ba", "bon", "tu", "nam", "sau", "bay", "tam", "chin", "muoi",
    "tram", "nghin", "trieu", "ty",
}
ENTITY_DESIGNATOR_TERMS = {
    "bao tang", "benh vien", "cau", "chua", "cong ty", "cong trinh", "cung dien",
    "den", "dia danh", "hoc vien", "lau dai", "nha hat", "nha may", "nha tho", "nui",
    "quan", "san van dong", "song", "tac pham", "tap doan", "thanh pho", "thap",
    "thu vien", "tinh", "to chuc", "truong", "vuong cung thanh duong",
}
ENTITY_CLAUSE_TERMS = {
    "co", "da", "dang", "duoc", "gom", "la", "lam", "mang", "nam", "tro thanh",
}


def _has_title_case_phrase(text: str) -> bool:
    words = re.findall(r"\b[A-ZÀ-ỸĐ][\wÀ-ỹĐđ'-]*\b", str(text or ""), flags=re.UNICODE)
    return len(words) >= 2 or (len(words) == 1 and len(str(text or "").split()) <= 4)


def assess_answer_type(
    expected_type: QuestionType | str,
    answer: str,
    *,
    relation_score: float | None = None,
    phrase_quality: float | None = None,
    candidate_method: str | None = None,
) -> AnswerTypeAssessment:
    """Return a soft answer-type compatibility score in the range [0, 1].

    The function validates a Reader-produced candidate; it never extracts an
    answer from the passage.
    """

    expected = QuestionType(expected_type)
    raw = str(answer or "").strip()
    normalized = _normalized(raw)
    if not normalized:
        return AnswerTypeAssessment(expected, 0.0, False, "EMPTY_CANDIDATE")

    if expected is QuestionType.TIME:
        if _matches_any(normalized, TIME_PATTERNS):
            return AnswerTypeAssessment(expected, 1.0, True, "TEMPORAL_EXPRESSION")
        if re.search(r"\b(?:1\d{3}|20\d{2})\b", normalized):
            return AnswerTypeAssessment(expected, 0.9, True, "YEAR_EXPRESSION")
        return AnswerTypeAssessment(expected, 0.0, False, "NO_TEMPORAL_EXPRESSION")

    if expected is QuestionType.NUMBER:
        if re.search(r"\b\d+(?:[.,]\d+)?\s*%?\b", normalized):
            return AnswerTypeAssessment(expected, 1.0, True, "NUMERIC_EXPRESSION")
        tokens = set(normalized.split())
        if tokens & NUMBER_WORDS:
            return AnswerTypeAssessment(expected, 0.8, True, "NUMBER_WORD")
        if re.search(r"\b(?:nhieu|mot so|hang tram|hang nghin)\b", normalized):
            return AnswerTypeAssessment(expected, 0.5, True, "IMPRECISE_QUANTITY")
        return AnswerTypeAssessment(expected, 0.0, False, "NO_NUMERIC_EXPRESSION")

    if expected is QuestionType.LOCATION:
        word_count = len(normalized.split())
        has_event_shape = any(
            re.search(rf"\b{re.escape(term)}\b", normalized)
            for term in LOCATION_EVENT_TERMS
        )
        has_organization_shape = any(
            re.search(rf"\b{re.escape(term)}\b", normalized)
            for term in LOCATION_ORGANIZATION_TERMS
        )
        has_location_cue = any(
            re.search(rf"\b{re.escape(term)}\b", normalized) for term in LOCATION_TERMS
        )
        has_name_shape = _has_title_case_phrase(raw)

        if has_event_shape:
            base_score = 0.15
            reason = "EVENT_PHRASE_NOT_LOCATION"
        elif has_organization_shape:
            base_score = 0.25
            reason = "ORGANIZATION_PHRASE_NOT_LOCATION"
        elif has_location_cue and word_count <= 12:
            base_score = 0.90
            reason = "LOCATION_DESIGNATOR_PHRASE"
        elif has_location_cue:
            base_score = 0.45
            reason = "LOCATION_CUE_IN_BROAD_CLAUSE"
        elif has_name_shape and word_count == 1:
            base_score = 0.82
            reason = "CONCISE_NAMED_LOCATION_CANDIDATE"
        elif has_name_shape and word_count <= 6:
            base_score = 0.78
            reason = "NAMED_LOCATION_CANDIDATE"
        elif has_name_shape:
            base_score = 0.40
            reason = "NAMED_ENTITY_IN_BROAD_CLAUSE"
        else:
            base_score = 0.30
            reason = "WEAK_LOCATION_EVIDENCE"

        if candidate_method == "whole_sentence":
            base_score = min(base_score, 0.45)
            reason = "WHOLE_SENTENCE_LOCATION_CANDIDATE"
        if relation_score is not None:
            relation = max(0.0, min(1.0, float(relation_score)))
            quality = max(0.0, min(1.0, float(phrase_quality or 0.0)))
            score = 0.55 * base_score + 0.35 * relation + 0.10 * quality
        else:
            score = base_score
        score = round(max(0.0, min(1.0, score)), 6)
        return AnswerTypeAssessment(expected, score, score >= 0.5, reason)

    if expected is QuestionType.PERSON:
        if any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in PERSON_TERMS):
            return AnswerTypeAssessment(expected, 1.0, True, "PERSON_CUE")
        if _has_title_case_phrase(raw):
            return AnswerTypeAssessment(expected, 0.85, True, "NAMED_PERSON_CANDIDATE")
        # Collective descriptions can be valid answers to "ai" but are less certain.
        if 1 <= len(normalized.split()) <= 12:
            return AnswerTypeAssessment(expected, 0.55, True, "PERSON_NOUN_PHRASE")
        return AnswerTypeAssessment(expected, 0.25, False, "WEAK_PERSON_EVIDENCE")

    if expected is QuestionType.DEFINITION:
        score = 0.85 if len(normalized.split()) >= 3 else 0.6
        return AnswerTypeAssessment(expected, score, True, "DEFINITION_TEXT")

    if expected is QuestionType.ENTITY:
        if re.search(r"\b(?:chia thanh|chia lam|bao gom|gom)\b", normalized):
            return AnswerTypeAssessment(expected, 0.9, True, "ENTITY_RELATION_CUE")
        word_count = len(normalized.split())
        has_designator = any(
            re.search(rf"\b{re.escape(term)}\b", normalized)
            for term in ENTITY_DESIGNATOR_TERMS
        )
        has_name_shape = _has_title_case_phrase(raw)
        if word_count <= 12:
            score = 0.78
        elif word_count <= 20:
            score = 0.70
        else:
            score = 0.55
        if has_designator:
            score += 0.12
        if has_name_shape:
            score += 0.10
        clause_hits = sum(
            1 for term in ENTITY_CLAUSE_TERMS if re.search(rf"\b{re.escape(term)}\b", normalized)
        )
        if word_count > 10 and clause_hits >= 2:
            score -= min(0.18, clause_hits * 0.06)
        score = round(max(0.0, min(1.0, score)), 6)
        if has_designator and has_name_shape:
            reason = "ENTITY_DESIGNATED_NAMED_PHRASE"
        elif has_name_shape:
            reason = "ENTITY_NAMED_PHRASE"
        elif has_designator:
            reason = "ENTITY_DESIGNATED_PHRASE"
        elif word_count <= 20:
            reason = "ENTITY_NOUN_PHRASE"
        else:
            reason = "ENTITY_CLAUSE_LIKE"
        return AnswerTypeAssessment(expected, score, score >= 0.5, reason)

    return AnswerTypeAssessment(expected, 0.5, True, "GENERAL_NEUTRAL")


__all__ = [
    "AnswerTypeAssessment",
    "QuestionType",
    "assess_answer_type",
    "detect_question_type",
]
