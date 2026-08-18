from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from reader.question_type import QuestionType


_WORD = r"[\wÀ-ỹĐđ'’-]+"
_LEADING_CONNECTORS = {"là", "và", "của", "thì", "mà"}
_TRAILING_CONNECTORS = {
    "là",
    "và",
    "của",
    "về",
    "khi",
    "tại",
    "ở",
    "trong",
    "đến",
    "từ",
}


@dataclass(frozen=True)
class BoundaryAssessment:
    score: float
    complete: bool
    reasons: tuple[str, ...]


def _words(text: str) -> list[str]:
    return re.findall(_WORD, str(text or ""), flags=re.UNICODE)


def _is_title_word(word: str) -> bool:
    return bool(word and word[0].isalpha() and word[0].isupper())


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text or "").casefold().replace("đ", "d"))
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def assess_span_boundary(
    context: str,
    start: int,
    end: int,
    question_types: list[QuestionType | str] | QuestionType | str,
    question: str = "",
) -> BoundaryAssessment:
    """Estimate whether a span starts and ends at a semantic phrase boundary.

    This does not expand an answer blindly. It detects high-confidence boundary
    defects—especially a span shifted inside a multi-token proper name—and is
    used to rerank several neural proposals from the same logits.
    """

    if start < 0 or end <= start or end > len(context):
        return BoundaryAssessment(0.0, False, ("INVALID_OFFSETS",))
    answer = context[start:end].strip()
    tokens = _words(answer)
    if not tokens:
        return BoundaryAssessment(0.0, False, ("EMPTY_SPAN",))

    if not isinstance(question_types, list):
        question_types = [question_types]
    expected_types = [QuestionType(qt) for qt in question_types]
    score = 1.0
    reasons: list[str] = []
    first = tokens[0].casefold()
    last = tokens[-1].casefold()

    if first in _LEADING_CONNECTORS:
        score -= 0.25
        reasons.append("LEADING_CONNECTOR")
    if last in _TRAILING_CONNECTORS:
        score -= 0.35
        reasons.append("TRAILING_CONNECTOR")
    if len(tokens) >= 2 and [token.casefold() for token in tokens[-2:]] == ["về", "mặt"]:
        score -= 0.45
        reasons.append("INCOMPLETE_VE_MAT_PHRASE")

    word_count = len(tokens)
    normalized_question = _normalize(question)
    if QuestionType.TIME in expected_types and word_count > 12:
        score = min(score, 0.15)
        reasons.append("OVERLONG_TIME_SPAN")
    elif QuestionType.NUMBER in expected_types and word_count > 10:
        score = min(score, 0.15)
        reasons.append("OVERLONG_NUMBER_SPAN")
    elif QuestionType.PERSON in expected_types:
        plural_person = bool(re.search(r"\b(?:nhung ai|nhung nguoi nao)\b", normalized_question))
        person_definition = bool(
            re.fullmatch(r".+\s+la\s+ai", normalized_question.strip(" ?!."))
        )
        maximum = 60 if person_definition else (32 if plural_person else 12)
        if word_count > maximum:
            score = min(score, 0.15)
            reasons.append("OVERLONG_PERSON_SPAN")
    elif QuestionType.LOCATION in expected_types and word_count > 16:
        score = min(score, 0.30)
        reasons.append("OVERLONG_LOCATION_SPAN")

    if any(qt in expected_types for qt in {QuestionType.ENTITY, QuestionType.PERSON, QuestionType.LOCATION}):
        right = re.match(rf"\s+(?P<word>{_WORD})", context[end:], flags=re.UNICODE)
        if right and _normalize(tokens[-1]) == "dau" and _normalize(right.group("word")) == "tien":
            score = min(score, 0.15)
            reasons.append("TRUNCATED_FIXED_PHRASE_RIGHT")
        if right and _is_title_word(tokens[-1]) and _is_title_word(right.group("word")):
            score = min(score, 0.15)
            reasons.append("TRUNCATED_NAMED_ENTITY_RIGHT")

        left = re.search(rf"(?P<word>{_WORD})\s+$", context[:start], flags=re.UNICODE)
        if left and _is_title_word(tokens[0]) and _is_title_word(left.group("word")):
            score = min(score, 0.15)
            reasons.append("TRUNCATED_NAMED_ENTITY_LEFT")

    score = round(max(0.0, min(1.0, score)), 6)
    return BoundaryAssessment(score, score >= 0.5, tuple(reasons) or ("CLEAN_BOUNDARY",))


__all__ = ["BoundaryAssessment", "assess_span_boundary"]
