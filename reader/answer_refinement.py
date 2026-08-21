from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from reader.question_type import QuestionType
from reader.answer_completeness import assess_answer_completeness, refine_dangling_clause


_WORD = r"[\wÀ-ỹĐđ'’-]+"
_LEADING_FACTOID_SCAFFOLD = re.compile(
    r"^(?:[,;:]\s*)?(?:(?:là|chính[\s_]+là)[\s_:,-]+)",
    flags=re.IGNORECASE | re.UNICODE,
)
_RELATIVE_CLAUSE = re.compile(
    r",\s*(?:được|đã|đang|sẽ|vốn|nơi|mà|do|khi|trong[\s_]+khi)\b",
    flags=re.IGNORECASE | re.UNICODE,
)
_LEFT_DESIGNATORS = (
    "vương cung thánh đường",
    "nhà thờ",
    "thành phố",
    "thủ đô",
    "tỉnh",
    "quận",
    "huyện",
    "xã",
    "đảo",
    "núi",
    "sông",
    "cầu",
    "bảo tàng",
    "công ty",
    "tổ chức",
    "trường đại học",
    "đại học",
    "Thiếu tướng",
    "Trung tướng",
    "Đại tướng",
    "Tổng thống",
    "Thủ tướng",
    "Chủ tịch",
    "Giáo sư",
    "Tiến sĩ",
)


class QuestionRelation(str, Enum):
    FACTOID = "FACTOID"
    CONTRAST = "CONTRAST"
    CAUSE = "CAUSE"
    PURPOSE = "PURPOSE"
    METHOD = "METHOD"
    DEFINITION = "DEFINITION"
    ATTRIBUTE = "ATTRIBUTE"


@dataclass(frozen=True)
class RefinementResult:
    raw_answer: str
    refined_answer: str
    refinement_method: str
    raw_start: int
    raw_end: int
    final_start: int
    final_end: int
    relation: str
    completeness_score: float
    completeness_before: float
    completeness_after: float
    relation_complete: bool
    completeness_reasons: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return (self.raw_start, self.raw_end) != (self.final_start, self.final_end)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["changed"] = self.changed
        return payload


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text or "").casefold().replace("đ", "d"))
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return " ".join(re.findall(r"[\w]+", value, flags=re.UNICODE))


def detect_question_relation(
    question: str,
    question_types: list[QuestionType | str] | QuestionType | str,
) -> QuestionRelation:
    normalized = _normalize(question)
    if not isinstance(question_types, list):
        question_types = [question_types]
    expected_types = [QuestionType(qt) for qt in question_types]
    if re.search(r"\b(?:khac nhau|su khac biet|khac biet giua|so voi)\b", normalized):
        return QuestionRelation.CONTRAST
    if re.search(
        r"\b(?:tai sao|vi sao|do dau|boi dau|do nguyen nhan nao|nguyen nhan|ly do|"
        r"dieu gi (?:khien|lam)|yeu to nao (?:khien|lam)|vi nguyen nhan gi)\b",
        normalized,
    ):
        return QuestionRelation.CAUSE
    if re.search(r"\b(?:de lam gi|nham muc dich|muc dich gi|voi muc dich gi)\b", normalized):
        return QuestionRelation.PURPOSE
    if re.search(r"\b(?:bang cach nao|theo cach nao|phuong phap nao)\b", normalized):
        return QuestionRelation.METHOD
    if re.search(r"\b(?:tinh chat|dac diem|tinh hinh|trang thai|vai tro|vi the)\b", normalized):
        return QuestionRelation.ATTRIBUTE
    if QuestionType.DEFINITION in expected_types or re.search(r"\b(?:la gi|la ai)\b", normalized):
        return QuestionRelation.DEFINITION
    return QuestionRelation.FACTOID


def _trim_outer_boundaries(context: str, start: int, end: int) -> tuple[int, int]:
    while start < end and (context[start].isspace() or context[start] in ",;:"):
        start += 1
    while end > start and (context[end - 1].isspace() or context[end - 1] in ",;:"):
        end -= 1
    return start, end


def _expand_left_designator(context: str, start: int, end: int) -> tuple[int, int]:
    prefix = context[:start]
    for phrase in sorted(_LEFT_DESIGNATORS, key=len, reverse=True):
        match = re.search(rf"(?P<phrase>{re.escape(phrase)})[\s_]+$", prefix, re.IGNORECASE)
        if match:
            return match.start("phrase"), end
    return start, end


def _expand_right_name_or_fixed_phrase(context: str, start: int, end: int) -> tuple[int, int]:
    answer_words = re.findall(_WORD, context[start:end], flags=re.UNICODE)
    if not answer_words:
        return start, end
    expansions = 0
    while expansions < 3:
        match = re.match(rf"(?P<space>\s+)(?P<word>{_WORD})", context[end:], flags=re.UNICODE)
        if not match:
            break
        next_word = match.group("word")
        last_word = answer_words[-1]
        fixed_pair = (_normalize(last_word), _normalize(next_word)) in {
            ("dau", "tien"),
        }
        proper_continuation = (
            last_word[:1].isupper()
            and next_word[:1].isupper()
        )
        if not (fixed_pair or proper_continuation):
            break
        end += match.end()
        answer_words.append(next_word)
        expansions += 1
    return start, end


def _expand_to_nlp_boundaries(context: str, start: int, end: int) -> tuple[int, int]:
    if start < 0 or end <= start or end > len(context):
        return start, end
    try:
        from underthesea import word_tokenize
        
        window_start = max(0, start - 50)
        window_end = min(len(context), end + 50)
        
        while window_start > 0 and context[window_start - 1].isalnum():
            window_start -= 1
        while window_end < len(context) and context[window_end].isalnum():
            window_end += 1
            
        window = context[window_start:window_end]
        words = word_tokenize(window)
        
        local_start = start - window_start
        local_end = end - window_start
        
        current = 0
        new_local_start = local_start
        new_local_end = local_end
        
        for w in words:
            pos = window.find(w, current)
            if pos == -1:
                break
            w_end = pos + len(w)
            
            if pos < local_start < w_end:
                new_local_start = pos
                
            if pos < local_end < w_end:
                new_local_end = w_end
                
            current = w_end
            if current >= max(local_end, new_local_end):
                break
                
        return window_start + new_local_start, window_start + new_local_end
    except Exception:
        return start, end


def assess_relation_completeness(
    question: str,
    question_types: list[QuestionType | str] | QuestionType | str,
    answer: str,
) -> tuple[float, bool, tuple[str, ...]]:
    relation = detect_question_relation(question, question_types)
    normalized = _normalize(answer)
    reasons: list[str] = []
    score = 1.0

    if relation is QuestionRelation.CONTRAST:
        has_two_sides = bool(
            re.search(r"\btrong khi\b", normalized)
            or re.search(r"\b(?:con|nhung)\b", normalized)
            or re.search(r"\bkhac biet giua\b.+\bva\b", normalized)
        )
        if not has_two_sides:
            score = 0.25
            reasons.append("MISSING_CONTRAST_SIDE")
    elif relation is QuestionRelation.CAUSE:
        if re.fullmatch(
            r"(?:vi )?(?:dieu nay|viec nay|su viec nay|nguyen nhan nay|ly do tren|do do)",
            normalized,
        ):
            score = 0.0
            reasons.append("ANAPHORIC_CAUSE_PHRASE")
        explicit = re.search(
            r"\b(?:vi|do|boi)\s+(?P<tail>.+)$",
            normalized,
        )
        incomplete_cause_nouns = {"xuat than", "anh huong", "tac dong", "nguyen nhan"}
        if score > 0 and explicit and (
            len(explicit.group("tail").split()) < 2
            or explicit.group("tail") in incomplete_cause_nouns
        ):
            score = 0.20
            reasons.append("INCOMPLETE_CAUSE_PHRASE")
        elif score > 0 and not re.search(r"\b(?:vi|do|boi|nguyen nhan|ly do|khien|dan den)\b", normalized):
            score = 0.70
            reasons.append("IMPLICIT_CAUSE_CUE")
    elif relation is QuestionRelation.PURPOSE:
        if not re.search(r"\b(?:de|nham|muc dich)\b", normalized):
            score = 0.65
            reasons.append("IMPLICIT_PURPOSE_CUE")
    elif relation is QuestionRelation.METHOD:
        if not re.search(r"\b(?:bang|thong qua|nho|theo cach|qua)\b", normalized):
            score = 0.65
            reasons.append("IMPLICIT_METHOD_CUE")

    tokens = normalized.split()
    delimiter_pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    if any(answer.count(left) != answer.count(right) for left, right in delimiter_pairs):
        score = min(score, 0.15)
        reasons.append("UNBALANCED_DELIMITER")
    if tokens and tokens[-1] in {"va", "cua", "ve", "tu", "den", "trong", "do", "tai"}:
        score = min(score, 0.20)
        reasons.append("DANGLING_TRAILING_CONNECTOR")
    score = round(max(0.0, min(1.0, score)), 6)
    return score, score >= 0.5, tuple(reasons) or ("RELATION_COMPLETE",)


def refine_answer(
    question: str,
    question_types: list[QuestionType | str] | QuestionType | str,
    context: str,
    start: int,
    end: int,
) -> RefinementResult:
    raw_start, raw_end = int(start), int(end)
    if raw_start < 0 or raw_end <= raw_start or raw_end > len(context):
        return RefinementResult(
            "", "", "INVALID_OFFSETS", raw_start, raw_end, raw_start, raw_end,
            detect_question_relation(question, question_types).value,
            0.0, 0.0, 0.0, False, ("INVALID_OFFSETS",),
        )

    raw_answer = context[raw_start:raw_end]
    raw_completeness = assess_answer_completeness(raw_answer)
    
    nlp_start, nlp_end = _expand_to_nlp_boundaries(context, raw_start, raw_end)
    final_start, final_end = _trim_outer_boundaries(context, nlp_start, nlp_end)
    
    methods: list[str] = []
    if (nlp_start, nlp_end) != (raw_start, raw_end):
        methods.append("nlp_boundary_expansion")
    if (final_start, final_end) != (nlp_start, nlp_end):
        methods.append("punctuation_cleanup")

    if not isinstance(question_types, list):
        question_types = [question_types]
    expected_types = [QuestionType(qt) for qt in question_types]
    relation = detect_question_relation(question, expected_types)
    factoid_types = {QuestionType.TIME, QuestionType.NUMBER, QuestionType.PERSON, QuestionType.LOCATION, QuestionType.ENTITY}
    if any(qt in factoid_types for qt in expected_types) and relation not in {QuestionRelation.DEFINITION, QuestionRelation.CONTRAST}:
        current = context[final_start:final_end]
        scaffold = _LEADING_FACTOID_SCAFFOLD.match(current)
        if scaffold:
            final_start += scaffold.end()
            methods.append("leading_scaffold_compression")

        current = context[final_start:final_end]
        relative = _RELATIVE_CLAUSE.search(current)
        if relative and relative.start() > 0:
            final_end = final_start + relative.start()
            methods.append("relative_clause_compression")

        expanded_start, expanded_end = _expand_left_designator(context, final_start, final_end)
        if expanded_start != final_start:
            final_start = expanded_start
            methods.append("noun_phrase_expand_left")
        expanded_start, expanded_end = _expand_right_name_or_fixed_phrase(
            context, final_start, final_end
        )
        if expanded_end != final_end:
            final_end = expanded_end
            methods.append("phrase_expand_right")

    final_start, final_end = _trim_outer_boundaries(context, final_start, final_end)
    clause_refinement = refine_dangling_clause(context, final_start, final_end)
    if clause_refinement.method != "UNCHANGED":
        final_start, final_end = clause_refinement.start, clause_refinement.end
        methods.append(clause_refinement.method)
    final_start, final_end = _trim_outer_boundaries(context, final_start, final_end)
    refined_answer = context[final_start:final_end]
    final_completeness = assess_answer_completeness(refined_answer)
    completeness_score, relation_complete, completeness_reasons = assess_relation_completeness(
        question, expected_types, refined_answer
    )
    completeness_score = min(completeness_score, final_completeness.score)
    relation_complete = bool(relation_complete and final_completeness.complete)
    if not final_completeness.complete:
        completeness_reasons = tuple(dict.fromkeys((*completeness_reasons, *final_completeness.reasons)))
    return RefinementResult(
        raw_answer=raw_answer,
        refined_answer=refined_answer,
        refinement_method="+".join(dict.fromkeys(methods)) if methods else "UNCHANGED",
        raw_start=raw_start,
        raw_end=raw_end,
        final_start=final_start,
        final_end=final_end,
        relation=relation.value,
        completeness_score=completeness_score,
        completeness_before=raw_completeness.score,
        completeness_after=final_completeness.score,
        relation_complete=relation_complete,
        completeness_reasons=completeness_reasons,
    )


__all__ = [
    "QuestionRelation",
    "RefinementResult",
    "assess_relation_completeness",
    "detect_question_relation",
    "refine_answer",
]
