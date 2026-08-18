from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from reader.answer_refinement import QuestionRelation, detect_question_relation
from reader.cause_relations import fold_text
from reader.question_type import QuestionType, detect_question_type


@dataclass(frozen=True)
class QuestionSemantics:
    question_type: str
    relation: str
    subject: str | None
    predicate: str | None
    target: str | None
    modifier: str | None
    expected_answer_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CAUSE_PREFIXES = (
    re.compile(r"^(?:vi sao|tai sao|do dau|boi dau)\s+(?P<body>.+?)\s*[?!.]*$"),
    re.compile(
        r"^(?:do nguyen nhan nao|nguyen nhan(?:\s+[\w]+){0,3}?\s+nao|"
        r"nguyen nhan gi|ly do gi|ly do nao|vi nguyen nhan gi)\s+"
        r"(?:(?:khien|lam)\s+)?(?P<body>.+?)\s*[?!.]*$"
    ),
    re.compile(r"^(?:dieu gi|yeu to nao)\s+(?:khien|lam)\s+(?P<body>.+?)\s*[?!.]*$"),
)
_CAUSE_SUFFIX = re.compile(
    r"^(?P<body>.+?)\s+(?:vi sao|tai sao|vi ly do gi|do nguyen nhan gi|"
    r"vi nguyen nhan gi)\s*[?!.]*$"
)

_PREDICATE_PATTERNS = (
    r"phat trien manh",
    r"phat trien",
    r"bi khinh miet",
    r"bi khinh thuong",
    r"bi coi thuong",
    r"bi khinh bi",
    r"duoc xay dung",
    r"duoc thanh lap",
    r"duoc sinh ra",
    r"duoc gui",
    r"bi bat",
    r"bi truc xuat",
    r"bi danh bai",
    r"sinh",
    r"ra doi",
    r"qua doi",
    r"mat",
    r"dien ra",
    r"xay ra",
    r"no ra",
    r"toa lac",
    r"nam",
    r"gui",
    r"tro thanh",
    r"co",
    r"la",
    r"bi",
    r"duoc",
)
_PREDICATE_RE = re.compile(
    r"\b(?P<predicate>" + "|".join(_PREDICATE_PATTERNS) + r")\b"
)
_MODIFIER_RE = re.compile(r"\s+\b(?:tai|o|tren|trong|vao|tu|den)\b\s+")


def _clean_original_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and (text[start].isspace() or text[start] in ",;:-"):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in "?!. ,;:"):
        end -= 1
    return start, end


def _surface(text: str, start: int, end: int) -> str | None:
    start, end = _clean_original_span(text, start, end)
    value = text[start:end].strip()
    return value or None


def _cause_body(question: str) -> str | None:
    folded = fold_text(question).strip()
    for pattern in _CAUSE_PREFIXES:
        match = pattern.match(folded)
        if match:
            return _surface(question, *match.span("body"))
    match = _CAUSE_SUFFIX.match(folded)
    return _surface(question, *match.span("body")) if match else None


def _strip_interrogative_tail(body: str, relation: str) -> str:
    folded = fold_text(body)
    patterns = {
        "BIRTH_TIME": (
            r"\s+(?:vao\s+)?nam\s+(?:nao|bao nhieu)\s*[?!.]*$",
            r"\s+bao\s+nhieu\s*[?!.]*$",
        ),
        "DEATH_TIME": (
            r"\s+(?:vao\s+)?nam\s+(?:nao|bao nhieu)\s*[?!.]*$",
        ),
        "EVENT_TIME": (
            r"\s+(?:khi nao|bao gio|vao nam nao|nam nao)\s*[?!.]*$",
        ),
        "PURPOSE": (
            r"\s+(?:voi muc dich gi|nham muc dich gi|de lam gi)\s*[?!.]*$",
        ),
        "EVENT_LOCATION": (
            r"\s+(?:o dau|tai dau|noi nao|dia diem nao)\s*[?!.]*$",
        ),
        "OBJECT_LOCATION": (
            r"\s+(?:o dau|tai dau|noi nao|dia diem nao)\s*[?!.]*$",
        ),
    }
    end = len(body)
    for pattern in patterns.get(relation, ()):
        match = re.search(pattern, folded)
        if match:
            end = min(end, match.start())
    return body[:end].strip(" ?!.,;:")


def _split_subject_predicate(body: str) -> tuple[str | None, str | None, str | None]:
    folded = fold_text(body)
    match = _PREDICATE_RE.search(folded)
    if not match or match.start() <= 0:
        return body.strip() or None, None, None

    subject = _surface(body, 0, match.start())
    predicate_start = match.start("predicate")
    modifier_match = _MODIFIER_RE.search(folded, match.end("predicate"))
    predicate_end = modifier_match.start() if modifier_match else len(body)
    modifier_start = modifier_match.start() if modifier_match else len(body)
    predicate = _surface(body, predicate_start, predicate_end)
    modifier = _surface(body, modifier_start, len(body)) if modifier_match else None
    return subject, predicate, modifier


def _time_relation(folded: str) -> str:
    if re.search(r"\b(?:nam sinh|sinh(?: vao)? nam|sinh nam|ra doi)\b", folded):
        return "BIRTH_TIME"
    if re.search(r"\b(?:mat|qua doi|tu tran)\b", folded):
        return "DEATH_TIME"
    return "EVENT_TIME"


def _location_relation(folded: str) -> str:
    if re.search(r"\b(?:sinh|ra doi)\b", folded):
        return "BIRTH_LOCATION"
    if re.search(r"\b(?:mat|qua doi|tu tran)\b", folded):
        return "DEATH_LOCATION"
    if re.search(r"\b(?:dien ra|xay ra|no ra|to chuc)\b", folded):
        return "EVENT_LOCATION"
    return "OBJECT_LOCATION"


def _subject_from_nominal_time_question(question: str, folded: str) -> str | None:
    match = re.match(
        r"^(?:nam sinh|ngay sinh)\s+cua\s+(?P<subject>.+?)\s+la\s+(?:bao nhieu|gi|ngay nao|nam nao)",
        folded,
    )
    return _surface(question, *match.span("subject")) if match else None


def parse_question_semantics(question: str) -> QuestionSemantics:
    question = str(question or "").strip()
    question_types = detect_question_type(question)
    question_type = question_types[0] if question_types else QuestionType.GENERAL
    folded = fold_text(question).strip()
    coarse_relation = detect_question_relation(question, question_type)
    birth_time_intent = bool(
        re.search(r"\b(?:nam sinh cua|ngay sinh cua|sinh(?: vao)? nam)\b", folded)
    )

    if coarse_relation is QuestionRelation.CAUSE:
        relation = "CAUSE"
        body = _cause_body(question) or question.strip(" ?!.")
    elif question_type is QuestionType.TIME or birth_time_intent:
        relation = "BIRTH_TIME" if birth_time_intent else _time_relation(folded)
        question_type = QuestionType.TIME
        nominal_subject = _subject_from_nominal_time_question(question, folded)
        if nominal_subject:
            return QuestionSemantics(
                question_type=question_type.value,
                relation=relation,
                subject=nominal_subject,
                predicate="sinh" if relation == "BIRTH_TIME" else None,
                target=None,
                modifier=None,
                expected_answer_type=QuestionType.TIME.value,
            )
        body = _strip_interrogative_tail(question, relation)
    elif question_type is QuestionType.LOCATION:
        relation = _location_relation(folded)
        body = _strip_interrogative_tail(question, relation)
    elif coarse_relation is QuestionRelation.PURPOSE:
        relation = "PURPOSE"
        body = _strip_interrogative_tail(question, relation)
    elif coarse_relation is QuestionRelation.CONTRAST:
        relation = "CONTRAST"
        body = question.strip(" ?!.")
    elif coarse_relation is QuestionRelation.ATTRIBUTE:
        relation = "ATTRIBUTE"
        body = question.strip(" ?!.")
    elif coarse_relation is QuestionRelation.DEFINITION:
        relation = "DEFINITION"
        definition = re.match(r"^(?P<subject>.+?)\s+(?:la ai|la gi)\s*[?!.]*$", folded)
        subject = _surface(question, *definition.span("subject")) if definition else None
        return QuestionSemantics(
            question_type=question_type.value,
            relation=relation,
            subject=subject,
            predicate="là",
            target=None,
            modifier=None,
            expected_answer_type=question_type.value,
        )
    elif re.search(r"\b(?:ten goi|bi danh|ten khac|ten nao)\b", folded):
        relation = "IDENTITY"
        body = question.strip(" ?!.")
    else:
        relation = "GENERAL"
        body = question.strip(" ?!.")

    subject, predicate, modifier = _split_subject_predicate(body)
    target = body if relation == "CAUSE" else None
    return QuestionSemantics(
        question_type=question_type.value,
        relation=relation,
        subject=subject,
        predicate=predicate,
        target=target,
        modifier=modifier,
        expected_answer_type=question_type.value,
    )


__all__ = ["QuestionSemantics", "parse_question_semantics"]
