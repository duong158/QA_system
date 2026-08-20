from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from reader.answer_refinement import QuestionRelation, detect_question_relation
from reader.cause_relations import fold_text
from reader.question_type import QuestionType, detect_question_type


@dataclass(frozen=True)
class QuestionSemantics:
    question_type: list[str]
    relation: str
    subject: str | None
    predicate: str | None
    target: str | None
    modifier: str | None
    expected_answer_type: list[str]
    matched_rule_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    pattern: re.Pattern[str]
    general: bool
    description: str
    relation: str | None = None


QUESTION_RULES = (
    Rule(
        "CAUSE_MARKER_PREFIX",
        "question_relation",
        re.compile(r"^(?:vi sao|tai sao|vi dau|do dau|boi dau)\s+(?P<body>.+?)\s*[?!.]*$"),
        True,
        "Vietnamese causal interrogative prefix",
        "CAUSE",
    ),
    Rule(
        "CAUSE_NOMINAL_PREFIX",
        "question_relation",
        re.compile(
            r"^(?:do nguyen nhan nao|nguyen nhan(?:\s+[\w]+){0,3}?\s+nao|"
            r"nguyen nhan gi|ly do gi|ly do nao|vi nguyen nhan gi)\s+"
            r"(?:(?:khien|lam)\s+)?(?P<body>.+?)\s*[?!.]*$"
        ),
        True,
        "Nominal cause-question construction",
        "CAUSE",
    ),
    Rule(
        "CAUSE_TRIGGER_PREFIX",
        "question_relation",
        re.compile(
            r"^(?:dieu gi|yeu to nao)\s+(?:khien|lam)\s+(?P<body>.+?)\s*[?!.]*$|"
            r"^dieu gi\s+(?:dan toi|gay ra)\s+(?:viec\s+)?(?P<body_alt>.+?)\s*[?!.]*$"
        ),
        True,
        "Cause trigger/effect construction",
        "CAUSE",
    ),
    Rule(
        "CAUSE_MARKER_SUFFIX",
        "question_relation",
        re.compile(
            r"^(?P<body>.+?)\s+(?:vi sao|tai sao|vi ly do gi|do nguyen nhan gi|"
            r"vi nguyen nhan gi)\s*[?!.]*$"
        ),
        True,
        "Vietnamese causal interrogative suffix",
        "CAUSE",
    ),
    Rule(
        "TIME_BIRTH_MARKER",
        "question_relation",
        re.compile(r"\b(?:nam sinh cua|ngay sinh cua|sinh(?: ra)?(?: vao)? nam|chao doi)\b"),
        True,
        "Birth-time question marker",
        "TIME",
    ),
    Rule(
        "TIME_DEATH_MARKER",
        "question_relation",
        re.compile(r"\b(?:mat|qua doi|tu tran)\b"),
        True,
        "Death-time question marker",
        "TIME",
    ),
    Rule(
        "LOCATION_INTERROGATIVE",
        "question_relation",
        re.compile(r"\b(?:o dau|tai dau|noi nao|dia diem nao)\b"),
        True,
        "Location interrogative marker",
        "LOCATION",
    ),
    Rule(
        "PURPOSE_INTERROGATIVE",
        "question_relation",
        re.compile(r"\b(?:voi|nham)?\s*(?:muc dich|muc tieu)\s+gi\b|\bde lam gi\b"),
        True,
        "Purpose interrogative marker",
        "PURPOSE",
    ),
    Rule(
        "IDENTITY_INTERROGATIVE",
        "question_relation",
        re.compile(r"\b(?:ten goi|bi danh|ten khac|ten nao)\b"),
        True,
        "Alias or identity interrogative marker",
        "ENTITY",
    ),
)


PREDICATE_BOUNDARY_RULES = (
    Rule(
        "PREDICATE_AUXILIARY_BOUNDARY",
        "predicate_boundary",
        re.compile(r"\b(?P<predicate>(?:bi|duoc|da|dang|se|tung|phai)\b)"),
        True,
        "Vietnamese aspect, passive, and modal auxiliary boundary",
    ),
    Rule(
        "PREDICATE_EVENT_BOUNDARY",
        "predicate_boundary",
        re.compile(
            r"\b(?P<predicate>(?:sinh|chao doi|ra doi|mat|qua doi|dien ra|xay ra|"
            r"no ra|toa lac|nam|gui|phat trien|gia tang|suy giam|bung no|sup do|"
            r"tro thanh|thanh lap|xay dung))\b"
        ),
        True,
        "General event/state-change verb boundary",
    ),
    Rule(
        "PREDICATE_COPULA_BOUNDARY",
        "predicate_boundary",
        re.compile(r"\b(?P<predicate>(?:co|la))\b"),
        True,
        "Vietnamese copular or possession boundary",
    ),
)


MODIFIER_RULE = Rule(
    "PREDICATE_MODIFIER_BOUNDARY",
    "predicate_modifier",
    re.compile(r"\s+\b(?:tai|o|tren|trong|vao|tu|den)\b\s+"),
    True,
    "Locative or temporal modifier boundary",
)


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
    for rule in QUESTION_RULES:
        if rule.relation != "CAUSE":
            continue
        match = rule.pattern.match(folded)
        if match:
            group = "body" if match.groupdict().get("body") is not None else "body_alt"
            return _surface(question, *match.span(group))
    return None


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
            r"\s+(?:voi muc dich gi|nham muc dich gi|nham muc tieu gi|de lam gi)\s*[?!.]*$",
        ),
        "EVENT_LOCATION": (
            r"\s+(?:o dau|tai dau|noi nao|dia diem nao)\s*[?!.]*$",
        ),
        "OBJECT_LOCATION": (
            r"\s+(?:o dau|tai dau|noi nao|dia diem nao)\s*[?!.]*$",
            r"\s+nam\s+ben\s+bo\s+(?:con\s+)?song\s+nao\s*[?!.]*$",
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
    matches = [
        match
        for rule in PREDICATE_BOUNDARY_RULES
        if (match := rule.pattern.search(folded)) is not None
    ]
    match = min(matches, key=lambda item: item.start()) if matches else None
    if not match or match.start() <= 0:
        return body.strip() or None, None, None

    subject = _surface(body, 0, match.start())
    predicate_start = match.start("predicate")
    modifier_match = MODIFIER_RULE.pattern.search(folded, match.end("predicate"))
    predicate_end = modifier_match.start() if modifier_match else len(body)
    modifier_start = modifier_match.start() if modifier_match else len(body)
    predicate = _surface(body, predicate_start, predicate_end)
    modifier = _surface(body, modifier_start, len(body)) if modifier_match else None
    return subject, predicate, modifier


def matched_question_rules(question: str) -> tuple[str, ...]:
    folded = fold_text(question).strip()
    matched = [rule.id for rule in QUESTION_RULES if rule.pattern.search(folded)]
    body = _cause_body(question) or question.strip(" ?!.,;:")
    folded_body = fold_text(body)
    matched.extend(
        rule.id for rule in PREDICATE_BOUNDARY_RULES if rule.pattern.search(folded_body)
    )
    if MODIFIER_RULE.pattern.search(folded_body):
        matched.append(MODIFIER_RULE.id)
    return tuple(dict.fromkeys(matched))


def _time_relation(folded: str) -> str:
    if re.search(r"\b(?:nam sinh|sinh(?: ra)?(?: vao)? nam|sinh nam|ra doi|chao doi)\b", folded):
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
    if re.search(r"\b(?:nam|toa lac|dat|thuoc|sinh song|cu tru)\b", folded):
        return "OBJECT_LOCATION"
    return "GENERIC_LOCATION"


def _subject_from_nominal_time_question(question: str, folded: str) -> str | None:
    match = re.match(
        r"^(?:nam sinh|ngay sinh)\s+cua\s+(?P<subject>.+?)\s+la\s+(?:bao nhieu|gi|ngay nao|nam nao)",
        folded,
    )
    return _surface(question, *match.span("subject")) if match else None


def parse_question_semantics(question: str) -> QuestionSemantics:
    question = str(question or "").strip()
    question_types = detect_question_type(question)
    if not question_types:
        question_types = [QuestionType.GENERAL]
    folded = fold_text(question).strip()
    matched_rules = matched_question_rules(question)
    coarse_relation = detect_question_relation(question, question_types)
    cause_body = _cause_body(question)
    birth_time_intent = bool(
        re.search(r"\b(?:nam sinh cua|ngay sinh cua|sinh(?: ra)?(?: vao)? nam|chao doi)\b", folded)
    )
    purpose_intent = bool(
        re.search(r"\b(?:voi|nham)?\s*(?:muc dich|muc tieu)\s+gi\b|\bde lam gi\b", folded)
    )
    identity_intent = bool(re.search(r"\b(?:ten goi|bi danh|ten khac|ten nao)\b", folded))

    if cause_body is not None:
        relation = "CAUSE"
        body = cause_body
    elif QuestionType.TIME in question_types or birth_time_intent:
        relation = "BIRTH_TIME" if birth_time_intent else _time_relation(folded)
        question_types = [QuestionType.TIME]
        nominal_subject = _subject_from_nominal_time_question(question, folded)
        if nominal_subject:
            return QuestionSemantics(
                question_type=[qt.value for qt in question_types],
                relation=relation,
                subject=nominal_subject,
                predicate="sinh" if relation == "BIRTH_TIME" else None,
                target=None,
                modifier=None,
                expected_answer_type=[QuestionType.TIME.value],
                matched_rule_ids=matched_rules,
            )
        body = _strip_interrogative_tail(question, relation)
    elif QuestionType.LOCATION in question_types:
        relation = _location_relation(folded)
        body = _strip_interrogative_tail(question, relation)
    elif purpose_intent or coarse_relation is QuestionRelation.PURPOSE:
        relation = "PURPOSE"
        body = _strip_interrogative_tail(question, relation)
    elif coarse_relation is QuestionRelation.CONTRAST:
        relation = "CONTRAST"
        body = question.strip(" ?!.")
    elif coarse_relation is QuestionRelation.ATTRIBUTE:
        relation = "ATTRIBUTE"
        body = question.strip(" ?!.")
    elif identity_intent:
        relation = "IDENTITY"
        body = question.strip(" ?!.")
    elif coarse_relation is QuestionRelation.DEFINITION:
        relation = "DEFINITION"
        definition = re.match(r"^(?P<subject>.+?)\s+(?:la ai|la gi)\s*[?!.]*$", folded)
        subject = _surface(question, *definition.span("subject")) if definition else None
        return QuestionSemantics(
            question_type=[qt.value for qt in question_types],
            relation=relation,
            subject=subject,
            predicate="là",
            target=None,
            modifier=None,
            expected_answer_type=[qt.value for qt in question_types],
            matched_rule_ids=matched_rules,
        )
    else:
        relation = "GENERAL"
        body = question.strip(" ?!.")

    subject, predicate, modifier = _split_subject_predicate(body)
    target = body if relation == "CAUSE" else None
    return QuestionSemantics(
        question_type=[qt.value for qt in question_types],
        relation=relation,
        subject=subject,
        predicate=predicate,
        target=target,
        modifier=modifier,
        expected_answer_type=[qt.value for qt in question_types],
        matched_rule_ids=matched_rules,
    )


__all__ = [
    "MODIFIER_RULE",
    "PREDICATE_BOUNDARY_RULES",
    "QUESTION_RULES",
    "QuestionSemantics",
    "Rule",
    "matched_question_rules",
    "parse_question_semantics",
]
