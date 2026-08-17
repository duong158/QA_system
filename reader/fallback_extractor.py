from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FallbackCandidate:
    answer: str
    method: str
    score: float
    evidence_sentence: str
    start_char: int
    end_char: int
    relation_type: str | None = None
    relation_score: float = 0.0
    phrase_quality: float = 0.0
    lexical_evidence: bool = False
    relation_evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _RelationPattern:
    name: str
    expression: re.Pattern[str]
    base_score: float


_RELATION_PATTERNS = (
    _RelationPattern(
        "called_name",
        re.compile(
            r"\b(?:được[\s_]+gọi[\s_]+là|được[\s_]+biết[\s_]+đến[\s_]+(?:như|là))\b",
            re.IGNORECASE,
        ),
        0.82,
    ),
    _RelationPattern(
        "typed_copula",
        re.compile(
            r"\b(?:chính[\s_]+)?là[\s_]+(?:vị[\s_]+trí|nơi|tên[\s_]+của)\b",
            re.IGNORECASE,
        ),
        0.84,
    ),
    _RelationPattern(
        "location_or_membership",
        re.compile(
            r"\b(?:tọa[\s_]+lạc[\s_]+tại|nằm[\s_]+tại|thuộc|bao[\s_]+gồm|gồm)\b",
            re.IGNORECASE,
        ),
        0.80,
    ),
    _RelationPattern("plain_copula", re.compile(r"\b(?:chính[\s_]+)?là\b", re.IGNORECASE), 0.68),
)

_LEADING_SCAFFOLD = re.compile(
    r"^(?:(?:là|chính[\s_]+là|vị[\s_]+trí|nơi|tên[\s_]+của)[\s_:,-]+)+",
    re.IGNORECASE,
)
_LEADING_DETERMINER = re.compile(r"^(?:một|các|những)[\s_]+", re.IGNORECASE)
_TRAILING_CLAUSE = re.compile(
    r",\s+(?:nơi|mà|vốn|được|đã|sẽ|từng|nhưng|còn|do|vì|khi|trong[\s_]+khi)\b",
    re.IGNORECASE,
)
_ENTITY_DESIGNATORS = {
    "bao tang",
    "benh vien",
    "cau",
    "chua",
    "cong ty",
    "cong trinh",
    "cung dien",
    "den",
    "dia danh",
    "hoc vien",
    "lau dai",
    "nha hat",
    "nha may",
    "nha tho",
    "nui",
    "quan",
    "san van dong",
    "song",
    "tac pham",
    "tap doan",
    "thanh pho",
    "thap",
    "thu vien",
    "tinh",
    "to chuc",
    "truong",
    "vuong cung thanh duong",
}
_CLAUSE_VERBS = {
    "co",
    "da",
    "dang",
    "duoc",
    "gom",
    "la",
    "lam",
    "mang",
    "nam",
    "tro thanh",
}
_QUESTION_STOPWORDS = {
    "ai",
    "bao",
    "cai",
    "cac",
    "co",
    "cong",
    "cua",
    "duoc",
    "gi",
    "la",
    "mot",
    "nao",
    "nay",
    "nhung",
    "o",
    "tai",
    "the",
    "thuoc",
    "tren",
    "trong",
    "va",
}

LOCATION_RELATION_PATTERNS = {
    "EVENT_LOCATION": r"(?:diễn[\s_]+ra|xảy[\s_]+ra|nổ[\s_]+ra)",
    "OBJECT_LOCATION": r"(?:nằm|tọa[\s_]+lạc|đặt|đóng|thuộc)",
    "BIRTH_LOCATION": r"(?:sinh|ra[\s_]+đời)",
    "DEATH_LOCATION": r"(?:mất|qua[\s_]+đời)",
    "ORGANIZED_LOCATION": r"(?:được[\s_]+tổ[\s_]+chức|tổ[\s_]+chức)",
    "HEADQUARTERS_LOCATION": r"(?:có[\s_]+trụ[\s_]+sở|đặt[\s_]+trụ[\s_]+sở)",
    "RESIDENCE_LOCATION": r"(?:sinh[\s_]+sống|cư[\s_]+trú|tập[\s_]+trung|sống)",
}
LOCATION_RELATION_NORMALIZED_PATTERNS = {
    "EVENT_LOCATION": r"(?:dien ra|xay ra|no ra)",
    "OBJECT_LOCATION": r"(?:nam|toa lac|dat|dong|thuoc)",
    "BIRTH_LOCATION": r"(?:sinh|ra doi)",
    "DEATH_LOCATION": r"(?:mat|qua doi)",
    "ORGANIZED_LOCATION": r"(?:duoc to chuc|to chuc)",
    "HEADQUARTERS_LOCATION": r"(?:co tru so|dat tru so)",
    "RESIDENCE_LOCATION": r"(?:sinh song|cu tru|tap trung|song(?=\s+(?:o|tai)))",
}
_LOCATION_PREPOSITION = r"(?:ở|tại|trên|trong)"
_LOCATION_DESIGNATORS = {
    "chau", "dao", "dia diem", "huyen", "khu vuc", "lanh tho", "mien", "nuoc",
    "phuong", "quan", "quoc gia", "thanh pho", "thi tran", "tinh", "xa",
}
_EVENT_TERMS = {
    "cach mang", "chien tranh", "cuoc chien", "hiep dinh", "hoi nghi", "khoi nghia",
    "le hoi", "phong trao", "su kien", "tran",
}
_ORGANIZATION_TERMS = {
    "benh vien", "cong ty", "dai hoc", "doanh nghiep", "hoc vien", "tap doan",
    "to chuc", "truong dai hoc", "vien nghien cuu",
}
_LOCATION_TRAILING_BOUNDARY = re.compile(
    r"\s+(?:vào|từ|kể[\s_]+từ|năm|tháng|ngày|khi|sau|trước|do|vì|nhằm|để)\b",
    re.IGNORECASE,
)
_LOCATION_CLAUSE_BOUNDARY = re.compile(
    r",\s+(?:nơi|mà|vốn|được|đã|sẽ|từng|nhưng|còn|do|vì|khi|trong[\s_]+khi|thủ[\s_]+đô)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text or "").casefold().replace("đ", "d"))
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return " ".join(re.findall(r"[\w%]+", value, flags=re.UNICODE))


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def detect_contrast_relation(question: str) -> bool:
    normalized = _normalize(question)
    return bool(
        re.search(r"\b(?:khac biet|khac nhau|phan biet|so voi|so sanh)\b", normalized)
    )


def assess_contrast_relation(answer: str) -> tuple[float, bool]:
    """Score whether an answer states two sides of a requested contrast."""

    normalized = _normalize(answer)
    if not normalized:
        return 0.0, False
    if re.search(r"\b(?:su )?khac biet (?:nay|do)\b", normalized):
        # An anaphoric sentence names the relation but omits both sides.
        return 0.05, False
    if re.search(r"\bgiua\b.+\bva\b.+", normalized):
        return 1.0, True
    if re.search(r"\btrong khi\b", normalized):
        return 0.95, True
    if re.search(r"\b(?:khac voi|so voi)\b", normalized):
        return 0.85, True
    return 0.0, False


def detect_alias_relation(question: str) -> bool:
    normalized = _normalize(question)
    return bool(re.search(r"\b(?:ten goi|bi danh|ten khac|ten) nao\b", normalized))


_UPPERCASE_VI = r"A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ"
_PROPER_NAME = (
    rf"[{_UPPERCASE_VI}][\wÀ-ỹĐđ'’-]*"
    rf"(?:[\s_]+[{_UPPERCASE_VI}][\wÀ-ỹĐđ'’-]*){{0,5}}"
)


def extract_alias_candidate(question: str, sentence: str) -> FallbackCandidate | None:
    """Extract a complete proper name from an explicit naming relation."""

    if not detect_alias_relation(question):
        return None
    expression = re.compile(
        rf"\b(?:có[\s_]+)?(?:tên[\s_]+gọi|bí[\s_]+danh|tên[\s_]+khác)"
        rf"(?:[\s_]+(?:của|là))?[\s_:,-]+(?:là[\s_]+)?(?P<answer>{_PROPER_NAME})",
        re.UNICODE,
    )
    candidates: list[FallbackCandidate] = []
    for match in expression.finditer(sentence):
        start, end = match.span("answer")
        answer = sentence[start:end].strip()
        if len(_tokens(answer)) < 2:
            continue
        candidates.append(
            FallbackCandidate(
                answer=answer,
                method="alias_relation_pattern",
                score=0.95,
                evidence_sentence=sentence,
                start_char=start,
                end_char=end,
                relation_type="ALIAS",
                relation_score=1.0,
                phrase_quality=1.0,
                lexical_evidence=True,
                relation_evidence=True,
            )
        )
    return max(candidates, key=lambda item: (item.score, len(_tokens(item.answer)))) if candidates else None


_TEMPORAL_EXPRESSIONS = (
    re.compile(r"\b(?:Năm|năm)[\s_]+\d{3,4}\b", re.UNICODE),
    re.compile(
        r"\b(?:Tháng|tháng)[\s_]+\d{1,2}(?:[\s_]+(?:năm)[\s_]+\d{3,4})?\b",
        re.UNICODE,
    ),
    re.compile(
        r"\b(?:Ngày|ngày)[\s_]+\d{1,2}(?:[\s_]+tháng[\s_]+\d{1,2})?"
        r"(?:[\s_]+năm[\s_]+\d{3,4})?\b",
        re.UNICODE,
    ),
    re.compile(r"\b(?:thế[\s_]+kỷ)[\s_]+(?:\d{1,2}|[IVX]{1,5})\b", re.IGNORECASE),
)


def extract_temporal_candidate(question: str, sentence: str) -> FallbackCandidate | None:
    candidates: list[FallbackCandidate] = []
    for expression in _TEMPORAL_EXPRESSIONS:
        for match in expression.finditer(sentence):
            local_context = sentence[max(0, match.start() - 100) : min(len(sentence), match.end() + 100)]
            score = 0.90 + _relation_to_question_score(question, local_context)
            candidates.append(
                FallbackCandidate(
                    answer=match.group(0),
                    method="temporal_expression_pattern",
                    score=round(min(1.0, score), 6),
                    evidence_sentence=sentence,
                    start_char=match.start(),
                    end_char=match.end(),
                    relation_type="TEMPORAL_EXPRESSION",
                    relation_score=0.9,
                    phrase_quality=1.0,
                    lexical_evidence=True,
                    relation_evidence=True,
                )
            )
    return max(candidates, key=lambda item: (item.score, -item.start_char)) if candidates else None


_NUMBER_WORD_EXPRESSION = re.compile(
    r"\b(?:không|một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|mười)"
    r"(?:[\s_]+(?:người|đứa|người[\s_]+con))?[\s_]+(?:con|người|lần|phần|nhóm|chiếc|năm)\b",
    re.IGNORECASE | re.UNICODE,
)
_DIGIT_NUMBER_EXPRESSION = re.compile(
    r"\b(?:hơn|gần|khoảng|trên|dưới)?[\s_]*\d+(?:[.,]\d+)?(?:[\s_]*%)?"
    r"(?:[\s_]+(?:người|con|lần|phần|nhóm|chiếc|km|mét|triệu|tỷ|nghìn))?\b",
    re.IGNORECASE | re.UNICODE,
)


def extract_number_candidate(question: str, sentence: str) -> FallbackCandidate | None:
    candidates = []
    for expression in (_DIGIT_NUMBER_EXPRESSION, _NUMBER_WORD_EXPRESSION):
        for match in expression.finditer(sentence):
            candidates.append(
                FallbackCandidate(
                    answer=match.group(0).strip(),
                    method="number_expression_pattern",
                    score=0.90,
                    evidence_sentence=sentence,
                    start_char=match.start(),
                    end_char=match.end(),
                    relation_type="NUMBER_EXPRESSION",
                    relation_score=0.85,
                    phrase_quality=0.95,
                    lexical_evidence=True,
                    relation_evidence=True,
                )
            )
    return max(candidates, key=lambda item: (-len(_tokens(item.answer)), -item.start_char)) if candidates else None


_PERSON_TITLE = (
    r"(?:Thiếu[\s_]+tướng|Đại[\s_]+tướng|Trung[\s_]+tướng|Tổng[\s_]+thống|"
    r"Thủ[\s_]+tướng|Chủ[\s_]+tịch|Giáo[\s_]+sư|Tiến[\s_]+sĩ)"
)


def _person_definition_subject(question: str) -> str | None:
    """Return the named subject in questions such as ``Phạm Văn Đồng là ai?``."""

    match = re.fullmatch(
        r"\s*(?P<subject>.+?)\s+là\s+ai\s*[?!.]*\s*",
        str(question or ""),
        flags=re.IGNORECASE | re.UNICODE,
    )
    if not match:
        return None
    subject = match.group("subject").strip()
    normalized = _normalize(subject)
    if len(_tokens(subject)) < 2 or normalized in {
        "nguoi nay",
        "nguoi do",
        "ong ay",
        "ba ay",
    }:
        return None
    return subject


def extract_person_definition_candidate(
    question: str,
    sentence: str,
) -> FallbackCandidate | None:
    """Extract the complete predicate for a named-person definition question."""

    subject = _person_definition_subject(question)
    if not subject:
        return None
    subject_match = re.search(re.escape(subject), sentence, flags=re.IGNORECASE | re.UNICODE)
    if not subject_match:
        return None
    relation = re.match(
        r"\s*(?:\([^)]*\))?\s+là\s+",
        sentence[subject_match.end() :],
        flags=re.IGNORECASE | re.UNICODE,
    )
    if not relation:
        return None
    start = subject_match.end() + relation.end()
    end = len(sentence)
    while end > start and (sentence[end - 1].isspace() or sentence[end - 1] in ".!?;"):
        end -= 1
    if end <= start:
        return None
    return FallbackCandidate(
        answer=sentence[start:end],
        method="person_definition_pattern",
        score=0.97,
        evidence_sentence=sentence,
        start_char=start,
        end_char=end,
        relation_type="PERSON_DEFINITION",
        relation_score=0.98,
        phrase_quality=0.98,
        lexical_evidence=True,
        relation_evidence=True,
    )


def extract_person_candidate(question: str, sentence: str) -> FallbackCandidate | None:
    patterns = (
        re.compile(
            rf"\b(?:là|do)[\s_]+(?P<answer>(?:{_PERSON_TITLE}[\s_]+)?{_PROPER_NAME})"
            rf"(?=[\s_]+(?:lãnh[\s_]+đạo|đứng[\s_]+đầu)|[,.;!?]|$)",
            re.UNICODE,
        ),
        re.compile(rf"\b(?P<answer>{_PERSON_TITLE}[\s_]+{_PROPER_NAME})", re.UNICODE),
    )
    candidates: list[FallbackCandidate] = []
    for pattern in patterns:
        for match in pattern.finditer(sentence):
            start, end = match.span("answer")
            answer = sentence[start:end]
            # The broad Unicode proper-name matcher can regard a lowercase
            # Vietnamese relation phrase as another name token. Keep the
            # grammatical relation as evidence, but do not return it as part
            # of the person's name.
            answer = re.sub(
                r"[\s_]+(?:lãnh[\s_]+đạo|đứng[\s_]+đầu)$",
                "",
                answer,
                flags=re.IGNORECASE,
            ).rstrip()
            end = start + len(answer)
            candidates.append(
                FallbackCandidate(
                    answer=answer,
                    method="person_relation_pattern",
                    score=0.93,
                    evidence_sentence=sentence,
                    start_char=start,
                    end_char=end,
                    relation_type="PERSON_RELATION",
                    relation_score=0.9,
                    phrase_quality=0.95,
                    lexical_evidence=True,
                    relation_evidence=True,
                )
            )
    return max(candidates, key=lambda item: (item.score, -len(_tokens(item.answer)))) if candidates else None


def extract_contrast_candidate(question: str, sentence: str) -> FallbackCandidate | None:
    if not detect_contrast_relation(question):
        return None
    patterns = (
        re.compile(
            r"(?:sự[\s_]+)?khác[\s_]+biệt[\s_]+giữa[\s_]+.+?[\s_]+và[\s_]+.+?(?=[.;!?]|$)",
            re.IGNORECASE,
        ),
        re.compile(r"[^.;!?]+?[\s_]+trong[\s_]+khi[\s_]+[^.;!?]+", re.IGNORECASE),
    )
    candidates: list[FallbackCandidate] = []
    for pattern in patterns:
        for match in pattern.finditer(sentence):
            start, end = match.span()
            while start < end and sentence[start].isspace():
                start += 1
            while end > start and sentence[end - 1].isspace():
                end -= 1
            answer = sentence[start:end]
            relation_score, relation_evidence = assess_contrast_relation(answer)
            if not relation_evidence:
                continue
            candidates.append(
                FallbackCandidate(
                    answer=answer,
                    method="contrast_relation_pattern",
                    score=0.90,
                    evidence_sentence=sentence,
                    start_char=start,
                    end_char=end,
                    relation_type="CONTRAST",
                    relation_score=relation_score,
                    phrase_quality=0.90,
                    lexical_evidence=True,
                    relation_evidence=True,
                )
            )
    return max(candidates, key=lambda item: (item.relation_score, item.score)) if candidates else None


def _contains_designator(text: str) -> bool:
    normalized = _normalize(text)
    return any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in _ENTITY_DESIGNATORS)


def _has_proper_name(text: str) -> bool:
    return bool(re.search(r"\b[A-ZÀ-ỸĐ][\wÀ-ỹĐđ'-]*\b", text, flags=re.UNICODE))


def detect_location_relation(question: str) -> str:
    normalized = _normalize(question)
    for relation_type, pattern in LOCATION_RELATION_NORMALIZED_PATTERNS.items():
        if re.search(rf"\b{pattern}\b", normalized):
            return relation_type
    if re.search(r"\b(?:o dau|tai dau)\b", normalized):
        return "OBJECT_LOCATION"
    return "GENERIC_LOCATION"


def _question_location_subject(question: str, relation_type: str) -> str:
    normalized = _normalize(question).strip(" ?!.")
    pattern = LOCATION_RELATION_NORMALIZED_PATTERNS.get(relation_type)
    if pattern:
        match = re.search(rf"\b{pattern}\b", normalized)
        if match:
            normalized = normalized[: match.start()]
    normalized = re.sub(
        r"\b(?:o dau|tai dau|noi nao|dia diem nao|khu vuc nao|quoc gia nao|nuoc nao|tinh nao)\b.*$",
        "",
        normalized,
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _subject_matches(subject: str, text: str) -> bool:
    if not subject:
        return True
    subject_tokens = {
        token for token in subject.split() if token not in _QUESTION_STOPWORDS and len(token) > 1
    }
    if not subject_tokens:
        return True
    text_tokens = set(_tokens(text))
    coverage = len(subject_tokens & text_tokens) / len(subject_tokens)
    return coverage >= 0.6


def _trim_location_span(sentence: str, start: int, end: int | None = None) -> tuple[int, int]:
    end = len(sentence) if end is None else min(len(sentence), end)
    while start < end and (sentence[start].isspace() or sentence[start] in "\"'“”‘’,:-"):
        start += 1

    fragment = sentence[start:end]
    punctuation = re.search(r"[;.!?…\n]", fragment)
    if punctuation:
        end = start + punctuation.start()
    temporal = _LOCATION_TRAILING_BOUNDARY.search(sentence[start:end])
    if temporal:
        end = start + temporal.start()
    clause = _LOCATION_CLAUSE_BOUNDARY.search(sentence[start:end])
    if clause:
        end = start + clause.start()
    dash = re.search(r"\s+[–—-]\s+", sentence[start:end])
    if dash:
        end = start + dash.start()
    while end > start and (sentence[end - 1].isspace() or sentence[end - 1] in "\"'“”‘’,:-"):
        end -= 1
    return start, end


def _location_phrase_quality(answer: str) -> float:
    normalized = _normalize(answer)
    tokens = normalized.split()
    if not tokens:
        return 0.0
    if any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in _EVENT_TERMS):
        return 0.05
    if any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in _ORGANIZATION_TERMS):
        return 0.15
    score = 0.52
    if any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in _LOCATION_DESIGNATORS):
        score += 0.25
    if _has_proper_name(answer):
        score += 0.16
    if 1 <= len(tokens) <= 8:
        score += 0.07
    elif len(tokens) > 16:
        score -= 0.30
    return round(max(0.0, min(1.0, score)), 6)


def _location_candidate(
    sentence: str,
    start: int,
    end: int,
    relation_type: str,
    relation_score: float,
) -> FallbackCandidate | None:
    start, end = _trim_location_span(sentence, start, end)
    if end <= start:
        return None
    answer = sentence[start:end]
    quality = _location_phrase_quality(answer)
    if quality < 0.60:
        return None
    return FallbackCandidate(
        answer=answer,
        method="location_relation_pattern",
        score=round(0.55 * relation_score + 0.45 * quality, 6),
        evidence_sentence=sentence,
        start_char=start,
        end_char=end,
        relation_type=relation_type,
        relation_score=relation_score,
        phrase_quality=quality,
        lexical_evidence=True,
        relation_evidence=relation_score >= 0.75,
    )


def extract_location_candidate(
    question: str,
    sentence: str,
    relation: str | None = None,
) -> FallbackCandidate:
    """Extract a location phrase that fills the relation requested by the question."""

    sentence = str(sentence or "").strip()
    relation_type = relation or detect_location_relation(question)
    subject = _question_location_subject(question, relation_type)
    relation_pattern = LOCATION_RELATION_PATTERNS.get(relation_type)
    candidates: list[FallbackCandidate] = []

    if relation_pattern:
        direct = re.compile(
            rf"\b(?P<relation>{relation_pattern})\b"
            rf"(?:[\s_]+(?:chủ[\s_]+yếu|chính|phần[\s_]+lớn))?"
            rf"[\s_,:-]+{_LOCATION_PREPOSITION}[\s_]+",
            re.IGNORECASE,
        )
        for match in direct.finditer(sentence):
            if not _subject_matches(subject, sentence[max(0, match.start() - 180) : match.start()]):
                continue
            candidate = _location_candidate(
                sentence,
                match.end(),
                len(sentence),
                relation_type,
                1.0,
            )
            if candidate:
                candidates.append(candidate)

        if relation_type == "OBJECT_LOCATION":
            object_links = re.compile(
                rf"\b(?:thuộc[\s_]+|{_LOCATION_PREPOSITION}[\s_]+)",
                re.IGNORECASE,
            )
            for match in object_links.finditer(sentence):
                if not _subject_matches(subject, sentence[: match.start()]):
                    continue
                candidate = _location_candidate(
                    sentence,
                    match.end(),
                    len(sentence),
                    relation_type,
                    0.90,
                )
                if candidate:
                    candidates.append(candidate)

        reverse = re.compile(
            rf"\b(?:là|được[\s_]+xem[\s_]+là)[\s_]+nơi[\s_]+"
            rf"(?P<relation>{relation_pattern})\b",
            re.IGNORECASE,
        )
        for match in reverse.finditer(sentence):
            tail = sentence[match.end() :]
            if not _subject_matches(subject, tail):
                continue
            prefix = sentence[: match.start()]
            deictic = re.search(r"\b(?:đây|nơi[\s_]+đây|thành[\s_]+phố[\s_]+này|khu[\s_]+vực[\s_]+này)\s*$", prefix, re.IGNORECASE)
            if deictic:
                temporal_prefix = re.match(
                    r"^(?:(?:vào|từ|tới|đến)[\s_]+)?(?:thế[\s_]+kỷ|năm|tháng|ngày)[^,]{0,30},[\s_]*",
                    sentence,
                    re.IGNORECASE,
                )
                topic_offset = temporal_prefix.end() if temporal_prefix else 0
                topic = re.match(
                    r"(?P<topic>[A-ZÀ-ỸĐ][\wÀ-ỹĐđ'-]*(?:[\s_]+[A-ZÀ-ỸĐ][\wÀ-ỹĐđ'-]*){0,3})\b",
                    sentence[topic_offset:],
                )
                if topic:
                    candidate = _location_candidate(
                        sentence,
                        topic_offset + topic.start("topic"),
                        topic_offset + topic.end("topic"),
                        relation_type,
                        0.92,
                    )
                    if candidate:
                        candidates.append(candidate)
            else:
                location_match = re.search(
                    r"(?P<location>(?:nước|thành[\s_]+phố|tỉnh|quận|huyện|khu[\s_]+vực)?[\s_]*"
                    r"[A-ZÀ-ỸĐ][\wÀ-ỹĐđ'-]*(?:[\s_]+[A-ZÀ-ỸĐ][\wÀ-ỹĐđ'-]*){0,3})[\s_]*$",
                    prefix,
                )
                if location_match:
                    candidate = _location_candidate(
                        sentence,
                        location_match.start("location"),
                        location_match.end("location"),
                        relation_type,
                        0.95,
                    )
                    if candidate:
                        candidates.append(candidate)

    if candidates:
        return max(candidates, key=lambda item: (item.score, -len(_tokens(item.answer))))
    return FallbackCandidate(
        answer=sentence,
        method="whole_sentence",
        score=0.20,
        evidence_sentence=sentence,
        start_char=0,
        end_char=len(sentence),
        relation_type=relation_type,
        relation_score=0.0,
        phrase_quality=0.0,
        lexical_evidence=True,
        relation_evidence=False,
    )


def _relation_to_question_score(question: str, sentence_prefix: str) -> float:
    question_tokens = {
        token for token in _tokens(question) if len(token) > 1 and token not in _QUESTION_STOPWORDS
    }
    if not question_tokens:
        return 0.0
    prefix_tokens = set(_tokens(sentence_prefix))
    coverage = len(question_tokens & prefix_tokens) / len(question_tokens)
    return min(0.12, coverage * 0.18)


def _trim_candidate_span(sentence: str, start: int) -> tuple[int, int]:
    while start < len(sentence) and (sentence[start].isspace() or sentence[start] in ":,-"):
        start += 1
    end = len(sentence)
    tail = sentence[start:]

    punctuation = re.search(r"[;.!?…\n]", tail)
    if punctuation:
        end = min(end, start + punctuation.start())
    clause = _TRAILING_CLAUSE.search(sentence[start:end])
    if clause:
        end = start + clause.start()
    dash = re.search(r"\s+[–—-]\s+", sentence[start:end])
    if dash:
        end = start + dash.start()

    while end > start and (sentence[end - 1].isspace() or sentence[end - 1] in "\"'“”‘’,:-"):
        end -= 1

    while True:
        fragment = sentence[start:end]
        scaffold = _LEADING_SCAFFOLD.match(fragment)
        if scaffold:
            start += scaffold.end()
            continue
        determiner = _LEADING_DETERMINER.match(fragment)
        if determiner:
            start += determiner.end()
            continue
        break

    while start < end and (sentence[start].isspace() or sentence[start] in "\"'“”‘’,:-"):
        start += 1
    return start, end


def _candidate_quality(
    question: str,
    sentence: str,
    relation_start: int,
    answer: str,
    base_score: float,
) -> float:
    tokens = _tokens(answer)
    word_count = len(tokens)
    score = base_score + _relation_to_question_score(question, sentence[:relation_start])
    if 1 <= word_count <= 8:
        score += 0.08
    elif word_count <= 16:
        score += 0.04
    elif word_count > 28:
        score -= 0.18
    if _contains_designator(answer):
        score += 0.08
    if _has_proper_name(answer):
        score += 0.07
    clause_hits = sum(
        1 for term in _CLAUSE_VERBS if re.search(rf"\b{re.escape(term)}\b", _normalize(answer))
    )
    if word_count > 10 and clause_hits >= 2:
        score -= min(0.20, 0.07 * clause_hits)
    return round(max(0.0, min(1.0, score)), 6)


def extract_fallback_answer(
    question: str,
    question_type: str,
    sentence: str,
) -> FallbackCandidate:
    """Narrow a selected supporting sentence to a grounded answer phrase.

    This function never searches the corpus. It only extracts an exact span
    from the supporting sentence already chosen by the retriever/fallback
    sentence scorer. Whole-sentence output remains the conservative last
    resort for unsupported question types or weak phrase candidates.
    """

    sentence = str(sentence or "").strip()
    if not sentence:
        return FallbackCandidate("", "whole_sentence", 0.0, "", -1, -1)

    normalized_type = str(getattr(question_type, "value", question_type)).upper()
    contrast = extract_contrast_candidate(question, sentence)
    if contrast is not None:
        return contrast
    alias = extract_alias_candidate(question, sentence)
    if alias is not None:
        return alias
    if normalized_type == "TIME":
        temporal = extract_temporal_candidate(question, sentence)
        if temporal is not None:
            return temporal
    if normalized_type == "NUMBER":
        number = extract_number_candidate(question, sentence)
        if number is not None:
            return number
    if normalized_type == "PERSON":
        definition = extract_person_definition_candidate(question, sentence)
        if definition is not None:
            return definition
        person = extract_person_candidate(question, sentence)
        if person is not None:
            return person
    if normalized_type == "LOCATION":
        return extract_location_candidate(question, sentence)
    if normalized_type != "ENTITY":
        return FallbackCandidate(
            sentence,
            "whole_sentence",
            0.4,
            sentence,
            0,
            len(sentence),
            lexical_evidence=True,
        )

    candidates: list[FallbackCandidate] = []
    seen_spans: set[tuple[int, int]] = set()
    for relation in _RELATION_PATTERNS:
        for match in relation.expression.finditer(sentence):
            if relation.name == "plain_copula":
                comparison_prefix = _normalize(sentence[max(0, match.start() - 32) : match.start()])
                if re.search(r"\b(?:hon|kem|thay vi|khong phai|cung nhu|vi nhu)\s*$", comparison_prefix):
                    continue
            start, end = _trim_candidate_span(sentence, match.end())
            if end <= start or (start, end) in seen_spans:
                continue
            seen_spans.add((start, end))
            answer = sentence[start:end]
            if len(_tokens(answer)) == 0 or _normalize(answer) == _normalize(sentence):
                continue
            score = _candidate_quality(
                question,
                sentence,
                match.start(),
                answer,
                relation.base_score,
            )
            if score < 0.72:
                continue
            candidates.append(
                FallbackCandidate(
                    answer=answer,
                    method="entity_relation_pattern",
                    score=score,
                    evidence_sentence=sentence,
                    start_char=start,
                    end_char=end,
                )
            )

    if candidates:
        return max(
            candidates,
            key=lambda item: (
                item.score,
                -len(_tokens(item.answer)),
                item.start_char,
            ),
        )
    return FallbackCandidate(sentence, "whole_sentence", 0.4, sentence, 0, len(sentence))


__all__ = [
    "FallbackCandidate",
    "detect_location_relation",
    "detect_contrast_relation",
    "detect_alias_relation",
    "assess_contrast_relation",
    "extract_contrast_candidate",
    "extract_alias_candidate",
    "extract_temporal_candidate",
    "extract_number_candidate",
    "extract_person_definition_candidate",
    "extract_person_candidate",
    "extract_fallback_answer",
    "extract_location_candidate",
]
