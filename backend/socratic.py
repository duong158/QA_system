from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from backend.chunking import split_sentences
from reader.question_semantics import QuestionSemantics
from reader.subject_consistency import SemanticStatus, score_subject_consistency


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "socratic.json"


@dataclass(frozen=True)
class SocraticConfig:
    enabled: bool = True
    max_followups: int = 3
    max_internal_candidates: int = 10
    min_answerability_score: float = 0.62
    min_topic_relevance: float = 0.50
    avoid_same_relation: bool = True
    allow_bm25_probe: bool = True
    probe_top_k: int = 5
    max_bm25_probes: int = 3
    max_context_passages: int = 12
    duplicate_similarity_threshold: float = 0.72
    one_hop_only: bool = True
    prefer_relation_diversity: bool = True
    answerability_weight: float = 0.45
    relevance_weight: float = 0.25
    novelty_weight: float = 0.2
    diversity_weight: float = 0.1


@dataclass(frozen=True)
class FollowUpCandidate:
    question: str
    subject: str | None
    relation: str | None
    question_type: str | None
    source_passage_id: str | None
    answerability_score: float
    novelty_score: float
    relevance_score: float
    ranking_score: float
    qa_verified: bool = False
    verification_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = payload.pop("question_type")
        return payload


@dataclass
class SocraticCandidateTrace:
    question: str
    relation: str | None
    subject: str | None
    source_passage_id: str | None
    generated_by: str
    evidence_sentence: str | None = None
    target: str | None = None
    predicate: str | None = None
    subject_match: str | None = None
    subject_score: float | None = None
    topic_relevance_score: float | None = None
    relevance_score: float | None = None
    novelty_score: float | None = None
    answerability_score: float | None = None
    ranking_score: float | None = None
    tier: str | None = None
    probe_latency_ms: float | None = None
    qa_verified: bool | None = None
    verification_method: str | None = None
    verification_rejection_reason: str | None = None
    why_accepted: str | None = None
    accepted: bool = False
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FollowUpOpportunity:
    subject: str
    relation: str
    question_type: str
    source_passage_id: str
    evidence_sentence: str
    generated_by: str
    evidence_strength: float
    relevance_score: float
    topic_relevance_score: float
    subject_match: str
    subject_score: float
    provenance: str
    target: str | None = None
    predicate: str | None = None


Probe = Callable[[str, int], Sequence[Mapping[str, Any]]]
AnswerabilityValidator = Callable[[str, str], Mapping[str, Any]]


def load_socratic_config(path: str | Path | None = None) -> SocraticConfig:
    config_path = Path(path or os.getenv("QA_SOCRATIC_CONFIG", DEFAULT_CONFIG_PATH))
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    weights = payload.get("ranking_weights", {})
    config = SocraticConfig(
        enabled=bool(payload.get("enabled", True)),
        max_followups=int(payload.get("max_followups", 3)),
        max_internal_candidates=int(payload.get("max_internal_candidates", 10)),
        min_answerability_score=float(payload.get("min_answerability_score", 0.62)),
        min_topic_relevance=float(payload.get("min_topic_relevance", 0.50)),
        avoid_same_relation=bool(payload.get("avoid_same_relation", True)),
        allow_bm25_probe=bool(payload.get("allow_bm25_probe", True)),
        probe_top_k=int(payload.get("probe_top_k", 5)),
        max_bm25_probes=int(payload.get("max_bm25_probes", 3)),
        max_context_passages=int(payload.get("max_context_passages", 12)),
        duplicate_similarity_threshold=float(payload.get("duplicate_similarity_threshold", 0.72)),
        one_hop_only=bool(payload.get("one_hop_only", True)),
        prefer_relation_diversity=bool(payload.get("prefer_relation_diversity", True)),
        answerability_weight=float(weights.get("answerability", 0.45)),
        relevance_weight=float(weights.get("relevance", 0.25)),
        novelty_weight=float(weights.get("novelty", 0.2)),
        diversity_weight=float(weights.get("diversity", 0.1)),
    )
    if not 1 <= config.max_followups <= 3:
        raise ValueError("socratic.max_followups must be within 1..3")
    if not 1 <= config.probe_top_k <= 20:
        raise ValueError("socratic.probe_top_k must be within 1..20")
    if not 1 <= config.max_internal_candidates <= 50:
        raise ValueError("socratic.max_internal_candidates must be within 1..50")
    if not 0 <= config.max_bm25_probes <= config.max_internal_candidates:
        raise ValueError("socratic.max_bm25_probes must be within 0..max_internal_candidates")
    if config.max_context_passages < 1:
        raise ValueError("socratic.max_context_passages must be positive")
    for name in ("min_answerability_score", "min_topic_relevance", "duplicate_similarity_threshold"):
        if not 0.0 <= float(getattr(config, name)) <= 1.0:
            raise ValueError(f"socratic.{name} must be within 0..1")
    weight_total = (
        config.answerability_weight
        + config.relevance_weight
        + config.novelty_weight
        + config.diversity_weight
    )
    if not math.isclose(weight_total, 1.0, abs_tol=1e-6):
        raise ValueError("socratic ranking weights must sum to 1")
    return config


SOCRATIC_CONFIG = load_socratic_config()


def _fold(text: Any) -> str:
    value = str(text or "").casefold().replace("đ", "d")
    value = unicodedata.normalize("NFD", value)
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", value).strip()


def normalize_question(text: str) -> str:
    return " ".join(re.findall(r"[\w%]+", _fold(text), flags=re.UNICODE))


_QUESTION_STOPWORDS = {
    "ai", "bao", "cai", "co", "cua", "duoc", "gi", "la", "nao", "nhung",
    "o", "tai", "the", "trong", "vao", "ve", "vi", "voi",
}

_TOPIC_STOPWORDS = _QUESTION_STOPWORDS | {
    "cho", "dang", "day", "den", "du", "giu", "hon", "khi", "lai", "mot",
    "nay", "nhieu", "nhu", "nhung", "qua", "ra", "sau", "theo", "thi", "tu",
    "tung", "va", "van", "viec",
}


def _question_tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_question(text).split()
        if len(token) > 1 and token not in _QUESTION_STOPWORDS
    }


def _ordered_question_tokens(text: str) -> list[str]:
    return [
        token
        for token in normalize_question(text).split()
        if len(token) > 1 and token not in _QUESTION_STOPWORDS
    ]


def question_similarity(left: str, right: str) -> float:
    left_normalized = normalize_question(left)
    right_normalized = normalize_question(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    left_tokens = _question_tokens(left)
    right_tokens = _question_tokens(right)
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    return max(jaccard, sequence)


_RELATION_EQUIVALENTS = {
    "ENTITY": "IDENTITY",
    "DEFINITION": "IDENTITY",
    "GENERIC_LOCATION": "LOCATION",
}


def _canonical_relation(value: Any) -> str:
    relation = str(value or "GENERAL").strip().upper()
    return _RELATION_EQUIVALENTS.get(relation, relation)


def _semantic_value(semantics: Any, key: str) -> Any:
    if isinstance(semantics, Mapping):
        return semantics.get(key)
    return getattr(semantics, key, None)


_WEAK_SUBJECTS = {
    "", "dieu nay", "su kien nay", "viec nay", "no", "ho", "nguoi nay",
    "doi tuong", "tai lieu", "cau hoi",
}


def _clean_subject(value: Any) -> str | None:
    subject = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n?!.,;:-\"")
    words = subject.split()
    folded_words = [_fold(word).strip("?!.,;:-\"") for word in words]
    interrogative_sequences = (
        ("la", "ai"),
        ("la", "gi"),
        ("o", "dau"),
        ("tai", "dau"),
        ("khi", "nao"),
        ("nhu", "the", "nao"),
    )
    cut = len(words)
    for index in range(len(folded_words)):
        for sequence in interrogative_sequences:
            if tuple(folded_words[index : index + len(sequence)]) == sequence:
                cut = min(cut, index)
    subject = " ".join(words[:cut]).strip(" \t\r\n?!.,;:-\"")
    if not subject or len(subject) > 120 or _fold(subject) in _WEAK_SUBJECTS:
        return None
    if len(_question_tokens(subject)) == 0:
        return None
    return subject


def _focus_subject(value: Any) -> str | None:
    """Reduce nominal question frames to the entity the learner is exploring."""

    subject = _clean_subject(value)
    if subject is None:
        return None
    words = subject.split()
    folded_words = [_fold(word).strip("?!.,;:-\"") for word in words]
    for prefix in (
        ("chuc", "nang", "cua"),
        ("vai", "tro", "cua"),
        ("muc", "dich", "cua"),
        ("dac", "diem", "cua"),
        ("dac", "trung", "cua"),
        ("y", "nghia", "cua"),
        ("lich", "su", "cua"),
        ("nguon", "goc", "cua"),
        ("cau", "tao", "cua"),
        ("thanh", "phan", "cua"),
        ("vi", "tri", "cua"),
        ("nguyen", "nhan", "cua"),
        ("hau", "qua", "cua"),
    ):
        if tuple(folded_words[: len(prefix)]) == prefix:
            focused = _clean_subject(" ".join(words[len(prefix) :]))
            if focused:
                return focused
    return subject


def _infer_subject(question: str) -> str | None:
    folded = _fold(question).strip(" ?!.,;:")
    patterns = (
        r"^(?P<subject>.+?)\s+(?:la ai|la gi)$",
        r"^(?P<subject>.+?)\s+(?:sinh|chao doi|ra doi|mat|qua doi)",
        r"^(?P<subject>.+?)\s+(?:dien ra|xay ra|no ra|nam|toa lac)",
        r"^(?:vi sao|tai sao)\s+(?P<subject>.+?)\s+(?:xay ra|phat trien|gia tang|suy giam)",
        r"^(?:muc dich cua)\s+(?P<subject>.+?)\s+(?:la gi)?$",
    )
    for pattern in patterns:
        match = re.search(pattern, folded)
        if match:
            # Folding preserves word order but not original offsets after Unicode decomposition.
            # The folded subject is still suitable as a retrieval query when API metadata is absent.
            return _clean_subject(match.group("subject"))
    return None


def _subject_coverage(subject: str, text: str) -> float:
    subject_tokens = _ordered_question_tokens(subject)
    if not subject_tokens:
        return 0.0
    text_tokens = _ordered_question_tokens(text)
    for index in range(0, len(text_tokens) - len(subject_tokens) + 1):
        if text_tokens[index : index + len(subject_tokens)] == subject_tokens:
            return 1.0
    overlap = len(set(subject_tokens) & set(text_tokens)) / len(set(subject_tokens))
    # Dispersed token matches (for example in unrelated names/phrases) are not
    # sufficient evidence for a multi-word subject.
    return min(0.6, overlap) if len(subject_tokens) > 1 else overlap


def _has_exact_surface_subject(subject: str, text: str) -> bool:
    """Preserve Vietnamese diacritics when binding ambiguous one-token subjects."""

    normalized_subject = re.sub(r"\s+", " ", subject.casefold()).strip()
    if not normalized_subject:
        return False
    return bool(
        re.search(
            rf"(?<!\w){re.escape(normalized_subject)}(?!\w)",
            text.casefold(),
            flags=re.UNICODE,
        )
    )


def _topic_anchor_tokens(text: str, subject: str) -> set[str]:
    subject_tokens = _question_tokens(subject)
    return {
        token
        for token in normalize_question(text).split()
        if len(token) > 2 and token not in _TOPIC_STOPWORDS and token not in subject_tokens
    }


def _has_topic_continuity(anchor_text: str, candidate_text: str, subject: str) -> bool:
    anchor_tokens = _topic_anchor_tokens(anchor_text, subject)
    candidate_tokens = _topic_anchor_tokens(candidate_text, subject)
    if not anchor_tokens or not candidate_tokens:
        return False
    shared = anchor_tokens & candidate_tokens
    smaller = min(len(anchor_tokens), len(candidate_tokens))
    return len(shared) >= 2 and len(shared) / max(1, smaller) >= 0.08


def _token_coverage(value: str, text: str) -> float:
    value_tokens = _question_tokens(value)
    if not value_tokens:
        return 0.0
    text_tokens = _question_tokens(text)
    return len(value_tokens & text_tokens) / len(value_tokens)


def _restore_subject_surface(subject: str, passages: Iterable[Any]) -> str:
    folded_subject = _fold(subject)
    if not folded_subject:
        return subject
    for passage in passages:
        text = str(_passage_value(passage, "text", ""))
        folded_text = _fold(text)
        match = re.search(rf"(?<!\w){re.escape(folded_subject)}(?!\w)", folded_text)
        if match:
            return text[match.start() : match.end()]
    return subject


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.UNICODE) is not None


_DATE = r"(?:\b(?:ngay|thang|nam|the ky|thap nien|giai doan|thoi ky)\b|\b(?:1\d{3}|20\d{2})\b)"
_BIRTH = r"\b(?:sinh|chao doi|ra doi)\b"
_DEATH = r"\b(?:mat|qua doi|tu tran)\b"
_EVENT = r"\b(?:dien ra|xay ra|no ra|khoi nghia|cach mang|chien tranh|hoi nghi|phong trao)\b"
_ACTIVITY = r"\b(?:tham gia|lanh dao|chi huy|ky ket|dam phan|sang lap|thanh lap|to chuc|hoat dong)\b"
_OBJECT_LOCATION = r"\b(?:nam|toa lac|dat tai|thuoc)\b.{0,70}\b(?:tai|o|mien|tinh|thanh pho|quoc gia|khu vuc)\b"
_CONSEQUENCE = r"\b(?:dan den|gay ra|ket qua la|hau qua)\b"
_CONTEXT = r"\b(?:trong boi canh|boi canh|trong thoi ky|trong giai doan)\b"
_COMPARISON = r"\b(?:khac voi|khac biet|so voi|tuong dong)\b"
_TEMPORAL_ACTION = (
    r"\b(?:duoc thanh lap|thanh lap|duoc xay dung|xay dung|bat dau|ket thuc|"
    r"bi bai bo|bai bo|tro thanh|ra mat|phat hien|duoc cong bo|ky ket)\b"
)
_HEIGHT_ATTRIBUTE = r"\b(?:co do cao|do cao(?: la)?|cao(?: la)?)\b.{0,28}\b\d+(?:[,.]\d+)?\s*(?:m|met|km)\b"
_SPATIAL_PROCESS = (
    r"\b(?:sinh ra|moc ra|phat trien|xuat hien|phan bo|sinh song|"
    r"duoc trong|duoc tim thay)\b.{0,70}\b(?:tai|o)\b"
)
_LOW_INFORMATION_DETAIL = (
    r"\b(?:duoc nhac den|duoc de cap|de cap den)\b|"
    r"\bnoi ve\b.{0,45}\b(?:tai lieu|van ban|doan van|cau hoi)\b"
)
_COREFERENCE_MARKER = r"\b(?:ong|ba|nguoi nay|nhan vat nay|vi nay)\b"


FOLLOWUP_RELATIONS: dict[str, dict[str, Any]] = {
    "BIRTH_TIME": {"question_type": "TIME", "diversity_group": "TIME"},
    "DEATH_TIME": {"question_type": "TIME", "diversity_group": "TIME"},
    "EVENT_TIME": {"question_type": "TIME", "diversity_group": "TIME"},
    "BIRTH_LOCATION": {"question_type": "LOCATION", "diversity_group": "LOCATION"},
    "DEATH_LOCATION": {"question_type": "LOCATION", "diversity_group": "LOCATION"},
    "EVENT_LOCATION": {"question_type": "LOCATION", "diversity_group": "LOCATION"},
    "OBJECT_LOCATION": {"question_type": "LOCATION", "diversity_group": "LOCATION"},
    "PROCESS_LOCATION": {"question_type": "LOCATION", "diversity_group": "LOCATION"},
    "CAUSE": {"question_type": "GENERAL", "diversity_group": "CAUSE"},
    "PURPOSE": {"question_type": "GENERAL", "diversity_group": "PURPOSE"},
    "CONSEQUENCE": {"question_type": "GENERAL", "diversity_group": "CONSEQUENCE"},
    "ROLE": {"question_type": "GENERAL", "diversity_group": "ROLE"},
    "EVENT": {"question_type": "ENTITY", "diversity_group": "EVENT"},
    "ATTRIBUTE": {"question_type": "NUMBER", "diversity_group": "ATTRIBUTE"},
    "CONTEXT": {"question_type": "GENERAL", "diversity_group": "CONTEXT"},
    "COMPARISON": {"question_type": "GENERAL", "diversity_group": "COMPARISON"},
    "IDENTITY": {"question_type": "DEFINITION", "diversity_group": "IDENTITY"},
    "EVIDENCE_DETAIL": {"question_type": "GENERAL", "diversity_group": "DETAIL"},
}


QUESTION_TEMPLATES: dict[str, str] = {
    "BIRTH_TIME": "{subject} sinh vào thời gian nào?",
    "DEATH_TIME": "{subject} qua đời vào thời gian nào?",
    "EVENT_TIME": "{target} vào thời gian nào?",
    "BIRTH_LOCATION": "{subject} sinh ở đâu?",
    "DEATH_LOCATION": "{subject} qua đời ở đâu?",
    "EVENT_LOCATION": "{target} diễn ra ở đâu?",
    "OBJECT_LOCATION": "{subject} nằm ở đâu?",
    "PROCESS_LOCATION": "{subject} {target} ở đâu?",
    "CAUSE": "Vì sao {target}?",
    "PURPOSE": "{target} nhằm mục đích gì?",
    "CONSEQUENCE": "{target} dẫn đến kết quả gì?",
    "ROLE": "{subject} từng giữ những chức vụ hoặc vai trò nào?",
    "EVENT": "{subject} tham gia hoặc thực hiện những hoạt động nào?",
    "ATTRIBUTE": "{subject} có độ cao bao nhiêu?",
    "CONTEXT": "{subject} diễn ra trong bối cảnh nào?",
    "COMPARISON": "Tài liệu nêu điểm khác biệt nào liên quan đến {subject}?",
    "IDENTITY": "{subject} là gì?",
    "EVIDENCE_DETAIL": "Ngoài nội dung vừa trả lời, tài liệu còn cho biết điều gì về {subject}?",
}


def _has_date(sentence: str) -> bool:
    original = sentence.casefold()
    return bool(
        re.search(
            r"\b(?:ngày|tháng|năm|thế kỷ|thập niên|giai đoạn|thời kỳ)\b|"
            r"\b(?:1\d{3}|20\d{2})\b|"
            r"\b(?:ngay|thang|the ky|thap nien|giai doan|thoi ky)\b|"
            r"\bnam\s+(?:1\d{3}|20\d{2})\b",
            original,
        )
    )


def _has_explicit_temporal_value(sentence: str) -> bool:
    folded = _fold(sentence)
    return bool(
        re.search(r"\b(?:1\d{3}|20\d{2})\b", folded)
        or re.search(r"\b(?:ngay|thang)\s+\d{1,2}\b", folded)
        or re.search(r"\bthe ky\s+(?:\d+|[ivxlcdm]+)\b", folded)
        or re.search(r"\bthap nien\s+\d+\b", folded)
    )


def _has_birth_marker(sentence: str) -> bool:
    folded = _fold(sentence)
    return bool(
        re.search(r"\b(?:chao doi|ra doi)\b", folded)
        or re.search(r"(?<!\bsan )\bsinh\b(?!\s+vien\b)", folded)
    )


def _has_death_marker(sentence: str) -> bool:
    original = sentence.casefold()
    return bool(
        re.search(r"\b(?:mất|qua đời|từ trần)\b", original)
        or re.search(r"\b(?:mat|qua doi|tu tran)\b", original)
    )


def _has_cause_marker(sentence: str) -> bool:
    original = sentence.casefold()
    return bool(
        re.search(
            r"\b(?:bởi vì|vì|do|bởi|nguyên nhân|bắt nguồn từ|xuất phát từ)\b|"
            r"\b(?:boi vi|nguyen nhan|bat nguon tu|xuat phat tu)\b",
            original,
        )
    )


def _has_purpose_marker(sentence: str) -> bool:
    original = sentence.casefold()
    return bool(
        re.search(
            r"\b(?:nhằm|với mục đích|với mục tiêu)\b(?=\s+\w)|"
            r"(?<!triệt )\bđể\b(?=\s+\w)|"
            r"\b(?:nham|voi muc dich|voi muc tieu)\b(?=\s+\w)",
            original,
        )
    )


def _has_object_location_marker(sentence: str) -> bool:
    original = sentence.casefold()
    return bool(
        re.search(
            r"\b(?:nằm|tọa lạc|đặt tại|thuộc)\b.{0,80}"
            r"\b(?:tại|ở|miền|tỉnh|thành phố|quốc gia|khu vực)\b|"
            r"\b(?:toa lac|dat tai|thuoc)\b.{0,80}"
            r"\b(?:tai|o|mien|tinh|thanh pho|quoc gia|khu vuc)\b",
            original,
        )
    )


def _has_location_marker(sentence: str) -> bool:
    original = sentence.casefold()
    return bool(
        re.search(r"\b(?:tại|ở)\s+[\w]", original)
        or re.search(r"\b(?:tai|o)\s+[\w]", original)
    )


def _subject_is_event(subject: str) -> bool:
    return _has(
        _fold(subject),
        r"\b(?:su kien|cach mang|chien tranh|hoi nghi|phong trao|khoi nghia|cuoc chien)\b",
    )


def _subject_near_pattern(subject: str, folded_sentence: str, pattern: str, distance: int = 45) -> bool:
    folded_subject = _fold(subject)
    if not folded_subject:
        return False
    return bool(
        re.search(
            rf"(?:{re.escape(folded_subject)}.{{0,{distance}}}{pattern}|"
            rf"{pattern}.{{0,{distance}}}{re.escape(folded_subject)})",
            folded_sentence,
        )
    )


def _relation_is_bound_to_subject(
    subject: str,
    folded_sentence: str,
    pattern: str,
    subject_match: str,
    *,
    distance: int = 55,
) -> bool:
    if subject_match == "COREFERENCE_SUBJECT":
        return _has(folded_sentence, pattern)
    if subject_match != "DIRECT_SUBJECT":
        return False
    return _subject_near_pattern(subject, folded_sentence, pattern, distance=distance)


def _subject_precedes_pattern(
    subject: str,
    folded_sentence: str,
    pattern: str,
    subject_match: str,
    *,
    distance: int = 60,
) -> bool:
    if subject_match == "COREFERENCE_SUBJECT":
        return _has(folded_sentence, pattern)
    if subject_match != "DIRECT_SUBJECT":
        return False
    return bool(
        re.search(
            rf"{re.escape(_fold(subject))}.{{0,{distance}}}{pattern}",
            folded_sentence,
        )
    )


def _usable_clause_target(subject: str, clause: str | None) -> str | None:
    if not clause or _subject_coverage(subject, clause) < 0.75:
        return None
    cleaned = clause.strip(" \t\r\n,;:-\"")
    if not cleaned or len(cleaned) > 130 or cleaned.count(",") > 2:
        return None
    if re.search(r"\b(?:cua|tai|o|tu|den|vao|rang)\s*$", _fold(cleaned)):
        return None
    return cleaned


def _looks_like_person_subject(subject: str, folded_context: str) -> bool:
    folded_subject = _fold(subject)
    if _has(
        folded_subject,
        r"\b(?:chu nghia|thanh pho|dia diem|phan mem|cong trinh|su kien|cach mang|"
        r"chien tranh|hoi nghi|phong trao|cuoc chien)\b",
    ):
        return False
    content_tokens = _question_tokens(subject)
    if len(content_tokens) >= 2:
        return True
    person_cue = (
        r"(?:ong|ba|tong thong|thu tuong|chu tich|hoang de|nha vua|dai tuong|tuong)"
    )
    return bool(
        re.search(
            rf"(?:\b{person_cue}\b.{{0,18}}{re.escape(folded_subject)}|"
            rf"{re.escape(folded_subject)}.{{0,55}}\b{person_cue}\b)",
            folded_context,
        )
    )


def _role_is_bound_to_subject(
    subject: str,
    folded_sentence: str,
    subject_match: str,
) -> bool:
    # Avoid accent-fold collisions such as "nhà thơ"/"nhà thờ" and "tướng"/"tượng".
    role_noun = r"(?:co van|lanh dao|giao su|nha van|nha viet kich|thu tuong|tong thong|chu tich|bo truong|dai tuong)"
    predicate = r"(?:voi tu cach|giu chuc|dam nhiem|duoc bau|duoc bo nhiem|tro thanh|la)"
    folded_subject = re.escape(_fold(subject))
    if subject_match == "COREFERENCE_SUBJECT":
        return bool(re.search(rf"\b(?:ong|ba|nguoi nay|nhan vat nay|vi nay)\b.{{0,65}}(?:{predicate}.{{0,30}})?{role_noun}\b", folded_sentence))
    if subject_match != "DIRECT_SUBJECT":
        return False
    return bool(
        re.search(
            rf"{folded_subject}.{{0,95}}{predicate}\s+(?:(?:mot|cac|nhung|nguoi)\s+)?{role_noun}\b",
            folded_sentence,
        )
        or re.search(rf"\b{role_noun}\b.{{0,20}}{folded_subject}", folded_sentence)
        or re.search(rf"{folded_subject}.{{0,45}}\b(?:giu chuc|dam nhiem|duoc bau|duoc bo nhiem)\b", folded_sentence)
    )


def _activity_is_bound_to_subject(
    subject: str,
    folded_sentence: str,
    subject_match: str,
) -> bool:
    activity = rf"(?:{_ACTIVITY}|\b(?:duoc gui|gui den|cu den|tham du|thuc hien)\b)"
    if subject_match == "COREFERENCE_SUBJECT":
        return bool(
            re.search(
                rf"\b(?:ong|ba|ho|nguoi nay|nhan vat nay|vi nay)\b.{{0,55}}{activity}",
                folded_sentence,
            )
        )
    if subject_match != "DIRECT_SUBJECT":
        return False
    return bool(
        re.search(
            rf"{re.escape(_fold(subject))}.{{0,55}}{activity}",
            folded_sentence,
        )
    )


def _clause_before(sentence: str, folded_sentence: str, pattern: str) -> str | None:
    match = re.search(pattern, folded_sentence)
    if not match:
        return None
    # _fold can alter code-point offsets for accented text, so use a conservative word split.
    marker_words = match.group(0).split()
    original_words = sentence.split()
    folded_words = _fold(sentence).split()
    for index in range(len(folded_words)):
        if folded_words[index : index + len(marker_words)] == marker_words:
            clause = " ".join(original_words[:index]).strip(" ,;:-")
            return clause if 3 <= len(clause) <= 150 else None
    return None


def _clause_before_original(sentence: str, pattern: str) -> str | None:
    match = re.search(pattern, sentence.casefold())
    if not match:
        return None
    prefix = sentence[: match.start()].strip(" ,;:-")
    boundary = max(prefix.rfind("."), prefix.rfind(";"), prefix.rfind(":"))
    clause = prefix[boundary + 1 :].strip(" ,;:-")
    return clause if 3 <= len(clause) <= 160 else None


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def _upper_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _spatial_process_predicate(folded_sentence: str) -> str:
    for folded_predicate, display_predicate in (
        ("duoc tim thay", "được tìm thấy"),
        ("duoc trong", "được trồng"),
        ("sinh ra", "sinh ra"),
        ("moc ra", "mọc ra"),
        ("phat trien", "phát triển"),
        ("xuat hien", "xuất hiện"),
        ("phan bo", "phân bố"),
        ("sinh song", "sinh sống"),
    ):
        if re.search(rf"\b{folded_predicate}\b", folded_sentence):
            return display_predicate
    return "xuất hiện"


def _question_semantics_for_subject(subject: str) -> QuestionSemantics:
    return QuestionSemantics(
        question_type=["GENERAL"],
        relation="GENERAL",
        subject=subject,
        predicate=None,
        target=None,
        modifier=None,
        expected_answer_type=["GENERAL"],
    )


def _subject_match_level(
    subject: str,
    sentence: str,
    context: str,
    passage_title: str,
) -> tuple[str, float, str]:
    folded_sentence = _fold(sentence)
    if (
        len(_question_tokens(subject)) == 1
        and not _has_exact_surface_subject(subject, sentence)
        and not _has(folded_sentence, _COREFERENCE_MARKER)
    ):
        return "NO_SUBJECT_MATCH", 0.0, "ONE_TOKEN_SURFACE_MISMATCH"
    consistency = score_subject_consistency(
        _question_semantics_for_subject(subject),
        sentence,
        context=context,
    )
    if consistency.status == SemanticStatus.VALID.value:
        if consistency.reason == "PREVIOUS_SENTENCE_COREFERENCE":
            return "COREFERENCE_SUBJECT", consistency.score, consistency.reason
        return "DIRECT_SUBJECT", consistency.score, consistency.reason

    if _has(folded_sentence, _COREFERENCE_MARKER):
        sentences = split_sentences(context)
        for index, item in enumerate(sentences):
            if normalize_question(item) != normalize_question(sentence) or index == 0:
                continue
            if _subject_coverage(subject, sentences[index - 1]) >= 0.75:
                return "COREFERENCE_SUBJECT", 0.80, "PREVIOUS_SENTENCE_COREFERENCE"
    title_coverage = _subject_coverage(subject, passage_title)
    if title_coverage >= 0.75 and _has(folded_sentence, _COREFERENCE_MARKER):
        return "COREFERENCE_SUBJECT", 0.76, "TITLE_ANCHORED_COREFERENCE"
    return "NO_SUBJECT_MATCH", max(0.0, consistency.score), consistency.reason


def _topic_relevance_score(
    *,
    subject_score: float,
    passage_relevance: float,
    provenance: str,
    subject: str,
    sentence: str,
    answer: str,
) -> float:
    provenance_score = 1.0 if provenance == "selected" else 0.65
    entity_overlap = max(
        _subject_coverage(subject, sentence),
        min(1.0, _token_coverage(answer, sentence)) if answer else 0.0,
    )
    return min(
        1.0,
        0.45 * subject_score
        + 0.25 * passage_relevance
        + 0.20 * provenance_score
        + 0.10 * entity_overlap,
    )


def _is_additional_evidence(subject: str, sentence: str, answer: str) -> bool:
    """Return true only for a grounded subject fact that adds information."""

    if _subject_coverage(subject, sentence) < 0.75:
        return False
    if _has(_fold(sentence), _LOW_INFORMATION_DETAIL):
        return False
    sentence_tokens = _question_tokens(sentence)
    subject_tokens = _question_tokens(subject)
    if len(sentence_tokens - subject_tokens) < 3:
        return False
    normalized_answer = normalize_question(answer)
    normalized_sentence = normalize_question(sentence)
    if normalized_answer and normalized_answer in normalized_sentence:
        return False
    answer_tokens = _question_tokens(answer)
    if len(answer_tokens) >= 3:
        answer_coverage = len(answer_tokens & sentence_tokens) / len(answer_tokens)
        if answer_coverage >= 0.80:
            return False
    return True


def _evidence_detail_opportunity(
    subject: str,
    sentence: str,
    passage_id: str,
    relevance: float,
    *,
    context: str,
    passage_title: str,
    provenance: str,
    answer: str,
) -> FollowUpOpportunity | None:
    """Create a safe last-resort prompt from an additional evidenced fact."""

    subject_match, subject_score, subject_reason = _subject_match_level(
        subject,
        sentence,
        context,
        passage_title,
    )
    if subject_match == "NO_SUBJECT_MATCH" or not _is_additional_evidence(
        subject, sentence, answer
    ):
        return None
    topic_score = _topic_relevance_score(
        subject_score=subject_score,
        passage_relevance=relevance,
        provenance=provenance,
        subject=subject,
        sentence=sentence,
        answer=answer,
    )
    return FollowUpOpportunity(
        subject=subject,
        relation="EVIDENCE_DETAIL",
        question_type="GENERAL",
        source_passage_id=passage_id,
        evidence_sentence=sentence,
        generated_by=f"additional_grounded_fact:{subject_reason}",
        evidence_strength=0.70,
        relevance_score=relevance,
        topic_relevance_score=topic_score,
        subject_match=subject_match,
        subject_score=subject_score,
        provenance=provenance,
        predicate="chi tiết",
    )


def _temporal_target(subject: str, sentence: str, folded_sentence: str) -> str:
    target = _clause_before(sentence, folded_sentence, _DATE)
    if target and _subject_coverage(subject, target) >= 0.75:
        target = re.sub(r"\s+(?:vào|từ|đến)\s*$", "", target, flags=re.I).strip(" ,;:-")
        if 3 <= len(target) <= 145:
            return target
    folded_subject = _fold(subject)
    subject_start = folded_sentence.find(folded_subject)
    if subject_start >= 0:
        suffix = sentence[subject_start:].strip(" ,;:-")
        suffix = re.split(r"[.;!?]", suffix, maxsplit=1)[0].strip(" ,;:-")
        if 3 <= len(suffix) <= 145:
            return suffix
    return subject


def _discover_sentence_opportunities(
    subject: str,
    sentence: str,
    passage_id: str,
    relevance: float,
    *,
    context: str,
    passage_title: str,
    provenance: str,
    answer: str,
) -> list[FollowUpOpportunity]:
    folded = _fold(sentence)
    subject_match, subject_score, subject_reason = _subject_match_level(
        subject,
        sentence,
        context,
        passage_title,
    )
    topic_score = _topic_relevance_score(
        subject_score=subject_score,
        passage_relevance=relevance,
        provenance=provenance,
        subject=subject,
        sentence=sentence,
        answer=answer,
    )
    opportunities: list[FollowUpOpportunity] = []

    def add(
        relation: str,
        strength: float,
        generated_by: str,
        *,
        target: str | None = None,
        predicate: str | None = None,
    ) -> None:
        spec = FOLLOWUP_RELATIONS[relation]
        opportunities.append(
            FollowUpOpportunity(
                subject=subject,
                relation=relation,
                question_type=str(spec["question_type"]),
                source_passage_id=passage_id,
                evidence_sentence=sentence,
                generated_by=f"{generated_by}:{subject_reason}",
                evidence_strength=strength,
                relevance_score=relevance,
                topic_relevance_score=topic_score,
                subject_match=subject_match,
                subject_score=subject_score,
                provenance=provenance,
                target=target,
                predicate=predicate,
            )
        )

    has_birth = _has_birth_marker(sentence)
    has_death = _has_death_marker(sentence)
    has_date = _has_date(sentence)
    folded_subject = _fold(subject)
    biography_dates = re.search(
        rf"{re.escape(folded_subject)}\s*\([^)]*\b(?P<birth>1\d{{3}}|20\d{{2}})\b"
        rf"[^)]*(?:-|–|—|den)\s*[^)]*\b(?P<death>1\d{{3}}|20\d{{2}})\b[^)]*\)",
        folded,
    )

    subject_is_grounded = subject_match != "NO_SUBJECT_MATCH"
    birth_is_bound = _relation_is_bound_to_subject(
        subject, folded, _BIRTH, subject_match, distance=40
    )
    death_is_bound = _relation_is_bound_to_subject(
        subject, folded, _DEATH, subject_match, distance=40
    )
    person_event_compatible = _looks_like_person_subject(subject, _fold(context))
    if (
        subject_is_grounded
        and person_event_compatible
        and ((has_birth and birth_is_bound and has_date) or biography_dates)
    ):
        add("BIRTH_TIME", 0.93, "explicit_birth_time", predicate="sinh")
    if (
        subject_is_grounded
        and person_event_compatible
        and has_birth
        and birth_is_bound
        and _has_location_marker(sentence)
    ):
        add("BIRTH_LOCATION", 0.88, "explicit_birth_location", predicate="sinh")
    if (
        subject_is_grounded
        and person_event_compatible
        and ((has_death and death_is_bound and has_date) or biography_dates)
    ):
        add("DEATH_TIME", 0.92, "explicit_death_time", predicate="qua đời")
    if (
        subject_is_grounded
        and person_event_compatible
        and has_death
        and death_is_bound
        and _has_location_marker(sentence)
    ):
        add("DEATH_LOCATION", 0.86, "explicit_death_location", predicate="qua đời")

    event_is_bound = _relation_is_bound_to_subject(
        subject, folded, _EVENT, subject_match, distance=60
    )
    subject_event = _subject_is_event(subject) and event_is_bound
    temporal_action = _subject_precedes_pattern(
        subject, folded, _TEMPORAL_ACTION, subject_match, distance=65
    )
    if (
        subject_is_grounded
        and has_date
        and _has_explicit_temporal_value(sentence)
        and not has_birth
        and not has_death
        and (subject_event or temporal_action)
    ):
        target = subject if subject_event else _temporal_target(subject, sentence, folded)
        target = _usable_clause_target(subject, target)
        if target and (subject_event or normalize_question(target) != normalize_question(subject)):
            add("EVENT_TIME", 0.87, "evidenced_event_time", target=target, predicate="thời gian")
    if subject_is_grounded and subject_event and _has_location_marker(sentence):
        add("EVENT_LOCATION", 0.84, "evidenced_event_location", target=subject, predicate="địa điểm")
    if (
        subject_is_grounded
        and _has_object_location_marker(sentence)
        and _relation_is_bound_to_subject(
            subject, folded, _OBJECT_LOCATION, subject_match, distance=55
        )
    ):
        add("OBJECT_LOCATION", 0.86, "evidenced_object_location", predicate="nằm")

    if (
        subject_is_grounded
        and _has(folded, _SPATIAL_PROCESS)
        and _relation_is_bound_to_subject(
            subject, folded, _SPATIAL_PROCESS, subject_match, distance=55
        )
    ):
        add(
            "PROCESS_LOCATION",
            0.86,
            "evidenced_spatial_process",
            predicate=_spatial_process_predicate(folded),
        )

    if subject_is_grounded and _has_cause_marker(sentence):
        effect = _clause_before_original(
            sentence,
            r"\b(?:bởi vì|vì|do|bởi|nguyên nhân|bắt nguồn từ|xuất phát từ|"
            r"boi vi|nguyen nhan|bat nguon tu|xuat phat tu)\b",
        )
        target = _usable_clause_target(subject, effect)
        if target:
            add("CAUSE", 0.88, "evidenced_cause", target=target, predicate="nguyên nhân")

    if subject_is_grounded and _has_purpose_marker(sentence):
        action = _clause_before_original(
            sentence,
            r"\b(?:nhằm|để|với mục đích|với mục tiêu|nham|voi muc dich|voi muc tieu)\b",
        )
        target = _usable_clause_target(subject, action)
        if target:
            add("PURPOSE", 0.83, "evidenced_purpose", target=target, predicate="mục đích")

    if subject_is_grounded and _has(folded, _CONSEQUENCE):
        cause = _clause_before(sentence, folded, _CONSEQUENCE)
        target = _usable_clause_target(subject, cause)
        if target:
            add("CONSEQUENCE", 0.84, "evidenced_consequence", target=target, predicate="kết quả")

    if (
        subject_is_grounded
        and _has(folded, _CONTEXT)
        and (subject_event or _relation_is_bound_to_subject(
            subject, folded, _ACTIVITY, subject_match, distance=60
        ))
    ):
        add("CONTEXT", 0.77, "evidenced_context", predicate="bối cảnh")

    if (
        subject_is_grounded
        and person_event_compatible
        and _role_is_bound_to_subject(subject, folded, subject_match)
    ):
        add("ROLE", 0.85, "evidenced_role", predicate="vai trò")

    if (
        subject_is_grounded
        and person_event_compatible
        and _activity_is_bound_to_subject(subject, folded, subject_match)
    ):
        add("EVENT", 0.81, "evidenced_activity", predicate="hoạt động")

    if (
        subject_is_grounded
        and _has(folded, _HEIGHT_ATTRIBUTE)
        and _relation_is_bound_to_subject(
            subject, folded, _HEIGHT_ATTRIBUTE, subject_match, distance=45
        )
    ):
        add("ATTRIBUTE", 0.90, "evidenced_height", target="độ cao", predicate="độ cao")

    if (
        subject_is_grounded
        and _has(folded, _COMPARISON)
        and _relation_is_bound_to_subject(
            subject, folded, _COMPARISON, subject_match, distance=55
        )
    ):
        add("COMPARISON", 0.76, "evidenced_comparison", predicate="khác biệt")

    definition = _has(folded, rf"(?:^|[.;]\s*){re.escape(folded_subject)}\s+(?:la|duoc xem la)\b")
    if subject_is_grounded and definition:
        add("IDENTITY", 0.82, "evidenced_identity", predicate="là")

    return opportunities


def _collect_context_passages(
    selected_passage: Any | None,
    retrieved_passages: Iterable[Any],
    max_passages: int,
) -> list[Any]:
    passages: list[Any] = []
    seen_passage_ids: set[str] = set()
    if selected_passage is not None:
        passage_id = str(_passage_value(selected_passage, "passage_id", ""))
        if passage_id:
            passages.append(selected_passage)
            seen_passage_ids.add(passage_id)
    for passage in retrieved_passages:
        passage_id = str(_passage_value(passage, "passage_id", ""))
        if passage_id and passage_id not in seen_passage_ids:
            passages.append(passage)
            seen_passage_ids.add(passage_id)
        if len(passages) >= max_passages:
            break
    return passages


def discover_followup_opportunities(
    semantics: Any,
    selected_passage: Any | None,
    retrieved_passages: Iterable[Any],
    *,
    question: str = "",
    answer: str = "",
    config: SocraticConfig | None = None,
) -> list[FollowUpOpportunity]:
    active_config = config or SOCRATIC_CONFIG
    semantic_subject = _focus_subject(_semantic_value(semantics, "subject"))
    subject = semantic_subject or _infer_subject(question)
    if not subject:
        return []
    passages = _collect_context_passages(
        selected_passage,
        retrieved_passages,
        active_config.max_context_passages,
    )
    if not passages:
        return []
    if semantic_subject is None or semantic_subject == semantic_subject.casefold():
        subject = _restore_subject_surface(subject, passages)

    selected_id = str(_passage_value(selected_passage, "passage_id", "")) or None
    anchor_text = str(_passage_value(selected_passage, "text", "")).strip()
    if not anchor_text and passages:
        anchor_text = str(_passage_value(passages[0], "text", "")).strip()
    short_subject = len(_question_tokens(subject)) == 1
    opportunities: list[FollowUpOpportunity] = []
    detail_fallbacks: list[FollowUpOpportunity] = []
    seen: set[tuple[str, str, str, str]] = set()
    for passage_index, passage in enumerate(passages):
        passage_id = str(_passage_value(passage, "passage_id", ""))
        text = str(_passage_value(passage, "text", "")).strip()
        if not passage_id or not text:
            continue
        if (
            short_subject
            and selected_id
            and passage_id != selected_id
            and not _has_topic_continuity(anchor_text, text, subject)
        ):
            continue
        relevance = _passage_relevance(passage, passage_index, selected_id)
        provenance = "selected" if passage_id == selected_id else "retrieved"
        title = str(_passage_value(passage, "title", ""))
        for sentence in split_sentences(text):
            detail = _evidence_detail_opportunity(
                subject,
                sentence,
                passage_id,
                relevance,
                context=text,
                passage_title=title,
                provenance=provenance,
                answer=answer,
            )
            if detail is not None:
                detail_fallbacks.append(detail)
            for opportunity in _discover_sentence_opportunities(
                subject,
                sentence,
                passage_id,
                relevance,
                context=text,
                passage_title=title,
                provenance=provenance,
                answer=answer,
            ):
                signature = (
                    _canonical_relation(opportunity.relation),
                    normalize_question(opportunity.subject),
                    normalize_question(opportunity.target or opportunity.predicate or ""),
                    normalize_question(opportunity.evidence_sentence),
                )
                if signature not in seen:
                    seen.add(signature)
                    opportunities.append(opportunity)
    current_relation = _canonical_relation(_semantic_value(semantics, "relation"))
    has_novel_typed_opportunity = any(
        _canonical_relation(opportunity.relation) != current_relation
        for opportunity in opportunities
    )
    if not has_novel_typed_opportunity and detail_fallbacks:
        detail_fallbacks.sort(
            key=lambda item: (
                item.provenance == "selected",
                item.topic_relevance_score,
                item.relevance_score,
                item.evidence_strength,
            ),
            reverse=True,
        )
        opportunities.append(detail_fallbacks[0])
    opportunities.sort(
        key=lambda item: (
            item.evidence_strength,
            item.topic_relevance_score,
            item.relevance_score,
        ),
        reverse=True,
    )
    return opportunities


def _opportunity_question(opportunity: FollowUpOpportunity) -> str | None:
    relation = _canonical_relation(opportunity.relation)
    template = QUESTION_TEMPLATES.get(relation)
    if not template:
        return None
    subject = opportunity.subject.strip()
    target = (opportunity.target or subject).strip(" .?!,;:")
    if relation == "PROCESS_LOCATION":
        predicate = (opportunity.predicate or "xuất hiện").strip()
        return f"{_upper_first(subject)} {predicate} ở đâu?"
    if relation == "EVENT_TIME" and normalize_question(target) == normalize_question(subject):
        return f"{subject} diễn ra vào thời gian nào?"
    if relation == "CAUSE":
        target = _lower_first(target)
    return template.format(subject=subject, target=target)


def _opportunity_signature(opportunity: FollowUpOpportunity) -> tuple[str, str, str]:
    return (
        normalize_question(opportunity.subject),
        _canonical_relation(opportunity.relation),
        normalize_question(opportunity.target or opportunity.predicate or opportunity.subject),
    )


def _relation_diversity_group(relation: str | None) -> str:
    canonical = _canonical_relation(relation)
    return str(FOLLOWUP_RELATIONS.get(canonical, {}).get("diversity_group", canonical))


def _passage_value(passage: Any, key: str, default: Any = None) -> Any:
    if isinstance(passage, Mapping):
        return passage.get(key, default)
    return getattr(passage, key, default)


def _passage_relevance(passage: Any, index: int, selected_id: str | None) -> float:
    passage_id = str(_passage_value(passage, "passage_id", ""))
    if selected_id and passage_id == selected_id:
        return 1.0
    for key in ("relevance_score", "retrieval_score_normalized", "retrieval_score"):
        value = _passage_value(passage, key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return min(1.0, max(0.0, float(value)))
    return max(0.25, 0.85 - index * 0.06)


def _probe_opportunity(
    opportunity: FollowUpOpportunity,
    question: str,
    probe: Probe,
    config: SocraticConfig,
) -> tuple[FollowUpOpportunity | None, float, str | None]:
    started = time.perf_counter()
    try:
        hits = probe(question, config.probe_top_k)
    except Exception as error:
        return None, (time.perf_counter() - started) * 1000, str(error)
    latency_ms = (time.perf_counter() - started) * 1000
    if not hits:
        return None, latency_ms, None
    for hit in hits:
        passage_id = str(_passage_value(hit, "passage_id", ""))
        if passage_id == opportunity.source_passage_id:
            return replace(
                opportunity,
                relevance_score=max(opportunity.relevance_score, _passage_relevance(hit, 0, None)),
                topic_relevance_score=max(opportunity.topic_relevance_score, config.min_topic_relevance),
                provenance="bm25_probe",
            ), latency_ms, None
        supported = discover_followup_opportunities(
            {"subject": opportunity.subject},
            hit,
            [],
            question=question,
            config=config,
        )
        for candidate in supported:
            if _canonical_relation(candidate.relation) != _canonical_relation(opportunity.relation):
                continue
            if candidate.subject_match == "NO_SUBJECT_MATCH":
                continue
            return replace(candidate, provenance="bm25_probe"), latency_ms, None
    return None, latency_ms, None


def _empty_generation_debug(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "candidate_generation": {
            "generated": 0,
            "after_same_relation": 0,
            "after_visited_relation": 0,
            "after_duplicate": 0,
            "after_subject_gate": 0,
            "after_topic_drift": 0,
            "after_grounding": 0,
            "after_bm25_probe": 0,
            "after_qa_validation": 0,
            "after_ranking": 0,
            "final": 0,
        },
        "rejection_distribution": {},
        "probe": {"attempted": 0, "accepted": 0, "latency_ms": 0.0},
        "candidates": [],
    }


def _generate_followups_internal(
    question: str,
    answer: str | None,
    semantics: Any,
    selected_passage: Any | None,
    retrieved_passages: Iterable[Any],
    limit: int = 3,
    *,
    visited_relations: Iterable[str] = (),
    asked_questions: Iterable[str] = (),
    probe: Probe | None = None,
    answerability_validator: AnswerabilityValidator | None = None,
    config: SocraticConfig | None = None,
) -> tuple[list[FollowUpCandidate], dict[str, Any]]:
    """Generate corpus-grounded, one-hop follow-ups after the main QA answer.

    The function never invokes a Reader or an external model. Candidates originate from
    selected/retrieved corpus passages, then optionally undergo a lightweight retrieval probe.
    """

    generation_started = time.perf_counter()
    active_config = config or SOCRATIC_CONFIG
    if not active_config.enabled or not str(question or "").strip() or not str(answer or "").strip():
        return [], _empty_generation_debug("INPUT_NOT_ELIGIBLE")
    requested_limit = min(active_config.max_followups, max(0, int(limit or 0)))
    if requested_limit == 0:
        return [], _empty_generation_debug("LIMIT_ZERO")

    semantic_subject = _focus_subject(_semantic_value(semantics, "subject"))
    subject = semantic_subject or _infer_subject(question)
    if subject is None:
        return [], _empty_generation_debug("NO_SUBJECT")

    retrieved_list = list(retrieved_passages)
    passages = _collect_context_passages(
        selected_passage,
        retrieved_list,
        active_config.max_context_passages,
    )
    if not passages:
        debug = _empty_generation_debug("NO_SEMANTIC_OPPORTUNITY")
        debug.update({"subject": subject, "current_relation": "GENERAL"})
        return [], debug
    if semantic_subject is None or semantic_subject == semantic_subject.casefold():
        subject = _restore_subject_surface(subject, passages)

    current_relation = _canonical_relation(
        _semantic_value(semantics, "relation")
        or _semantic_value(semantics, "semantic_relation")
        or "GENERAL"
    )
    current_target = _clean_subject(
        _semantic_value(semantics, "target")
        or _semantic_value(semantics, "question_target")
    )
    current_predicate = _clean_subject(
        _semantic_value(semantics, "predicate")
        or _semantic_value(semantics, "question_predicate")
    )
    visited = {_canonical_relation(relation) for relation in visited_relations if relation}
    visited.discard(current_relation)
    prior_questions = list(dict.fromkeys([question, *(str(item) for item in asked_questions if item)]))

    opportunities = discover_followup_opportunities(
        {"subject": subject, "relation": current_relation},
        selected_passage,
        retrieved_list,
        question=question,
        answer=str(answer),
        config=active_config,
    )
    processable = opportunities[: active_config.max_internal_candidates]
    truncated = opportunities[active_config.max_internal_candidates :]
    counts = _empty_generation_debug("PENDING")["candidate_generation"]
    counts["generated"] = len(processable)
    traces: list[SocraticCandidateTrace] = []
    eligible: list[tuple[FollowUpCandidate, SocraticCandidateTrace, FollowUpOpportunity]] = []
    rejection_distribution: dict[str, int] = {}
    probe_attempted = 0
    probe_accepted = 0
    probe_latency_ms = 0.0

    def reject(trace: SocraticCandidateTrace, reason: str) -> None:
        trace.rejection_reason = reason
        traces.append(trace)
        rejection_distribution[reason] = rejection_distribution.get(reason, 0) + 1

    for opportunity in truncated:
        question_text = _opportunity_question(opportunity) or ""
        reject(
            SocraticCandidateTrace(
                question=question_text,
                relation=opportunity.relation,
                subject=opportunity.subject,
                source_passage_id=opportunity.source_passage_id,
                generated_by=opportunity.generated_by,
                evidence_sentence=opportunity.evidence_sentence,
                target=opportunity.target,
                predicate=opportunity.predicate,
                subject_match=opportunity.subject_match,
                subject_score=round(opportunity.subject_score, 6),
                topic_relevance_score=round(opportunity.topic_relevance_score, 6),
                relevance_score=round(opportunity.relevance_score, 6),
            ),
            "LOW_RANKING_SCORE",
        )

    seen_signatures: set[tuple[str, str, str]] = set()
    for opportunity in processable:
        question_text = _opportunity_question(opportunity)
        trace = SocraticCandidateTrace(
            question=question_text or "",
            relation=opportunity.relation,
            subject=opportunity.subject,
            source_passage_id=opportunity.source_passage_id,
            generated_by=opportunity.generated_by,
            evidence_sentence=opportunity.evidence_sentence,
            target=opportunity.target,
            predicate=opportunity.predicate,
            subject_match=opportunity.subject_match,
            subject_score=round(opportunity.subject_score, 6),
            topic_relevance_score=round(opportunity.topic_relevance_score, 6),
            relevance_score=round(opportunity.relevance_score, 6),
        )
        if not question_text:
            reject(trace, "RELATION_EVIDENCE_NOT_FOUND")
            continue

        canonical = _canonical_relation(opportunity.relation)
        same_relation = canonical == current_relation
        if same_relation and active_config.avoid_same_relation:
            question_duplicate = question_similarity(question_text, question) >= 0.93
            current_signature = normalize_question(current_target or current_predicate or subject)
            opportunity_signature = normalize_question(
                opportunity.target or opportunity.predicate or opportunity.subject
            )
            target_duplicate = bool(
                current_signature
                and opportunity_signature
                and (
                    current_signature == opportunity_signature
                    or question_similarity(current_signature, opportunity_signature)
                    >= active_config.duplicate_similarity_threshold
                )
            )
            if question_duplicate or target_duplicate or (not current_target and not current_predicate):
                reject(trace, "SAME_RELATION")
                continue
        counts["after_same_relation"] += 1

        # Relations are broad families, not unique conversation turns. A
        # different predicate inside a visited relation remains a valid next
        # step; asked_questions below prevents actual conversational loops.
        counts["after_visited_relation"] += 1

        signature = _opportunity_signature(opportunity)
        maximum_similarity = max(
            (question_similarity(question_text, previous) for previous in prior_questions),
            default=0.0,
        )
        surface_duplicate = maximum_similarity >= 0.93
        if signature in seen_signatures or surface_duplicate:
            reject(trace, "DUPLICATE_QUESTION")
            continue
        if any(
                _opportunity_signature(existing_opportunity) == signature
                or (
                    _canonical_relation(existing_opportunity.relation) == canonical
                    and question_similarity(question_text, candidate.question)
                    >= 0.93
                )
            for candidate, _candidate_trace, existing_opportunity in eligible
        ):
            reject(trace, "DUPLICATE_QUESTION")
            continue
        counts["after_duplicate"] += 1

        if opportunity.subject_match == "NO_SUBJECT_MATCH":
            reject(trace, "SUBJECT_RELEVANCE_LOW")
            continue
        counts["after_subject_gate"] += 1

        selected_bonus = 0.06 if opportunity.provenance == "selected" else 0.0
        answerability = min(
            1.0,
            0.58 * opportunity.evidence_strength
            + 0.22 * opportunity.subject_score
            + 0.14 * opportunity.topic_relevance_score
            + selected_bonus,
        )
        topic_pass = opportunity.topic_relevance_score >= active_config.min_topic_relevance
        grounding_pass = answerability >= active_config.min_answerability_score
        accepted_opportunity = opportunity
        tier = "TIER_1"

        if not (topic_pass and grounding_pass):
            can_probe = (
                active_config.allow_bm25_probe
                and probe is not None
                and probe_attempted < active_config.max_bm25_probes
            )
            if not can_probe:
                reject(trace, "TOPIC_DRIFT" if not topic_pass else "ANSWERABILITY_LOW")
                continue
            probe_attempted += 1
            supported, candidate_probe_latency, _probe_error = _probe_opportunity(
                opportunity,
                question_text,
                probe,
                active_config,
            )
            probe_latency_ms += candidate_probe_latency
            trace.probe_latency_ms = round(candidate_probe_latency, 3)
            if supported is None:
                reject(trace, "BM25_PROBE_FAILED")
                continue
            probe_accepted += 1
            accepted_opportunity = supported
            tier = "TIER_2_BM25"
            question_text = _opportunity_question(supported) or question_text
            trace.question = question_text
            trace.source_passage_id = supported.source_passage_id
            trace.evidence_sentence = supported.evidence_sentence
            trace.relevance_score = round(supported.relevance_score, 6)
            trace.topic_relevance_score = round(supported.topic_relevance_score, 6)
            answerability = min(
                1.0,
                0.58 * supported.evidence_strength
                + 0.22 * supported.subject_score
                + 0.14 * max(supported.topic_relevance_score, active_config.min_topic_relevance),
            )

        counts["after_topic_drift"] += 1
        counts["after_grounding"] += 1
        counts["after_bm25_probe"] += 1

        verification_method: str | None = None
        if answerability_validator is not None:
            try:
                verification = answerability_validator(
                    question_text,
                    accepted_opportunity.source_passage_id,
                )
            except Exception as error:
                trace.qa_verified = False
                trace.verification_rejection_reason = f"VALIDATOR_ERROR:{type(error).__name__}"
                reject(trace, "QA_ANSWERABILITY_FAILED")
                continue
            verified = bool(verification.get("verified") or verification.get("has_answer"))
            verified_source_id = str(verification.get("source_passage_id") or "")
            source_matches = verified_source_id == accepted_opportunity.source_passage_id
            trace.qa_verified = bool(verified and source_matches)
            trace.verification_method = str(verification.get("method") or "") or None
            trace.verification_rejection_reason = (
                str(verification.get("rejection_reason") or "") or None
            )
            if not trace.qa_verified:
                reject(trace, "QA_ANSWERABILITY_FAILED")
                continue
            verification_method = trace.verification_method
        else:
            trace.qa_verified = False
        counts["after_qa_validation"] += 1

        novelty = min(1.0, max(0.0, 1.0 - maximum_similarity))
        base_ranking = (
            active_config.answerability_weight * answerability
            + active_config.relevance_weight * accepted_opportunity.relevance_score
            + active_config.novelty_weight * novelty
        )
        candidate = FollowUpCandidate(
            question=question_text,
            subject=accepted_opportunity.subject,
            relation=accepted_opportunity.relation,
            question_type=accepted_opportunity.question_type,
            source_passage_id=accepted_opportunity.source_passage_id,
            answerability_score=round(answerability, 6),
            novelty_score=round(novelty, 6),
            relevance_score=round(accepted_opportunity.relevance_score, 6),
            ranking_score=round(base_ranking, 6),
            qa_verified=bool(answerability_validator is not None),
            verification_method=verification_method,
        )
        trace.answerability_score = candidate.answerability_score
        trace.novelty_score = candidate.novelty_score
        trace.ranking_score = candidate.ranking_score
        trace.tier = tier
        eligible.append((candidate, trace, accepted_opportunity))
        seen_signatures.add(signature)

    counts["after_ranking"] = len(eligible)
    selected_pairs: list[tuple[FollowUpCandidate, SocraticCandidateTrace, FollowUpOpportunity]] = []
    remaining = list(eligible)
    selected_groups: set[str] = set()
    while remaining and len(selected_pairs) < requested_limit:
        rescored: list[
            tuple[float, FollowUpCandidate, SocraticCandidateTrace, FollowUpOpportunity]
        ] = []
        for candidate, trace, opportunity in remaining:
            group = _relation_diversity_group(candidate.relation)
            diversity = 1.0 if group not in selected_groups else 0.0
            diversity_bonus = active_config.diversity_weight * diversity if active_config.prefer_relation_diversity else 0.0
            score = candidate.ranking_score + diversity_bonus
            rescored.append((score, candidate, trace, opportunity))
        _score, candidate, trace, opportunity = max(
            rescored,
            key=lambda item: (
                item[0],
                item[1].answerability_score,
                item[1].novelty_score,
            ),
        )
        candidate = replace(candidate, ranking_score=round(_score, 6))
        trace.ranking_score = candidate.ranking_score
        selected_pairs.append((candidate, trace, opportunity))
        selected_groups.add(_relation_diversity_group(candidate.relation))
        remaining = [item for item in remaining if id(item[1]) != id(trace)]

    selected_trace_ids = {id(trace) for _candidate, trace, _opportunity in selected_pairs}
    for _candidate, trace, _opportunity in eligible:
        if id(trace) in selected_trace_ids:
            trace.accepted = True
            trace.why_accepted = (
                f"{trace.tier}; {trace.subject_match}; relation evidence in "
                f"{trace.source_passage_id}"
            )
            traces.append(trace)
        else:
            reject(trace, "LOW_RANKING_SCORE")
    candidates = [candidate for candidate, _trace, _opportunity in selected_pairs]
    counts["final"] = len(candidates)

    if not opportunities:
        status = "NO_SEMANTIC_OPPORTUNITY"
    elif not candidates:
        status = "OPPORTUNITIES_FOUND_BUT_ALL_REJECTED"
    else:
        status = "FOLLOWUPS_GENERATED"
    relation_counts: dict[str, int] = {}
    for opportunity in opportunities:
        relation_counts[opportunity.relation] = relation_counts.get(opportunity.relation, 0) + 1
    elapsed_ms = (time.perf_counter() - generation_started) * 1000
    debug = {
        "status": status,
        "subject": subject,
        "current_relation": current_relation,
        "current_target": current_target,
        "current_predicate": current_predicate,
        "passages_scanned": len(passages),
        "semantic_opportunities": {
            "detected": len(opportunities),
            "processed": len(processable),
            "truncated": len(truncated),
            "by_relation": relation_counts,
        },
        "candidate_generation": counts,
        "rejection_distribution": rejection_distribution,
        "latency": {
            "tier_1_ms": round(max(0.0, elapsed_ms - probe_latency_ms), 3),
            "bm25_probe_ms": round(probe_latency_ms, 3),
            "total_ms": round(elapsed_ms, 3),
        },
        "probe": {
            "attempted": probe_attempted,
            "accepted": probe_accepted,
            "max_probes": active_config.max_bm25_probes,
            "latency_ms": round(probe_latency_ms, 3),
        },
        "candidates": [trace.to_dict() for trace in traces],
    }
    return candidates, debug


def generate_followups(
    question: str,
    answer: str | None,
    semantics: Any,
    selected_passage: Any | None,
    retrieved_passages: Iterable[Any],
    limit: int = 3,
    *,
    visited_relations: Iterable[str] = (),
    asked_questions: Iterable[str] = (),
    probe: Probe | None = None,
    answerability_validator: AnswerabilityValidator | None = None,
    config: SocraticConfig | None = None,
) -> list[FollowUpCandidate]:
    candidates, _debug = _generate_followups_internal(
        question,
        answer,
        semantics,
        selected_passage,
        retrieved_passages,
        limit,
        visited_relations=visited_relations,
        asked_questions=asked_questions,
        probe=probe,
        answerability_validator=answerability_validator,
        config=config,
    )
    return candidates


def generate_followup_response(
    payload: Mapping[str, Any],
    *,
    passage_lookup: Callable[[str], Mapping[str, Any] | None],
    probe: Probe | None = None,
    answerability_validator: AnswerabilityValidator | None = None,
    config: SocraticConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    question = str(payload.get("question", "")).strip()
    answer = payload.get("answer")
    try:
        requested_limit = int(payload.get("limit", 3) or 3)
    except (TypeError, ValueError) as error:
        raise ValueError("limit must be an integer") from error
    if not question:
        raise ValueError("question is required")
    if requested_limit < 1:
        raise ValueError("limit must be positive")

    selected_id = str(payload.get("selected_passage_id") or "").strip()
    retrieved_ids = payload.get("retrieved_passage_ids") or []
    if not isinstance(retrieved_ids, list):
        raise ValueError("retrieved_passage_ids must be an array")
    visited_relations = payload.get("visited_relations") or []
    asked_questions = payload.get("asked_questions") or []
    if not isinstance(visited_relations, list):
        raise ValueError("visited_relations must be an array")
    if not isinstance(asked_questions, list):
        raise ValueError("asked_questions must be an array")

    selected = passage_lookup(selected_id) if selected_id else None
    retrieved = [
        passage
        for passage_id in retrieved_ids
        if (passage := passage_lookup(str(passage_id))) is not None
    ]
    semantics = {
        "subject": payload.get("subject"),
        "relation": payload.get("relation") or payload.get("semantic_relation"),
        "question_type": payload.get("question_type"),
        "target": payload.get("target") or payload.get("question_target"),
        "predicate": payload.get("predicate") or payload.get("question_predicate"),
        "modifier": payload.get("modifier") or payload.get("question_modifier"),
    }
    candidates, debug = _generate_followups_internal(
        question,
        str(answer) if answer is not None else None,
        semantics,
        selected,
        retrieved,
        limit=requested_limit,
        visited_relations=visited_relations,
        asked_questions=asked_questions,
        probe=probe,
        answerability_validator=answerability_validator,
        config=config,
    )
    response = {
        "followups": [candidate.to_dict() for candidate in candidates],
        "processing_time_ms": int((time.perf_counter() - started) * 1000),
        "grounding": "selected_and_retrieved_corpus_passages",
        "probe": "bm25" if (config or SOCRATIC_CONFIG).allow_bm25_probe and probe else None,
        "answerability_gate": "qa_pipeline" if answerability_validator else "heuristic_only",
    }
    debug_requested = bool(payload.get("debug")) or _fold(os.getenv("QA_SOCRATIC_DEBUG")) in {
        "1", "true", "yes", "on",
    }
    if debug_requested:
        response["debug"] = debug
    return response


__all__ = [
    "AnswerabilityValidator",
    "FollowUpCandidate",
    "SocraticCandidateTrace",
    "SOCRATIC_CONFIG",
    "SocraticConfig",
    "FollowUpOpportunity",
    "FOLLOWUP_RELATIONS",
    "QUESTION_TEMPLATES",
    "discover_followup_opportunities",
    "generate_followup_response",
    "generate_followups",
    "load_socratic_config",
    "normalize_question",
    "question_similarity",
]
