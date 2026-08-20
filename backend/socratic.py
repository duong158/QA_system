from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from backend.chunking import split_sentences


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "socratic.json"


@dataclass(frozen=True)
class SocraticConfig:
    enabled: bool = True
    max_followups: int = 3
    min_answerability_score: float = 0.62
    avoid_same_relation: bool = True
    allow_bm25_probe: bool = True
    probe_top_k: int = 5
    max_context_passages: int = 12
    duplicate_similarity_threshold: float = 0.72
    one_hop_only: bool = True
    answerability_weight: float = 0.5
    relevance_weight: float = 0.3
    novelty_weight: float = 0.2


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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = payload.pop("question_type")
        return payload


@dataclass(frozen=True)
class _Draft:
    question: str
    subject: str
    relation: str
    question_type: str
    source_passage_id: str
    evidence_sentence: str
    evidence_strength: float
    relevance_score: float


Probe = Callable[[str, int], Sequence[Mapping[str, Any]]]


def load_socratic_config(path: str | Path | None = None) -> SocraticConfig:
    config_path = Path(path or os.getenv("QA_SOCRATIC_CONFIG", DEFAULT_CONFIG_PATH))
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    weights = payload.get("ranking_weights", {})
    config = SocraticConfig(
        enabled=bool(payload.get("enabled", True)),
        max_followups=int(payload.get("max_followups", 3)),
        min_answerability_score=float(payload.get("min_answerability_score", 0.62)),
        avoid_same_relation=bool(payload.get("avoid_same_relation", True)),
        allow_bm25_probe=bool(payload.get("allow_bm25_probe", True)),
        probe_top_k=int(payload.get("probe_top_k", 5)),
        max_context_passages=int(payload.get("max_context_passages", 12)),
        duplicate_similarity_threshold=float(payload.get("duplicate_similarity_threshold", 0.72)),
        one_hop_only=bool(payload.get("one_hop_only", True)),
        answerability_weight=float(weights.get("answerability", 0.5)),
        relevance_weight=float(weights.get("relevance", 0.3)),
        novelty_weight=float(weights.get("novelty", 0.2)),
    )
    if not 1 <= config.max_followups <= 3:
        raise ValueError("socratic.max_followups must be within 1..3")
    if not 1 <= config.probe_top_k <= 20:
        raise ValueError("socratic.probe_top_k must be within 1..20")
    if config.max_context_passages < 1:
        raise ValueError("socratic.max_context_passages must be positive")
    for name in ("min_answerability_score", "duplicate_similarity_threshold"):
        if not 0.0 <= float(getattr(config, name)) <= 1.0:
            raise ValueError(f"socratic.{name} must be within 0..1")
    weight_total = config.answerability_weight + config.relevance_weight + config.novelty_weight
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
_ROLE = r"\b(?:vai tro|chuc vu|giu chuc|dam nhiem|duoc bau|duoc bo nhiem|dai dien|lanh dao|chi huy|thu tuong|tong thong|chu tich|bo truong|tuong)\b"
_LOCATION = r"\b(?:tai|o)\s+[\w]"
_OBJECT_LOCATION = r"\b(?:nam|toa lac|dat tai|thuoc)\b.{0,70}\b(?:tai|o|mien|tinh|thanh pho|quoc gia|khu vuc)\b"
_CAUSE = r"\b(?:boi vi|do viec|bat nguon tu|xuat phat tu|nguyen nhan|vi|do|boi)\b"
_PURPOSE = r"\b(?:nham|voi muc dich|voi muc tieu|de)\b"
_CONSEQUENCE = r"\b(?:dan den|gay ra|ket qua la|hau qua)\b"
_CONTEXT = r"\b(?:trong boi canh|boi canh|trong thoi ky|trong giai doan)\b"
_COMPARISON = r"\b(?:khac voi|khac biet|so voi|tuong dong)\b"


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


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def _relation_drafts(subject: str, sentence: str, passage_id: str, relevance: float) -> list[_Draft]:
    folded = _fold(sentence)
    if _subject_coverage(subject, sentence) < 0.75:
        return []

    drafts: list[_Draft] = []

    def add(relation: str, question_type: str, question: str, strength: float) -> None:
        drafts.append(
            _Draft(
                question=question,
                subject=subject,
                relation=relation,
                question_type=question_type,
                source_passage_id=passage_id,
                evidence_sentence=sentence,
                evidence_strength=strength,
                relevance_score=relevance,
            )
        )

    has_birth = _has(folded, _BIRTH)
    has_death = _has(folded, _DEATH)
    has_date = _has(folded, _DATE)
    folded_subject = _fold(subject)
    biography_dates = re.search(
        rf"{re.escape(folded_subject)}\s*\([^)]*\b(?P<birth>1\d{{3}}|20\d{{2}})\b"
        rf"[^)]*(?:-|–|—|den)\s*[^)]*\b(?P<death>1\d{{3}}|20\d{{2}})\b[^)]*\)",
        folded,
    )

    if (has_birth and has_date) or biography_dates:
        add("BIRTH_TIME", "TIME", f"{subject} sinh vào thời gian nào?", 0.91)
    if has_birth and _has(folded, _LOCATION):
        add("BIRTH_LOCATION", "LOCATION", f"{subject} sinh ở đâu?", 0.88)
    if (has_death and has_date) or biography_dates:
        add("DEATH_TIME", "TIME", f"{subject} qua đời vào thời gian nào?", 0.90)
    if has_death and _has(folded, _LOCATION):
        add("DEATH_LOCATION", "LOCATION", f"{subject} qua đời ở đâu?", 0.86)

    subject_event = _has(folded, rf"{re.escape(folded_subject)}.{{0,45}}{_EVENT}")
    if subject_event and has_date and not has_birth and not has_death:
        add("EVENT_TIME", "TIME", f"{subject} diễn ra vào thời gian nào?", 0.86)
    if subject_event and _has(folded, _LOCATION):
        add("EVENT_LOCATION", "LOCATION", f"{subject} diễn ra ở đâu?", 0.84)
    if _has(folded, rf"{re.escape(folded_subject)}.{{0,35}}{_OBJECT_LOCATION}"):
        add("OBJECT_LOCATION", "LOCATION", f"{subject} nằm ở đâu?", 0.86)

    if _has(folded, _CAUSE) and _has(folded, rf"{re.escape(folded_subject)}.{{0,100}}\b(?:xay ra|phat trien|gia tang|suy giam|sup do|bung no|bi|duoc)\b"):
        effect = _clause_before(sentence, folded, _CAUSE)
        target = effect if effect and _subject_coverage(subject, effect) >= 0.75 else f"{subject} xảy ra"
        add("CAUSE", "GENERAL", f"Vì sao {_lower_first(target.rstrip('.?!'))}?", 0.88)

    if _has(folded, _PURPOSE) and _has(folded, rf"{re.escape(folded_subject)}.{{0,100}}(?:{_ACTIVITY}|\bduoc\b)"):
        action = _clause_before(sentence, folded, _PURPOSE)
        if action and _subject_coverage(subject, action) >= 0.75:
            add("PURPOSE", "GENERAL", f"{action.rstrip('.?!')} nhằm mục đích gì?", 0.81)

    if _has(folded, _CONSEQUENCE):
        cause = _clause_before(sentence, folded, _CONSEQUENCE)
        if cause and _subject_coverage(subject, cause) >= 0.75:
            add("CONSEQUENCE", "GENERAL", f"{cause.rstrip('.?!')} dẫn đến kết quả gì?", 0.84)

    if _has(folded, _CONTEXT) and (_has(folded, _EVENT) or _has(folded, _ACTIVITY)):
        add("CONTEXT", "GENERAL", f"{subject} diễn ra trong bối cảnh nào?", 0.77)

    if _has(folded, _ROLE):
        add("ROLE", "GENERAL", f"{subject} từng giữ những chức vụ hoặc vai trò nào?", 0.83)

    if _has(folded, _ACTIVITY):
        add("EVENT", "ENTITY", f"{subject} tham gia những sự kiện hoặc hoạt động nào?", 0.80)

    if _has(folded, _COMPARISON):
        add("COMPARISON", "GENERAL", f"Tài liệu nêu điểm khác biệt nào liên quan đến {subject}?", 0.76)

    definition = _has(folded, rf"(?:^|[.;]\s*){re.escape(folded_subject)}\s+(?:la|duoc xem la)\b")
    if definition:
        add("IDENTITY", "DEFINITION", f"{subject} là gì?", 0.82)

    return drafts


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


def _supports_relation(text: str, relation: str, subject: str) -> bool:
    return any(
        draft.relation == relation
        for draft in _relation_drafts(subject, text, "probe", 1.0)
    )


def _probe_supports(draft: _Draft, probe: Probe, config: SocraticConfig) -> bool:
    try:
        hits = probe(draft.question, config.probe_top_k)
    except Exception:
        # Probe failure is isolated from both QA and tier-1 grounded generation.
        return True
    if not hits:
        return False
    for hit in hits:
        passage_id = str(_passage_value(hit, "passage_id", ""))
        text = str(_passage_value(hit, "text", ""))
        if passage_id == draft.source_passage_id:
            return True
        if text and _supports_relation(text, draft.relation, draft.subject):
            return True
    return False


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
    config: SocraticConfig | None = None,
) -> list[FollowUpCandidate]:
    """Generate corpus-grounded, one-hop follow-ups after the main QA answer.

    The function never invokes a Reader or an external model. Candidates originate from
    selected/retrieved corpus passages, then optionally undergo a lightweight retrieval probe.
    """

    active_config = config or SOCRATIC_CONFIG
    if not active_config.enabled or not str(question or "").strip() or not str(answer or "").strip():
        return []
    requested_limit = min(active_config.max_followups, max(0, int(limit or 0)))
    if requested_limit == 0:
        return []

    subject = _clean_subject(_semantic_value(semantics, "subject")) or _infer_subject(question)
    if subject is None:
        return []

    current_relation = _canonical_relation(
        _semantic_value(semantics, "relation") or _semantic_value(semantics, "semantic_relation")
    )
    visited = {_canonical_relation(relation) for relation in visited_relations if relation}
    visited.add(current_relation)
    prior_questions = [question, *(str(item) for item in asked_questions if item)]

    passages: list[Any] = []
    seen_passage_ids: set[str] = set()
    for passage in ([selected_passage] if selected_passage is not None else []):
        passage_id = str(_passage_value(passage, "passage_id", ""))
        if passage_id and passage_id not in seen_passage_ids:
            passages.append(passage)
            seen_passage_ids.add(passage_id)
    for passage in retrieved_passages:
        passage_id = str(_passage_value(passage, "passage_id", ""))
        if passage_id and passage_id not in seen_passage_ids:
            passages.append(passage)
            seen_passage_ids.add(passage_id)
        if len(passages) >= active_config.max_context_passages:
            break
    if not passages:
        return []

    subject = _restore_subject_surface(subject, passages)

    selected_id = str(_passage_value(selected_passage, "passage_id", "")) or None
    drafts: list[_Draft] = []
    for passage_index, passage in enumerate(passages):
        passage_id = str(_passage_value(passage, "passage_id", ""))
        text = str(_passage_value(passage, "text", "")).strip()
        if not passage_id or not text:
            continue
        relevance = _passage_relevance(passage, passage_index, selected_id)
        # The final answer acts as a weak anchor for retrieved (non-selected) passages.
        # It never creates a relation by itself; relation evidence must still be in the corpus.
        if passage_id != selected_id:
            answer_alignment = _token_coverage(str(answer), text)
            relevance = min(1.0, 0.9 * relevance + 0.1 * answer_alignment)
        for sentence in split_sentences(text):
            drafts.extend(_relation_drafts(subject, sentence, passage_id, relevance))

    candidates: list[FollowUpCandidate] = []
    seen_relation_subject: set[tuple[str, str]] = set()
    for draft in sorted(
        drafts,
        key=lambda item: (item.evidence_strength, item.relevance_score),
        reverse=True,
    ):
        canonical = _canonical_relation(draft.relation)
        if active_config.avoid_same_relation and canonical in visited:
            continue
        relation_subject = (canonical, normalize_question(draft.subject))
        if relation_subject in seen_relation_subject:
            continue

        maximum_similarity = max(
            (question_similarity(draft.question, previous) for previous in prior_questions),
            default=0.0,
        )
        if maximum_similarity >= active_config.duplicate_similarity_threshold:
            continue
        if any(
            _canonical_relation(candidate.relation) == canonical
            and question_similarity(draft.question, candidate.question)
            >= active_config.duplicate_similarity_threshold
            for candidate in candidates
        ):
            continue

        subject_coverage = _subject_coverage(draft.subject, draft.evidence_sentence)
        selected_bonus = 0.04 if selected_id == draft.source_passage_id else 0.0
        answerability = min(
            1.0,
            0.72 * draft.evidence_strength + 0.20 * subject_coverage + selected_bonus,
        )
        if answerability < active_config.min_answerability_score:
            continue
        needs_probe = (
            active_config.allow_bm25_probe
            and probe is not None
            and draft.source_passage_id != selected_id
            and draft.relevance_score < 0.75
        )
        if needs_probe and not _probe_supports(draft, probe, active_config):
            continue

        novelty = min(1.0, max(0.0, 1.0 - maximum_similarity))
        ranking = (
            active_config.answerability_weight * answerability
            + active_config.relevance_weight * draft.relevance_score
            + active_config.novelty_weight * novelty
        )
        candidates.append(
            FollowUpCandidate(
                question=draft.question,
                subject=draft.subject,
                relation=draft.relation,
                question_type=draft.question_type,
                source_passage_id=draft.source_passage_id,
                answerability_score=round(answerability, 6),
                novelty_score=round(novelty, 6),
                relevance_score=round(draft.relevance_score, 6),
                ranking_score=round(ranking, 6),
            )
        )
        seen_relation_subject.add(relation_subject)

    candidates.sort(
        key=lambda item: (item.ranking_score, item.answerability_score, item.novelty_score),
        reverse=True,
    )
    return candidates[:requested_limit]


def generate_followup_response(
    payload: Mapping[str, Any],
    *,
    passage_lookup: Callable[[str], Mapping[str, Any] | None],
    probe: Probe | None = None,
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
    }
    candidates = generate_followups(
        question,
        str(answer) if answer is not None else None,
        semantics,
        selected,
        retrieved,
        limit=requested_limit,
        visited_relations=visited_relations,
        asked_questions=asked_questions,
        probe=probe,
        config=config,
    )
    return {
        "followups": [candidate.to_dict() for candidate in candidates],
        "processing_time_ms": int((time.perf_counter() - started) * 1000),
        "grounding": "selected_and_retrieved_corpus_passages",
        "probe": "bm25" if (config or SOCRATIC_CONFIG).allow_bm25_probe and probe else None,
    }


__all__ = [
    "FollowUpCandidate",
    "SOCRATIC_CONFIG",
    "SocraticConfig",
    "generate_followup_response",
    "generate_followups",
    "load_socratic_config",
    "normalize_question",
    "question_similarity",
]
