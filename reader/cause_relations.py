from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any


_CAUSE_QUESTION_PATTERNS = (
    re.compile(r"^(?:vi sao|tai sao|do dau|boi dau)\s+(?P<target>.+?)\s*[?!.]*$"),
    re.compile(
        r"^(?:do nguyen nhan nao|nguyen nhan gi|nguyen nhan nao|ly do gi|ly do nao|"
        r"vi nguyen nhan gi)\s+(?:khien\s+)?(?P<target>.+?)\s*[?!.]*$"
    ),
    re.compile(r"^(?:dieu gi|yeu to nao)\s+khien\s+(?P<target>.+?)\s*[?!.]*$"),
    re.compile(
        r"^nguyen nhan(?:\s+[\w]+){0,3}?\s+nao\s+"
        r"(?:khien|tao nen|gay ra|dan den|lam cho)\s+(?P<target>.+?)\s*[?!.]*$"
    ),
    re.compile(
        r"^dau\s+duoc\s+xem\s+la\s+nguyen\s+nhan.+?\s+(?:cho|cua)\s+"
        r"(?P<target>.+?)\s*[?!.]*$"
    ),
    re.compile(
        r"^(?P<target>.+?)\s+(?:vi\s+ly\s+do\s+gi|do\s+nguyen\s+nhan\s+gi|"
        r"vi\s+nguyen\s+nhan\s+gi)\s*[?!.]*$"
    ),
    re.compile(r"^(?P<target>.+?)\s+(?:vi sao|tai sao)\s*[?!.]*$"),
)

_PREDICATE_START = re.compile(
    r"\b(?:phat trien manh|phat trien|bi|duoc|da|dang|se|phai|co|la|tro thanh|"
    r"sinh|ra doi|dien ra|xay ra|roi|mat|khien|lam)\b"
)
_ANAPHORIC_ONLY = re.compile(
    r"^(?:vi\s+)?(?:dieu nay|viec nay|su viec nay|nguyen nhan nay|ly do tren|do do)$"
)
_WEAK_SUBJECT_PREFIXES = {
    "cua",
    "nhu",
    "voi",
    "gom",
    "bao gom",
    "ung ho cua",
}
_TARGET_STOPWORDS = {
    "ai", "bi", "boi", "cai", "co", "cua", "da", "dang", "do", "duoc",
    "gi", "khien", "la", "lam", "mot", "nao", "nay", "nhung", "o", "phai",
    "se", "tai", "the", "thi", "trong", "va", "vi", "voi",
}
_CAUSE_MARKER = r"(?:boi\s+vi|do\s+viec|bat\s+nguon\s+tu|xuat\s+phat\s+tu|vi|do|boi)"
_EFFECT_LINK = r"(?:khien|lam\s+cho|dan\s+den|gay\s+ra|nen)"


@dataclass(frozen=True)
class CauseQuestion:
    relation: str
    subject: str
    target: str
    target_proposition: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CauseEvidence:
    answer: str
    relation_method: str
    relation_score: float
    cause_pattern_score: float
    subject_match_score: float
    target_relation_score: float
    start_char: int
    end_char: int
    evidence_sentence: str
    effect: str
    relation_evidence: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fold_text(text: str) -> str:
    """Accent-insensitive text with one output character per input character."""

    folded: list[str] = []
    for char in str(text or ""):
        if char in {"đ", "Đ"}:
            folded.append("d")
            continue
        decomposed = unicodedata.normalize("NFD", char.casefold())
        base = next(
            (item for item in decomposed if unicodedata.category(item) != "Mn"),
            char.casefold(),
        )
        folded.append(base)
    return "".join(folded)


def _clean_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and (text[start].isspace() or text[start] in ",;:-"):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in ",;:!? ."):
        end -= 1
    return start, end


def extract_cause_question(question: str) -> CauseQuestion | None:
    folded = fold_text(question).strip()
    target_start = -1
    target_end = -1
    for pattern in _CAUSE_QUESTION_PATTERNS:
        match = pattern.match(folded)
        if match:
            target_start, target_end = match.span("target")
            break
    if target_start < 0:
        return None

    target_start, target_end = _clean_span(question, target_start, target_end)
    target_proposition = question[target_start:target_end]
    folded_target = fold_text(target_proposition)
    predicate = _PREDICATE_START.search(folded_target)
    if predicate and predicate.start() > 0:
        subject_end = predicate.start()
        predicate_start = predicate.start()
    else:
        # Event/state noun phrases (for example "khí hậu của Paris") are the
        # semantic target as a whole, not merely their first token.
        subject_end = len(target_proposition)
        predicate_start = 0
    subject_start, subject_end = _clean_span(target_proposition, 0, subject_end)
    predicate_start, predicate_end = _clean_span(
        target_proposition, predicate_start, len(target_proposition)
    )
    subject = target_proposition[subject_start:subject_end]
    target = target_proposition[predicate_start:predicate_end]
    return CauseQuestion(
        relation="CAUSE",
        subject=subject,
        target=target,
        target_proposition=target_proposition,
    )


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w]+", fold_text(text), flags=re.UNICODE)
        if len(token) > 1 and token not in _TARGET_STOPWORDS
    }


def _semantic_predicate_score(target: str, effect: str, sentence: str) -> float:
    target_tokens = _content_tokens(target)
    effect_tokens = _content_tokens(effect)
    if target_tokens:
        overlap = len(target_tokens & effect_tokens) / len(target_tokens)
        if overlap >= 0.66:
            return 1.0
        if overlap > 0:
            return 0.72

    target_folded = fold_text(target)
    evidence_folded = fold_text(f"{effect} {sentence}")
    semantic_groups = (
        (
            "khinh miet", "khinh thuong", "khinh bi", "coi thuong", "khong vua mat",
            "cam ghet", "ghet bo", "re rung",
        ),
        ("roi khoi", "roi", "bo di", "buoc phai roi", "bi duoi", "truc xuat"),
        ("that bai", "bi danh bai", "thua tran", "thua cuoc"),
        ("thanh cong", "dat duoc", "chien thang", "hoan thanh"),
        ("suy giam", "giam", "thap xuong", "mat gia"),
        ("tang", "gia tang", "phat trien manh", "cao hon"),
    )
    for group in semantic_groups:
        if any(item in target_folded for item in group) and any(
            item in evidence_folded for item in group
        ):
            return 0.95
    return 0.0


def _subject_score(subject: str, effect: str, sentence: str) -> float:
    subject_folded = " ".join(fold_text(subject).split())
    if not subject_folded:
        return 0.0
    effect_folded = fold_text(effect)
    sentence_folded = fold_text(sentence)
    subject_pattern = re.compile(rf"\b{re.escape(subject_folded)}\b")
    if subject_pattern.search(effect_folded):
        return 1.0

    matches = list(subject_pattern.finditer(sentence_folded))
    for match in matches:
        prefix_tokens = re.findall(r"[\w]+", sentence_folded[max(0, match.start() - 28):match.start()])
        prefix = " ".join(prefix_tokens[-3:])
        if any(prefix.endswith(item) for item in _WEAK_SUBJECT_PREFIXES):
            continue
        return 0.95

    # A nearby pronoun is only accepted when the named subject occurs in the
    # adjacent evidence context; it receives less confidence than a direct name.
    if matches and re.search(r"\b(?:ong|ba|nguoi nay|nha van nay|triet gia nay)\b", effect_folded):
        return 0.80
    return 0.0


def _cause_frames(sentence: str) -> list[tuple[str, re.Match[str]]]:
    folded = fold_text(sentence)
    patterns = (
        (
            "CAUSE_MARKER_THEN_EFFECT",
            re.compile(
                rf"\b(?P<marker>{_CAUSE_MARKER})\s+(?P<cause>[^,.;!?]{{2,180}}?)"
                rf"(?:\s*\([^)]*\))?\s+(?P<link>{_EFFECT_LINK})\s+(?P<effect>[^.;!?]+)",
                re.IGNORECASE | re.UNICODE,
            ),
        ),
        (
            "EFFECT_THEN_CAUSE",
            re.compile(
                rf"(?P<effect>[^.;!?]{{2,220}}?)\s+\b(?P<marker>{_CAUSE_MARKER})\s+"
                rf"(?P<cause>[^,.;!?]{{2,180}}?)(?=[,.;!?]|$)",
                re.IGNORECASE | re.UNICODE,
            ),
        ),
        (
            "CAUSE_THEN_EFFECT",
            re.compile(
                rf"(?P<cause>[^,.;!?]{{2,180}}?)\s+\b(?P<link>{_EFFECT_LINK})\s+"
                rf"(?P<effect>[^.;!?]{{2,220}})",
                re.IGNORECASE | re.UNICODE,
            ),
        ),
    )
    frames: list[tuple[str, re.Match[str]]] = []
    for method, pattern in patterns:
        for match in pattern.finditer(folded):
            marker_start = match.start("marker") if "marker" in match.re.groupindex else -1
            if marker_start >= 0 and fold_text(match.groupdict().get("marker") or "") == "do":
                previous = re.findall(r"[\w]+", folded[max(0, marker_start - 16):marker_start])
                # Avoid interpreting nouns such as "tự do" or "thái độ" as
                # the Vietnamese causal preposition "do".
                if previous and previous[-1] in {"tu", "thai", "toc", "muc", "che", "cuong"}:
                    continue
            frames.append((method, match))
    return frames


def extract_cause_candidate(
    question: str,
    target: CauseQuestion | None,
    supporting_sentence: str,
    answer_hint: str | None = None,
) -> CauseEvidence | None:
    frame = target or extract_cause_question(question)
    if frame is None:
        return None

    candidates: list[CauseEvidence] = []
    for method, match in _cause_frames(supporting_sentence):
        start, end = match.span("cause")
        start, end = _clean_span(supporting_sentence, start, end)
        answer = supporting_sentence[start:end]
        trailing_parenthetical = re.search(r"\s*\([^)]*\)\s*$", answer)
        if trailing_parenthetical:
            end = start + trailing_parenthetical.start()
            start, end = _clean_span(supporting_sentence, start, end)
            answer = supporting_sentence[start:end]
        folded_answer = " ".join(fold_text(answer).split())
        if not folded_answer or _ANAPHORIC_ONLY.fullmatch(folded_answer):
            continue
        effect = supporting_sentence[slice(*match.span("effect"))].strip()
        pattern_score = 0.96 if method != "EFFECT_THEN_CAUSE" else 0.92
        subject_score = _subject_score(frame.subject, effect, supporting_sentence)
        target_score = _semantic_predicate_score(frame.target, effect, supporting_sentence)
        relation_score = round(
            0.40 * pattern_score + 0.30 * subject_score + 0.30 * target_score,
            6,
        )
        relation_evidence = bool(
            pattern_score >= 0.85 and subject_score >= 0.75 and target_score >= 0.55
        )
        if subject_score < 0.50:
            reason = "CAUSE_SUBJECT_MISMATCH"
        elif target_score < 0.55:
            reason = "CAUSE_TARGET_MISMATCH"
        elif pattern_score < 0.65:
            reason = "CAUSE_RELATION_MISMATCH"
        else:
            reason = None
        candidates.append(
            CauseEvidence(
                answer=answer,
                relation_method=method,
                relation_score=relation_score,
                cause_pattern_score=pattern_score,
                subject_match_score=subject_score,
                target_relation_score=target_score,
                start_char=start,
                end_char=end,
                evidence_sentence=supporting_sentence,
                effect=effect,
                relation_evidence=relation_evidence,
                rejection_reason=reason,
            )
        )
    if answer_hint:
        hint_folded = " ".join(fold_text(answer_hint).split())
        candidates = [
            item
            for item in candidates
            if hint_folded in " ".join(fold_text(item.answer).split())
            or " ".join(fold_text(item.answer).split()) in hint_folded
        ]
    return max(
        candidates,
        key=lambda item: (
            item.relation_evidence,
            item.relation_score,
            -len(_content_tokens(item.answer)),
        ),
        default=None,
    )


def assess_cause_candidate(
    question: str,
    context: str,
    answer: str,
    evidence_sentence: str | None = None,
) -> CauseEvidence:
    frame = extract_cause_question(question)
    empty = CauseEvidence(
        answer=answer,
        relation_method="CAUSE_PHRASE_NOT_FOUND",
        relation_score=0.0,
        cause_pattern_score=0.0,
        subject_match_score=0.0,
        target_relation_score=0.0,
        start_char=-1,
        end_char=-1,
        evidence_sentence=evidence_sentence or "",
        effect="",
        relation_evidence=False,
        rejection_reason="CAUSE_PHRASE_NOT_FOUND",
    )
    if frame is None:
        return empty

    answer_folded = " ".join(fold_text(answer).split())
    if not answer_folded or _ANAPHORIC_ONLY.fullmatch(answer_folded):
        return empty

    sentences = [evidence_sentence] if evidence_sentence else re.split(r"(?<=[.!?])\s+", context)
    best: CauseEvidence | None = None
    for sentence in filter(None, sentences):
        extracted = extract_cause_candidate(
            question,
            frame,
            sentence,
            answer_hint=answer,
        )
        if extracted is None:
            continue
        extracted_folded = " ".join(fold_text(extracted.answer).split())
        same_phrase = bool(extracted_folded) and (
            answer_folded in extracted_folded or extracted_folded in answer_folded
        )
        if not same_phrase:
            continue
        if best is None or extracted.relation_score > best.relation_score:
            best = extracted
    return best or empty


def cause_subject_match_score(question: str, context: str) -> float:
    frame = extract_cause_question(question)
    if frame is None:
        return 0.0
    return _subject_score(frame.subject, "", context)


__all__ = [
    "CauseEvidence",
    "CauseQuestion",
    "assess_cause_candidate",
    "cause_subject_match_score",
    "extract_cause_candidate",
    "extract_cause_question",
    "fold_text",
]
