from __future__ import annotations

import re


_WHITESPACE = re.compile(r"\s+")
_TRAILING_PAREN = re.compile(r"\s*\((?P<value>[^()]*)\)\s*$")
_YEAR = re.compile(r"\b(?:1\d{3}|20\d{2})\b")
_LEADING_PAREN = re.compile(
    r"^(?P<entity>[^()\n]{2,100}?)\s*\((?P<details>[^()]{1,140})\)\s*(?P<rest>.*)$",
    flags=re.IGNORECASE,
)
_ENTITY_PREDICATE = re.compile(
    r"^(?P<entity>.{2,100}?)(?=\s+(?:là|có|được|từng|thuộc|nằm|sinh|mất|giữ|đảm nhiệm)\b)",
    flags=re.IGNORECASE,
)
_NON_ENTITY_LEAD = re.compile(
    r"^(?:năm|ngày|tháng|sau|trước|khi|trong|tại|theo|đến|từ|có)\b",
    flags=re.IGNORECASE,
)


def _looks_like_lifespan(value: str) -> bool:
    """Return true only for birth/death-like parentheticals, not any year."""
    normalized = _WHITESPACE.sub(" ", str(value or "").strip())
    years = _YEAR.findall(normalized)
    if len(years) < 2:
        return False
    has_range = bool(re.search(r"[–—-]", normalized))
    has_full_dates = normalized.lower().count("năm") >= 2 and "tháng" in normalized.lower()
    has_life_words = bool(re.search(r"\b(?:sinh|mất|qua đời)\b", normalized, re.IGNORECASE))
    return has_range or has_full_dates or has_life_words


def clean_source_title(raw_title: str) -> str:
    """Normalize a real metadata title without destroying meaningful brackets."""
    title = _WHITESPACE.sub(" ", str(raw_title or "").strip())
    title = re.sub(r"^#{1,6}\s*", "", title)
    trailing = _TRAILING_PAREN.search(title)
    if trailing and _looks_like_lifespan(trailing.group("value")):
        title = title[: trailing.start()]
    return title.strip(" \t\r\n-–—,:;.")


def _plausible_entity(value: str) -> bool:
    candidate = clean_source_title(value)
    words = candidate.split()
    return (
        1 <= len(words) <= 14
        and len(candidate) <= 100
        and not _NON_ENTITY_LEAD.match(candidate)
        and not candidate.endswith((" ông", " bà", " họ", " nó"))
    )


def _entity_from_passage_start(text: str) -> str:
    lead = _WHITESPACE.sub(" ", str(text or "").strip())[:360]
    if not lead:
        return ""

    parenthetical = _LEADING_PAREN.match(lead)
    if parenthetical and _looks_like_lifespan(parenthetical.group("details")):
        entity = clean_source_title(parenthetical.group("entity"))
        if _plausible_entity(entity):
            return entity

    predicate = _ENTITY_PREDICATE.match(lead)
    if predicate:
        entity = clean_source_title(predicate.group("entity"))
        if _plausible_entity(entity):
            return entity
    return ""


def _first_sentence_fallback(text: str, max_length: int = 88) -> str:
    clean = _WHITESPACE.sub(" ", str(text or "").strip())
    if not clean:
        return ""
    sentence = re.split(r"(?<=[.!?…])\s+", clean, maxsplit=1)[0]
    if len(sentence) <= max_length:
        return clean_source_title(sentence)
    shortened = sentence[: max_length + 1].rsplit(" ", 1)[0].rstrip()
    if shortened.count("(") > shortened.count(")"):
        before_paren = shortened.rsplit("(", 1)[0].rstrip()
        if len(before_paren.split()) >= 1:
            shortened = before_paren
    return f"{clean_source_title(shortened)}…"


def derive_source_title(
    document_id: str,
    text: str,
    *,
    document_title: str | None = None,
    heading: str | None = None,
) -> str:
    """Resolve a source label using metadata first and passage heuristics last."""
    for explicit_title in (document_title, heading):
        cleaned = clean_source_title(explicit_title or "")
        if cleaned:
            return cleaned

    entity = _entity_from_passage_start(text)
    if entity:
        return entity

    fallback = _first_sentence_fallback(text)
    return fallback or str(document_id or "Tài liệu nguồn")


__all__ = ["clean_source_title", "derive_source_title"]
