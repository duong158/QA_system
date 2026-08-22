from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable


SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?…])(?:[\"'”’)]*)\s+(?=(?:[\"'“‘(]*[^\W_]))",
    flags=re.UNICODE,
)
PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n+")


@dataclass(frozen=True)
class Passage:
    document_id: str
    passage_id: str
    title: str
    paragraph_id: str
    sentence_start: int
    sentence_end: int
    text: str
    page: int | None = None


def split_sentences(text: str) -> list[str]:
    clean = re.sub(r"[ \t]+", " ", str(text or "").strip())
    if not clean:
        return []
        
    parts = [part.strip() for part in SENTENCE_BOUNDARY.split(clean) if part.strip()]
    
    merged: list[str] = []
    abbreviations = {"tr", "tp", "gs", "ts", "ths", "bs", "vs", "vd", "st", "mr", "mrs", "ms", "dr", "prof", "v.v"}
    
    for part in parts:
        if not merged:
            merged.append(part)
            continue
            
        prev = merged[-1]
        words = prev.split()
        if words:
            last_word = words[-1].lower()
            if last_word.endswith('.'):
                core = last_word[:-1]
                if core in abbreviations or len(core) == 1 or core.isdigit():
                    merged[-1] = f"{prev} {part}"
                    continue
                    
        merged.append(part)
        
    return merged


def _split_oversized_sentence(
    sentence: str,
    max_tokens: int,
    token_count: Callable[[str], int],
) -> list[str]:
    """Last-resort split for a single sentence that exceeds the token budget."""
    clauses = [part.strip() for part in re.split(r"(?<=[;:])\s+|(?<=,)\s+", sentence) if part.strip()]
    if len(clauses) > 1 and all(token_count(clause) <= max_tokens for clause in clauses):
        return clauses

    words = sentence.split()
    pieces: list[str] = []
    for start in range(0, len(words), max_tokens):
        pieces.append(" ".join(words[start : start + max_tokens]))
    return pieces or [sentence]


def _pack_sentences(
    sentences: list[str],
    max_tokens: int,
    overlap_sentences: int,
    token_count: Callable[[str], int],
) -> Iterable[tuple[int, int, str]]:
    expanded: list[str] = []
    for sentence in sentences:
        if token_count(sentence) > max_tokens:
            expanded.extend(_split_oversized_sentence(sentence, max_tokens, token_count))
        else:
            expanded.append(sentence)

    start = 0
    while start < len(expanded):
        end = start
        current: list[str] = []
        while end < len(expanded):
            candidate = " ".join([*current, expanded[end]])
            if current and token_count(candidate) > max_tokens:
                break
            current.append(expanded[end])
            end += 1

        yield start, end - 1, " ".join(current).strip()
        if end >= len(expanded):
            break
        start = max(start + 1, end - overlap_sentences)


def chunk_document(
    document_id: str,
    text: str,
    title: str = "",
    page: int | None = None,
    max_tokens: int = 220,
    overlap_sentences: int = 2,
    token_count: Callable[[str], int] | None = None,
) -> list[Passage]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative")

    count = token_count or (lambda value: len(value.split()))
    paragraphs = [part.strip() for part in PARAGRAPH_BOUNDARY.split(str(text or "")) if part.strip()]
    if not paragraphs and str(text or "").strip():
        paragraphs = [str(text).strip()]

    passages: list[Passage] = []
    global_sentence = 0
    for paragraph_index, paragraph in enumerate(paragraphs):
        sentences = split_sentences(paragraph)
        paragraph_start = global_sentence
        for local_start, local_end, passage_text in _pack_sentences(
            sentences,
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences,
            token_count=count,
        ):
            passage_number = len(passages) + 1
            passages.append(
                Passage(
                    document_id=document_id,
                    passage_id=f"{document_id}_P{passage_number:04d}",
                    title=title or document_id,
                    paragraph_id=f"{document_id}_PAR{paragraph_index + 1:04d}",
                    sentence_start=paragraph_start + local_start,
                    sentence_end=paragraph_start + local_end,
                    text=passage_text,
                    page=page,
                )
            )
        global_sentence += len(sentences)
    return passages
