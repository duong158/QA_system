from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.chunking import Passage, chunk_document, split_sentences


DOCS_DB = ROOT / "data" / "processed" / "docs.db"
HOST = os.getenv("QA_HOST", "0.0.0.0")
PORT = int(os.getenv("QA_PORT", "8000"))
CHUNK_MAX_TOKENS = int(os.getenv("QA_CHUNK_MAX_TOKENS", "220"))
CHUNK_OVERLAP_SENTENCES = int(os.getenv("QA_CHUNK_OVERLAP_SENTENCES", "2"))
RETRIEVER_CANDIDATE_MULTIPLIER = int(os.getenv("QA_RETRIEVER_CANDIDATE_MULTIPLIER", "3"))
RETRIEVER_MIN_CANDIDATES = int(os.getenv("QA_RETRIEVER_MIN_CANDIDATES", "8"))
RETRIEVER_WEIGHT = float(os.getenv("QA_RETRIEVER_WEIGHT", "0.15"))
READER_WEIGHT = float(os.getenv("QA_READER_WEIGHT", "0.85"))
ANSWER_THRESHOLD = float(os.getenv("QA_ANSWER_THRESHOLD", "0.30"))
READER_FALLBACK_THRESHOLD = float(os.getenv("QA_READER_FALLBACK_THRESHOLD", "0.30"))
SENTENCE_FALLBACK_THRESHOLD = float(os.getenv("QA_SENTENCE_FALLBACK_THRESHOLD", "0.42"))
QA_DEBUG = os.getenv("QA_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
SUPPORTED_RETRIEVERS = {"tfidf", "bm25"}
UNIMPLEMENTED_RETRIEVERS = {
    "dense": "Dense retrieval is not wired into this API yet: no embedding model/vector index is configured for online serving.",
    "pyserini": "Pyserini BM25 is not wired into this API yet: no Lucene index/runtime is configured for online serving.",
}
SUPPORTED_READERS = {"phobert"}
UNIMPLEMENTED_READERS = {
    "mock": "Mock Reader is forbidden in the real API.",
    "vibert": "viBERT QA is not implemented: no viBERT QA checkpoint is available under models/reader.",
    "xlmr": "XLM-R QA is not implemented: no XLM-R QA checkpoint is available under models/reader.",
}

if RETRIEVER_WEIGHT < 0 or READER_WEIGHT < 0 or RETRIEVER_WEIGHT + READER_WEIGHT <= 0:
    raise ValueError("QA retriever/reader weights must be non-negative and have a positive sum")

WEIGHT_TOTAL = RETRIEVER_WEIGHT + READER_WEIGHT
RETRIEVER_WEIGHT /= WEIGHT_TOTAL
READER_WEIGHT /= WEIGHT_TOTAL

STOPWORDS = {
    "ai", "anh", "ay", "ban", "bang", "bao", "bi", "cac", "cai", "can", "chi",
    "cho", "co", "con", "cua", "da", "dang", "day", "de", "den", "di", "do",
    "duoc", "duoi", "gi", "giua", "hay", "hon", "khi", "khong", "la", "lai",
    "lam", "may", "mot", "nao", "nay", "neu", "ngay", "nhieu", "nhu", "nhung",
    "o", "phai", "qua", "ra", "rang", "sau", "se", "so", "tai", "the", "thi",
    "theo", "tren", "trong", "truoc", "tu", "va", "vao", "ve", "vi", "voi",
}

DATE_TOKENS = {"ngay", "thang", "nam"}


class PipelineError(RuntimeError):
    pass


def validate_retriever(method: str) -> None:
    if method in SUPPORTED_RETRIEVERS:
        return
    if method in UNIMPLEMENTED_RETRIEVERS:
        raise ValueError(f"Retriever '{method}' is not implemented. {UNIMPLEMENTED_RETRIEVERS[method]}")
    raise ValueError(f"Retriever '{method}' is not supported.")


def validate_reader(reader_name: str) -> None:
    if reader_name in SUPPORTED_READERS:
        return
    if reader_name in UNIMPLEMENTED_READERS:
        raise ValueError(f"Reader '{reader_name}' is not implemented. {UNIMPLEMENTED_READERS[reader_name]}")
    raise ValueError(f"Reader '{reader_name}' is not supported.")


@dataclass(frozen=True)
class IndexedPassage:
    metadata: Passage
    tokens: tuple[str, ...]
    term_counts: Counter[str]


@dataclass(frozen=True)
class SearchHit:
    passage: IndexedPassage
    retrieval_score_raw: float
    retrieval_score_normalized: float = 0.0
    retrieval_rank: int = 0


def normalize_text(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    value = unicodedata.normalize("NFD", value.lower())
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")


def tokenize(value: str) -> list[str]:
    tokens = re.findall(r"[\w%]+", normalize_text(value), flags=re.UNICODE)
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def raw_tokens(value: str) -> list[str]:
    return re.findall(r"[\w%]+", normalize_text(value), flags=re.UNICODE)


def normalized_token_text(tokens: list[str] | tuple[str, ...]) -> str:
    return " ".join(tokens)


def query_ngrams(tokens: list[str], min_n: int = 2, max_n: int = 4) -> list[str]:
    grams: list[str] = []
    limit = min(max_n, len(tokens))
    for size in range(limit, min_n - 1, -1):
        grams.extend(" ".join(tokens[index : index + size]) for index in range(0, len(tokens) - size + 1))
    return grams


def find_sequence(tokens: list[str], sequence: list[str]) -> int:
    if not sequence or len(sequence) > len(tokens):
        return -1
    for index in range(0, len(tokens) - len(sequence) + 1):
        if tokens[index : index + len(sequence)] == sequence:
            return index
    return -1


def softmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    high = max(scores)
    values = [math.exp(min(50.0, score - high)) for score in scores]
    total = sum(values)
    if not total:
        return [0.0 for _ in scores]
    return [value / total for value in values]


def min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if math.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def finite_or_none(value: Any) -> float | None:
    number = float(value)
    return round(number, 6) if math.isfinite(number) else None


ANSWER_CUE_PATTERNS = (
    "duoc chia thanh",
    "duoc chia lam",
    "chia thanh",
    "chia lam",
    "bao gom",
    "gom",
    "la",
    "duoc goi la",
    "co nghia la",
    "duoc dinh nghia",
)


def sentence_fallback_predict(question: str, context: str) -> dict[str, Any]:
    """Extract a useful full-sentence answer when the neural reader is not confident.

    This is deliberately generic: it does not map specific questions to answers.
    It picks the context sentence with strong lexical overlap and Vietnamese answer cues.
    """
    question_tokens = set(tokenize(question))
    if not question_tokens:
        return {"answer": "", "confidence": 0.0, "start": -1, "end": -1, "reason": "empty_question_tokens"}

    normalized_question = normalize_text(question)
    subject_phrase = re.sub(
        r"\b(bao gom nhung gi|gom nhung gi|co nhung gi|duoc chia thanh|duoc chia lam|duoc chia|chia thanh|chia lam|nhu the nao|nhu nao|la gi|la ai|chia|duoc|nao|gi|ai)\b",
        " ",
        normalized_question,
    )
    subject_phrase = re.sub(r"\s+", " ", subject_phrase).strip()
    best: dict[str, Any] = {"answer": "", "confidence": 0.0, "start": -1, "end": -1, "reason": "no_sentence"}
    search_from = 0
    for sentence in split_sentences(context):
        sentence = sentence.strip()
        if not sentence:
            continue
        start = context.find(sentence, search_from)
        if start < 0:
            start = context.find(sentence)
        end = start + len(sentence) if start >= 0 else -1
        search_from = max(search_from, end)

        sentence_tokens = set(tokenize(sentence))
        overlap = len(question_tokens & sentence_tokens) / max(1, len(question_tokens))
        normalized_sentence = normalize_text(sentence)
        cue_bonus = 0.0
        if any(pattern in normalized_sentence for pattern in ANSWER_CUE_PATTERNS):
            cue_bonus += 0.18
        if len(subject_phrase) >= 8 and subject_phrase in normalized_sentence:
            cue_bonus += 0.22
        if len(sentence_tokens) >= 6:
            cue_bonus += 0.04
        if len(sentence) > 520:
            cue_bonus -= 0.12
        score = max(0.0, min(0.92, overlap * 0.78 + cue_bonus))

        if score > best["confidence"]:
            best = {
                "answer": sentence,
                "confidence": round(score, 6),
                "start": start,
                "end": end,
                "reason": "sentence_overlap_cue",
            }
    return best


def choose_reader_output(neural_output: dict[str, Any], fallback_output: dict[str, Any]) -> dict[str, Any]:
    neural_confidence = float(neural_output["confidence"])
    fallback_confidence = float(fallback_output["confidence"])
    if neural_output.get("answer") and neural_confidence >= READER_FALLBACK_THRESHOLD:
        return {
            "method": "phobert",
            "answer": neural_output["answer"],
            "confidence": neural_confidence,
            "start": int(neural_output["start"]),
            "end": int(neural_output["end"]),
        }
    if fallback_output.get("answer") and fallback_confidence >= SENTENCE_FALLBACK_THRESHOLD:
        return {
            "method": "sentence_fallback",
            "answer": fallback_output["answer"],
            "confidence": fallback_confidence,
            "start": int(fallback_output["start"]),
            "end": int(fallback_output["end"]),
        }
    return {
        "method": "phobert",
        "answer": neural_output.get("answer") or None,
        "confidence": neural_confidence,
        "start": int(neural_output["start"]),
        "end": int(neural_output["end"]),
    }


def expand_answer_to_sentence(context: str, answer: str, start: int, end: int) -> str:
    answer = str(answer or "").strip()
    if not answer or start < 0 or end <= start or end > len(context):
        return answer

    raw_span = context[start:end].strip()
    if normalize_text(answer) not in normalize_text(raw_span):
        return answer

    max_chars = int(os.getenv("QA_ANSWER_SENTENCE_MAX_CHARS", "360"))
    left_boundaries = [context.rfind(mark, 0, start) for mark in ".!?\n"]
    sentence_start = max(left_boundaries) + 1
    while sentence_start < len(context) and context[sentence_start].isspace():
        sentence_start += 1

    right_candidates = [context.find(mark, end) for mark in ".!?\n"]
    right_candidates = [index for index in right_candidates if index >= 0]
    sentence_end = (min(right_candidates) + 1) if right_candidates else len(context)
    while sentence_end > sentence_start and context[sentence_end - 1].isspace():
        sentence_end -= 1

    sentence = context[sentence_start:sentence_end].strip()
    if len(sentence) <= len(answer) or len(sentence) > max_chars:
        return answer
    return sentence


class PassageIndex:
    def __init__(self, db_path: Path) -> None:
        if not db_path.exists():
            raise FileNotFoundError(f"Missing docs database: {db_path}")
        self.passages = self._load_passages(db_path)
        self.avg_passage_len = sum(len(item.tokens) for item in self.passages) / max(1, len(self.passages))
        frequencies: dict[str, int] = defaultdict(int)
        for passage in self.passages:
            for term in passage.term_counts:
                frequencies[term] += 1
        self.idf = {
            term: math.log(1 + (len(self.passages) - count + 0.5) / (count + 0.5))
            for term, count in frequencies.items()
        }

    @staticmethod
    def _guess_title(document_id: str, text: str) -> str:
        sentences = split_sentences(text)
        words = (sentences[0] if sentences else text).split()
        return " ".join(words[:8]) if words else document_id

    def _load_passages(self, db_path: Path) -> list[IndexedPassage]:
        connection = sqlite3.connect(str(db_path))
        try:
            rows = connection.execute("SELECT id, text FROM documents").fetchall()
        finally:
            connection.close()

        indexed: list[IndexedPassage] = []
        for document_id, raw_text in rows:
            text = str(raw_text or "").strip()
            if not text:
                continue
            title = self._guess_title(str(document_id), text)
            for passage in chunk_document(
                str(document_id),
                text,
                title=title,
                max_tokens=CHUNK_MAX_TOKENS,
                overlap_sentences=CHUNK_OVERLAP_SENTENCES,
            ):
                tokens = tuple(tokenize(passage.text))
                indexed.append(IndexedPassage(passage, tokens, Counter(tokens)))
        return indexed

    def _bm25(self, query_tokens: list[str], passage: IndexedPassage) -> float:
        k1, b = 1.5, 0.75
        passage_length = max(1, len(passage.tokens))
        score = 0.0
        for term in query_tokens:
            frequency = passage.term_counts.get(term, 0)
            if not frequency:
                continue
            frequency = min(frequency, 1)
            denominator = frequency + k1 * (
                1 - b + b * passage_length / max(1.0, self.avg_passage_len)
            )
            score += self.idf.get(term, 0.0) * (frequency * (k1 + 1)) / denominator
        return score

    def _tfidf(self, query_tokens: list[str], passage: IndexedPassage) -> float:
        query_counts = Counter(query_tokens)
        numerator = 0.0
        query_norm = 0.0
        document_norm = 0.0
        for term, count in query_counts.items():
            idf = self.idf.get(term, 0.0)
            query_weight = (1 + math.log(count)) * idf
            document_count = passage.term_counts.get(term, 0)
            document_weight = (1 + math.log(document_count)) * idf if document_count else 0.0
            numerator += query_weight * document_weight
            query_norm += query_weight * query_weight
        for term, count in passage.term_counts.items():
            weight = (1 + math.log(count)) * self.idf.get(term, 0.0)
            document_norm += weight * weight
        if not query_norm or not document_norm:
            return 0.0
        return numerator / math.sqrt(query_norm * document_norm)

    def _lexical_boost(self, query_tokens: list[str], passage: IndexedPassage) -> float:
        query_terms = list(dict.fromkeys(query_tokens))
        if not query_terms:
            return 0.0

        passage_terms = set(passage.tokens)
        matched_terms = [term for term in query_terms if term in passage_terms]
        if not matched_terms:
            return 0.0

        coverage = len(matched_terms) / len(query_terms)
        boost = coverage * 2.0

        title_tokens = set(tokenize(passage.metadata.title))
        if title_tokens:
            title_matches = sum(1 for term in query_terms if term in title_tokens)
            boost += 0.8 * title_matches / len(query_terms)

        passage_token_text = normalized_token_text(passage.tokens)
        phrase_hits = 0
        for gram in query_ngrams(query_tokens):
            if gram in passage_token_text:
                phrase_hits += 1
        if phrase_hits:
            boost += min(1.5, 0.35 * phrase_hits)

        positions: dict[str, list[int]] = defaultdict(list)
        for index, token in enumerate(passage.tokens):
            if token in matched_terms:
                positions[token].append(index)
        first_positions = [indexes[0] for indexes in positions.values() if indexes]
        if len(first_positions) >= 2:
            window = max(first_positions) - min(first_positions) + 1
            if window <= 40:
                boost += (40 - window) / 40

        return boost

    def _lead_passage_boost(self, query_tokens: list[str], passage: IndexedPassage) -> float:
        query_terms = list(dict.fromkeys(query_tokens))
        if len(query_terms) < 2:
            return 0.0

        sentences = split_sentences(passage.metadata.text)
        if not sentences:
            return 0.0

        first_tokens = raw_tokens(sentences[0])
        first_content = [token for token in first_tokens if len(token) > 1 and token not in STOPWORDS]
        phrase_index = find_sequence(first_content, query_terms)
        if phrase_index < 0:
            return 0.0

        boost = 0.0
        if phrase_index <= 2:
            boost += 1.0

        raw_phrase_index = find_sequence(first_tokens, query_terms)
        if raw_phrase_index >= 0:
            boost += max(0.0, 0.8 - 0.08 * raw_phrase_index)
            after_phrase = raw_phrase_index + len(query_terms)
            try:
                definition_index = first_tokens.index("la", after_phrase)
            except ValueError:
                definition_index = -1
            if definition_index > after_phrase:
                gap = first_tokens[after_phrase:definition_index]
                gap_content = [
                    token for token in gap
                    if token not in STOPWORDS and token not in DATE_TOKENS and not token.isdigit()
                ]
                if not gap_content and definition_index - raw_phrase_index <= 18:
                    boost += 2.4

        return boost

    def retrieve(self, question: str, method: str, top_k: int) -> list[SearchHit]:
        query_tokens = tokenize(question)
        if not query_tokens:
            return []
        validate_retriever(method)

        scored: list[tuple[IndexedPassage, float]] = []
        scorer = self._bm25 if method == "bm25" else self._tfidf
        for passage in self.passages:
            base_score = scorer(query_tokens, passage)
            score = (
                base_score
                + self._lexical_boost(query_tokens, passage)
                + self._lead_passage_boost(query_tokens, passage)
            )
            if score > 0:
                scored.append((passage, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = scored[:top_k]
        normalized = min_max_normalize([score for _, score in selected])
        return [
            SearchHit(passage, raw, norm, rank)
            for rank, ((passage, raw), norm) in enumerate(zip(selected, normalized), start=1)
        ]


class ReaderManager:
    MODEL_FOLDERS = {
        "phobert": "vinai_phobert-base-v2",
    }

    def __init__(self) -> None:
        self.predictors: dict[str, Any] = {}

    def get(self, reader_name: str):
        validate_reader(reader_name)
        folder = self.MODEL_FOLDERS.get(reader_name)
        if reader_name not in self.predictors:
            model_dir = ROOT / "models" / "reader" / folder
            if not model_dir.exists():
                raise PipelineError(f"Reader checkpoint is missing: {model_dir}")
            try:
                from reader.predict import ReaderPredictor

                self.predictors[reader_name] = ReaderPredictor(str(model_dir))
            except Exception as error:
                raise PipelineError(f"Failed to load reader '{reader_name}': {error}") from error
        return self.predictors[reader_name]


INDEX = PassageIndex(DOCS_DB)
READERS = ReaderManager()


def _empty_response(question: str, retriever: str, reader: str, elapsed: int) -> dict[str, Any]:
    return {
        "question": question,
        "answer": None,
        "has_answer": False,
        "confidence": 0.0,
        "selected_passage_id": None,
        "processing_time_ms": elapsed,
        "retriever": retriever,
        "reader": reader,
        "source": None,
        "answer_source": None,
        "top_retrieved_passage": None,
        "no_answer_reason": "Retriever returned no passage with a positive score.",
        "answer_span": None,
        "passages": [],
    }


def ask_question(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    question = str(payload.get("question", "")).strip()
    retriever = str(payload.get("retriever", "bm25")).lower()
    reader_name = str(payload.get("reader", "phobert")).lower()
    try:
        top_k = min(20, max(1, int(payload.get("top_k", 5) or 5)))
    except (TypeError, ValueError) as error:
        raise ValueError("top_k must be an integer") from error
    if not question:
        raise ValueError("question is required")
    validate_retriever(retriever)
    validate_reader(reader_name)

    candidate_count = min(
        20,
        max(top_k, RETRIEVER_MIN_CANDIDATES, top_k * max(1, RETRIEVER_CANDIDATE_MULTIPLIER)),
    )
    hits = INDEX.retrieve(question, retriever, candidate_count)
    if not hits:
        return _empty_response(
            question,
            retriever,
            reader_name,
            int((time.perf_counter() - started) * 1000),
        )

    predictor = READERS.get(reader_name)
    passages: list[dict[str, Any]] = []
    for hit in hits:
        output = predictor.predict(
            question,
            hit.passage.metadata.text,
            no_answer_threshold=ANSWER_THRESHOLD,
        )
        fallback_output = sentence_fallback_predict(question, hit.passage.metadata.text)
        chosen_output = choose_reader_output(output, fallback_output)
        reader_score = float(chosen_output["confidence"])
        final_score = RETRIEVER_WEIGHT * hit.retrieval_score_normalized + READER_WEIGHT * reader_score
        metadata = hit.passage.metadata
        chosen_span_answer = str(chosen_output["answer"] or "")
        display_answer = expand_answer_to_sentence(
            metadata.text,
            chosen_span_answer,
            int(chosen_output["start"]),
            int(chosen_output["end"]),
        )
        passages.append(
            {
                "rank": 0,
                "retrieval_rank": hit.retrieval_rank,
                "document_id": metadata.document_id,
                "passage_id": metadata.passage_id,
                "title": metadata.title,
                "paragraph_id": metadata.paragraph_id,
                "sentence_start": metadata.sentence_start,
                "sentence_end": metadata.sentence_end,
                "page": metadata.page,
                "text": metadata.text,
                "retrieval_score": round(hit.retrieval_score_normalized, 6),
                "retrieval_score_raw": round(hit.retrieval_score_raw, 6),
                "retrieval_score_normalized": round(hit.retrieval_score_normalized, 6),
                "reader_method": chosen_output["method"],
                "reader_answer": display_answer or None,
                "reader_span_answer": chosen_span_answer or None,
                "reader_score": round(reader_score, 6),
                "neural_reader_answer": output["answer"] or None,
                "neural_reader_score": round(float(output["confidence"]), 6),
                "reader_score_raw": finite_or_none(output["score"]),
                "reader_null_score": finite_or_none(output["null_score"]),
                "reader_score_margin": finite_or_none(output["score_margin"]),
                "fallback_answer": fallback_output["answer"] or None,
                "fallback_score": round(float(fallback_output["confidence"]), 6),
                "fallback_reason": fallback_output["reason"],
                "answer_span": {
                    "text": chosen_span_answer,
                    "start": int(chosen_output["start"]),
                    "end": int(chosen_output["end"]),
                },
                "final_score": round(final_score, 6),
            }
        )

    margins = [
        float(item["reader_score_margin"])
        for item in passages
        if item["reader_method"] != "sentence_fallback"
        and item["reader_score_margin"] is not None
        and math.isfinite(float(item["reader_score_margin"]))
    ]
    margin_scores = softmax_normalize(margins)
    margin_index = 0
    for item in passages:
        if item["reader_method"] == "sentence_fallback":
            item["reader_margin_score"] = float(item["reader_score"])
        elif item["reader_score_margin"] is not None and math.isfinite(float(item["reader_score_margin"])):
            item["reader_margin_score"] = round(margin_scores[margin_index], 6)
            margin_index += 1
        else:
            item["reader_margin_score"] = 0.0
        reader_signal = 0.75 * float(item["reader_score"]) + 0.25 * float(item["reader_margin_score"])
        item["final_score"] = round(
            RETRIEVER_WEIGHT * float(item["retrieval_score_normalized"])
            + READER_WEIGHT * reader_signal,
            6,
        )

    top_retrieved = min(passages, key=lambda item: item["retrieval_rank"])
    passages.sort(key=lambda item: item["final_score"], reverse=True)
    passages = passages[:top_k]
    for rank, item in enumerate(passages, start=1):
        item["rank"] = rank

    selected = passages[0]
    has_answer = bool(selected["reader_answer"]) and selected["reader_score"] >= ANSWER_THRESHOLD
    elapsed = int((time.perf_counter() - started) * 1000)
    answer_source = {
        "document_id": selected["document_id"],
        "passage_id": selected["passage_id"],
        "title": selected["title"],
        "paragraph_id": selected["paragraph_id"],
        "sentence_start": selected["sentence_start"],
        "sentence_end": selected["sentence_end"],
        "page": selected["page"],
    }
    best_reader_score = max((float(passage["reader_score"]) for passage in passages), default=0.0)
    no_answer_reason = None
    if not has_answer:
        if best_reader_score < ANSWER_THRESHOLD:
            no_answer_reason = "Reader confidence below threshold."
        elif not selected["reader_answer"]:
            no_answer_reason = "Reader did not extract a valid answer span."
        else:
            no_answer_reason = "No answer satisfied the QA acceptance criteria."

    response = {
        "question": question,
        "answer": selected["reader_answer"] if has_answer else None,
        "has_answer": has_answer,
        "confidence": selected["reader_score"] if has_answer else 0.0,
        "selected_passage_id": selected["passage_id"] if has_answer else None,
        "processing_time_ms": elapsed,
        "retriever": retriever,
        "reader": reader_name,
        "source": answer_source if has_answer else None,
        "answer_source": answer_source if has_answer else None,
        "top_retrieved_passage": top_retrieved,
        "no_answer_reason": no_answer_reason,
        "best_reader_score": round(best_reader_score, 6),
        "answer_span": selected["answer_span"] if has_answer else None,
        "passages": passages,
        "scoring": {
            "retriever_weight": RETRIEVER_WEIGHT,
            "reader_weight": READER_WEIGHT,
            "answer_threshold": ANSWER_THRESHOLD,
            "reader_fallback_threshold": READER_FALLBACK_THRESHOLD,
            "sentence_fallback_threshold": SENTENCE_FALLBACK_THRESHOLD,
            "retrieval_normalization": "min_max_within_top_k",
            "candidate_count": candidate_count,
            "rerank": "retrieval_reader_margin",
            "final_score_formula": "retriever_weight*retrieval + reader_weight*(0.75*reader_confidence + 0.25*reader_margin)",
        },
    }
    if QA_DEBUG:
        _log_debug(response)
    return response


def _log_debug(response: dict[str, Any]) -> None:
    print(f"[QA_DEBUG] QUESTION {response['question']!r}")
    for passage in response["passages"]:
        print(
            "[QA_DEBUG] "
            f"rank={passage['rank']} passage={passage['passage_id']} "
            f"retrieval_raw={passage['retrieval_score_raw']:.6f} "
            f"retrieval_norm={passage['retrieval_score_normalized']:.6f} "
            f"reader={passage['reader_score']:.6f} final={passage['final_score']:.6f} "
            f"answer={passage['reader_answer']!r}"
        )
    print(
        f"[QA_DEBUG] FINAL passage={response['selected_passage_id']} "
        f"confidence={response['confidence']:.6f} has_answer={response['has_answer']}"
    )


def compare_retrievers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise ValueError("question is required")
    rows: list[dict[str, Any]] = []
    for method, label in (("tfidf", "TF-IDF"), ("bm25", "BM25")):
        started = time.perf_counter()
        hits = INDEX.retrieve(question, method, 3)
        elapsed = int((time.perf_counter() - started) * 1000)
        first = hits[0] if hits else None
        rows.append(
            {
                "retriever": method,
                "label": label,
                "correctPassageRank": None,
                "recallAt1": None,
                "recallAt3": None,
                "responseTimeMs": elapsed,
                "topPassagePreview": first.passage.metadata.text if first else "",
                "retrievalScore": round(first.retrieval_score_raw, 6) if first else 0.0,
                "evaluationNote": "Ground truth is required to compute correctness and Recall@k",
            }
        )
    return rows


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "passages": len(INDEX.passages),
                    "reader_models": ReaderManager.MODEL_FOLDERS,
                    "supported_retrievers": sorted(SUPPORTED_RETRIEVERS),
                    "unsupported_retrievers": UNIMPLEMENTED_RETRIEVERS,
                    "supported_readers": sorted(SUPPORTED_READERS),
                    "unsupported_readers": UNIMPLEMENTED_READERS,
                    "config": {
                        "chunk_max_tokens": CHUNK_MAX_TOKENS,
                        "chunk_overlap_sentences": CHUNK_OVERLAP_SENTENCES,
                        "retriever_weight": RETRIEVER_WEIGHT,
                        "reader_weight": READER_WEIGHT,
                        "answer_threshold": ANSWER_THRESHOLD,
                        "reader_fallback_threshold": READER_FALLBACK_THRESHOLD,
                        "sentence_fallback_threshold": SENTENCE_FALLBACK_THRESHOLD,
                    },
                }
            )
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            path = urlparse(self.path).path
            if path == "/api/ask":
                self._send_json(ask_question(payload))
                return
            if path == "/api/compare":
                self._send_json(compare_retrievers(payload))
                return
            self._send_json({"error": "Not found"}, status=404)
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
        except PipelineError as error:
            self._send_json({"error": str(error)}, status=503)
        except Exception as error:
            traceback.print_exc()
            self._send_json({"error": f"QA pipeline failed: {error}"}, status=500)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, data: Any, status: int = 200) -> None:
        encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[VIQA API] {self.address_string()} - {format % args}")


def main() -> None:
    print(f"VIQA API indexed {len(INDEX.passages)} sentence-aware passages from {DOCS_DB}")
    print(f"Serving http://localhost:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
