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
RETRIEVER_WEIGHT = float(os.getenv("QA_RETRIEVER_WEIGHT", "0.30"))
READER_WEIGHT = float(os.getenv("QA_READER_WEIGHT", "0.70"))
ANSWER_THRESHOLD = float(os.getenv("QA_ANSWER_THRESHOLD", "0.30"))
QA_DEBUG = os.getenv("QA_DEBUG", "false").lower() in {"1", "true", "yes", "on"}

if RETRIEVER_WEIGHT < 0 or READER_WEIGHT < 0 or RETRIEVER_WEIGHT + READER_WEIGHT <= 0:
    raise ValueError("QA retriever/reader weights must be non-negative and have a positive sum")

WEIGHT_TOTAL = RETRIEVER_WEIGHT + READER_WEIGHT
RETRIEVER_WEIGHT /= WEIGHT_TOTAL
READER_WEIGHT /= WEIGHT_TOTAL

STOPWORDS = {
    "ai", "bao", "cac", "cho", "co", "cua", "den", "do", "duoc", "gi", "khi",
    "khong", "la", "may", "mot", "nao", "nhieu", "nhu", "nhung", "o", "sau",
    "tai", "the", "trong", "truoc", "tu", "va", "ve", "voi",
}


class PipelineError(RuntimeError):
    pass


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


def normalize_text(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    value = unicodedata.normalize("NFD", value.lower())
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")


def tokenize(value: str) -> list[str]:
    tokens = re.findall(r"[\w%]+", normalize_text(value), flags=re.UNICODE)
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


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
                tokens = tuple(tokenize(f"{passage.title} {passage.text}"))
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

    def retrieve(self, question: str, method: str, top_k: int) -> list[SearchHit]:
        query_tokens = tokenize(question)
        if not query_tokens:
            return []
        if method not in {"bm25", "tfidf"}:
            raise PipelineError(
                f"Retriever '{method}' is unavailable: no compatible passage-level index/model is configured"
            )

        scored: list[tuple[IndexedPassage, float]] = []
        scorer = self._bm25 if method == "bm25" else self._tfidf
        for passage in self.passages:
            score = scorer(query_tokens, passage)
            if score > 0:
                scored.append((passage, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = scored[:top_k]
        normalized = min_max_normalize([score for _, score in selected])
        return [
            SearchHit(passage, raw, norm)
            for (passage, raw), norm in zip(selected, normalized)
        ]


class ReaderManager:
    MODEL_FOLDERS = {
        "phobert": "vinai_phobert-base-v2",
        "vibert": "vibert-base-v2",
        "xlmr": "xlm-roberta-base",
    }

    def __init__(self) -> None:
        self.predictors: dict[str, Any] = {}

    def get(self, reader_name: str):
        if reader_name == "mock":
            raise PipelineError("Mock Reader is forbidden in the real API")
        folder = self.MODEL_FOLDERS.get(reader_name)
        if not folder:
            raise PipelineError(f"Unknown reader '{reader_name}'")
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

    hits = INDEX.retrieve(question, retriever, top_k)
    if not hits:
        return _empty_response(
            question,
            retriever,
            reader_name,
            int((time.perf_counter() - started) * 1000),
        )

    predictor = READERS.get(reader_name)
    passages: list[dict[str, Any]] = []
    for retrieval_rank, hit in enumerate(hits, start=1):
        output = predictor.predict(
            question,
            hit.passage.metadata.text,
            no_answer_threshold=ANSWER_THRESHOLD,
        )
        reader_score = float(output["confidence"])
        final_score = RETRIEVER_WEIGHT * hit.retrieval_score_normalized + READER_WEIGHT * reader_score
        metadata = hit.passage.metadata
        passages.append(
            {
                "rank": retrieval_rank,
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
                "reader_answer": output["answer"] or None,
                "reader_score": round(reader_score, 6),
                "reader_score_raw": finite_or_none(output["score"]),
                "reader_null_score": finite_or_none(output["null_score"]),
                "reader_score_margin": finite_or_none(output["score_margin"]),
                "answer_span": {
                    "text": output["answer"],
                    "start": int(output["start"]),
                    "end": int(output["end"]),
                },
                "final_score": round(final_score, 6),
            }
        )

    selected = max(passages, key=lambda item: item["final_score"])
    has_answer = bool(selected["reader_answer"]) and selected["reader_score"] >= ANSWER_THRESHOLD
    elapsed = int((time.perf_counter() - started) * 1000)
    source = {
        "document_id": selected["document_id"],
        "passage_id": selected["passage_id"],
        "title": selected["title"],
        "paragraph_id": selected["paragraph_id"],
        "sentence_start": selected["sentence_start"],
        "sentence_end": selected["sentence_end"],
        "page": selected["page"],
    }
    response = {
        "question": question,
        "answer": selected["reader_answer"] if has_answer else None,
        "has_answer": has_answer,
        "confidence": selected["reader_score"],
        "selected_passage_id": selected["passage_id"],
        "processing_time_ms": elapsed,
        "retriever": retriever,
        "reader": reader_name,
        "source": source,
        "answer_span": selected["answer_span"] if has_answer else None,
        "passages": passages,
        "scoring": {
            "retriever_weight": RETRIEVER_WEIGHT,
            "reader_weight": READER_WEIGHT,
            "answer_threshold": ANSWER_THRESHOLD,
            "retrieval_normalization": "min_max_within_top_k",
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
                    "config": {
                        "chunk_max_tokens": CHUNK_MAX_TOKENS,
                        "chunk_overlap_sentences": CHUNK_OVERLAP_SENTENCES,
                        "retriever_weight": RETRIEVER_WEIGHT,
                        "reader_weight": READER_WEIGHT,
                        "answer_threshold": ANSWER_THRESHOLD,
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
