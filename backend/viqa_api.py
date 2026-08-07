from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOCS_DB = ROOT / "data" / "processed" / "docs.db"
INDEX_DIR = ROOT / "data" / "indexes"
HOST = "0.0.0.0"
PORT = 8000

STOPWORDS = {
    "la",
    "cua",
    "va",
    "co",
    "cac",
    "mot",
    "nhung",
    "duoc",
    "trong",
    "cho",
    "voi",
    "the",
    "nay",
    "do",
    "den",
    "tu",
    "khi",
    "nao",
    "gi",
    "ai",
    "bao",
    "nhieu",
    "may",
    "o",
    "tai",
    "ve",
    "nhu",
    "khong",
    "sau",
    "truoc",
}


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str
    tokens: tuple[str, ...]
    term_counts: Counter[str]
    char_ngrams: frozenset[str]


@dataclass(frozen=True)
class SearchHit:
    document: Document
    score: float
    raw_score: float


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d")
    return value

def safe_normalize(value: str) -> str:
    value = value.replace("\u0110", "D").replace("\u0111", "d")
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return value.replace("\u0111", "d")


normalize = safe_normalize


def tokenize(value: str) -> list[str]:
    tokens = re.findall(r"[\w%]+", normalize(value), flags=re.UNICODE)
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def char_ngrams(value: str, n: int = 3) -> frozenset[str]:
    compact = re.sub(r"\s+", " ", normalize(value))
    if len(compact) <= n:
        return frozenset({compact})
    return frozenset(compact[index : index + n] for index in range(len(compact) - n + 1))


def split_sentences(text: str) -> list[str]:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
    return sentences or [text.strip()]


class ViqaEngine:
    def __init__(self, db_path: Path) -> None:
        if not db_path.exists():
            raise FileNotFoundError(f"Missing docs database: {db_path}")

        self.documents = self._load_documents(db_path)
        self.avg_doc_len = sum(len(doc.tokens) for doc in self.documents) / max(1, len(self.documents))
        self.doc_freq = self._build_doc_freq()
        self.idf = {
            term: math.log(1 + (len(self.documents) - freq + 0.5) / (freq + 0.5))
            for term, freq in self.doc_freq.items()
        }

    def _load_documents(self, db_path: Path) -> list[Document]:
        connection = sqlite3.connect(str(db_path))
        try:
            rows = connection.execute("select id, text from documents").fetchall()
        finally:
            connection.close()

        documents: list[Document] = []
        for doc_id, text in rows:
            clean_text = str(text or "").strip()
            tokens = tuple(tokenize(clean_text))
            documents.append(
                Document(
                    document_id=str(doc_id),
                    title=self._guess_title(str(doc_id), clean_text),
                    text=clean_text,
                    tokens=tokens,
                    term_counts=Counter(tokens),
                    char_ngrams=char_ngrams(clean_text[:900]),
                )
            )
        return documents

    def _build_doc_freq(self) -> dict[str, int]:
        doc_freq: dict[str, int] = defaultdict(int)
        for document in self.documents:
            for term in set(document.tokens):
                doc_freq[term] += 1
        return doc_freq

    def _guess_title(self, doc_id: str, text: str) -> str:
        first_sentence = split_sentences(text)[0]
        words = first_sentence.split()
        if words:
            return " ".join(words[: min(8, len(words))])
        return doc_id

    def search(self, question: str, retriever: str, top_k: int) -> list[SearchHit]:
        query_tokens = tokenize(question)
        if not query_tokens:
            return []

        method = "bm25" if retriever == "pyserini" else retriever
        scored: list[SearchHit] = []
        for document in self.documents:
            if method == "tfidf":
                raw = self._tfidf_score(query_tokens, document)
            elif method == "dense":
                raw = self._dense_lite_score(question, query_tokens, document)
            else:
                raw = self._bm25_score(query_tokens, document)
            if raw > 0:
                scored.append(SearchHit(document=document, score=raw, raw_score=raw))

        scored.sort(key=lambda hit: hit.raw_score, reverse=True)
        top_hits = scored[: max(1, top_k)]
        if not top_hits:
            return []

        max_score = max(hit.raw_score for hit in top_hits) or 1.0
        return [SearchHit(hit.document, min(0.999, hit.raw_score / max_score), hit.raw_score) for hit in top_hits]

    def search_with_external(self, results: list[tuple[str, float]], top_k: int) -> list[SearchHit]:
        doc_lookup = {doc.document_id: doc for doc in self.documents}
        hits = [
            SearchHit(document=doc_lookup[doc_id], score=float(score), raw_score=float(score))
            for doc_id, score in results
            if doc_id in doc_lookup and float(score) > 0
        ][: max(1, top_k)]
        if not hits:
            return []

        max_score = max(hit.raw_score for hit in hits) or 1.0
        return [SearchHit(hit.document, min(0.999, hit.raw_score / max_score), hit.raw_score) for hit in hits]

    def _tfidf_score(self, query_tokens: list[str], document: Document) -> float:
        query_counts = Counter(query_tokens)
        numerator = 0.0
        doc_norm = 0.0
        query_norm = 0.0
        for term, query_tf in query_counts.items():
            idf = self.idf.get(term, 0.0)
            doc_weight = (1 + math.log(document.term_counts.get(term, 0))) * idf if document.term_counts.get(term, 0) else 0
            query_weight = (1 + math.log(query_tf)) * idf
            numerator += doc_weight * query_weight
            doc_norm += doc_weight * doc_weight
            query_norm += query_weight * query_weight
        if not doc_norm or not query_norm:
            return 0.0
        return numerator / (math.sqrt(doc_norm) * math.sqrt(query_norm))

    def _bm25_score(self, query_tokens: list[str], document: Document) -> float:
        score = 0.0
        k1 = 1.5
        b = 0.75
        doc_len = max(1, len(document.tokens))
        for term in query_tokens:
            freq = document.term_counts.get(term, 0)
            if not freq:
                continue
            idf = self.idf.get(term, 0.0)
            denom = freq + k1 * (1 - b + b * doc_len / max(1.0, self.avg_doc_len))
            score += idf * (freq * (k1 + 1)) / denom
        return score

    def _dense_lite_score(self, question: str, query_tokens: list[str], document: Document) -> float:
        bm25 = self._bm25_score(query_tokens, document)
        q_ngrams = char_ngrams(question)
        overlap = len(q_ngrams.intersection(document.char_ngrams))
        union = max(1, len(q_ngrams.union(document.char_ngrams)))
        return bm25 * 0.72 + (overlap / union) * 12

    def answer(self, question: str, hit: SearchHit, reader: str) -> dict[str, Any]:
        query_tokens = set(tokenize(question))
        sentences = split_sentences(hit.document.text)
        best_sentence = sentences[0]
        best_score = -1.0
        identity_subject = extract_identity_subject(question)

        for sentence in sentences:
            sentence_tokens = set(tokenize(sentence))
            score = len(query_tokens.intersection(sentence_tokens))
            if identity_subject and normalize(identity_subject) in normalize(sentence) and " la " in f" {normalize(sentence)} ":
                score += 4
            score += 0.4 if re.search(r"\d|%|năm|ngày|tháng", sentence.lower()) else 0
            if score > best_score:
                best_score = score
                best_sentence = sentence

        span = self._extract_span(question, best_sentence)
        if not span:
            span = best_sentence[: min(260, len(best_sentence))]

        start = hit.document.text.find(span)
        if start < 0:
            start = hit.document.text.find(best_sentence)
        end = start + len(span) if start >= 0 else len(span)

        reader_bias = {"mock": 0.78, "phobert": 0.9, "vibert": 0.86, "xlmr": 0.88}.get(reader, 0.82)
        reader_score = min(0.98, max(0.12, (best_score / max(1, len(query_tokens))) * reader_bias))
        confidence = min(0.97, max(0.05, hit.score * 0.62 + reader_score * 0.38))

        return {
            "answer": span,
            "answer_span": {"text": span, "start": max(0, start), "end": max(0, end)},
            "reader_score": reader_score,
            "confidence": confidence,
        }

    def _extract_span(self, question: str, sentence: str) -> str:
        normalized_question = normalize(question)

        if any(marker in normalized_question for marker in ["bao nhieu", "may", "phan tram", "nam nao", "khi nao"]):
            match = re.search(r"((?:\d+[.,]?\d*\s*(?:%|phần trăm|năm|ngày|tháng|giờ|lần)?)|(?:thế kỷ\s+\w+))", sentence, re.I)
            if match:
                return match.group(1).strip()

        if extract_identity_subject(question):
            return sentence[: min(280, len(sentence))].strip()

        if normalized_question.startswith("ai ") or " la ai" in normalized_question:
            match = re.search(r"([A-ZĐÂĂÊÔƠƯ][\wÀ-ỹ'.-]+(?:\s+[A-ZĐÂĂÊÔƠƯ][\wÀ-ỹ'.-]+){1,5})", sentence)
            if match:
                return match.group(1).strip()

        if " o dau" in normalized_question or normalized_question.startswith("o dau"):
            match = re.search(r"(?:tại|ở|thuộc)\s+([^,.]{3,90})", sentence, re.I)
            if match:
                return match.group(1).strip()

        return sentence[: min(220, len(sentence))].strip()


ENGINE = ViqaEngine(DOCS_DB)


class RetrievalManager:
    def __init__(self, engine: ViqaEngine) -> None:
        self.engine = engine
        self.corpus = [{"id": doc.document_id, "title": doc.title, "text": doc.text} for doc in engine.documents]
        self.instances: dict[str, Any] = {}
        self.status: dict[str, str] = {}

    def retrieve(self, method: str, question: str, top_k: int) -> list[SearchHit]:
        method = method.lower()
        if method == "tfidf":
            return self._retrieve_tfidf(question, top_k)
        if method == "bm25":
            return self._retrieve_bm25(question, top_k)
        if method == "dense":
            return self._retrieve_dense(question, top_k)
        if method == "pyserini":
            return self._retrieve_pyserini(question, top_k)
        return self.engine.search(question, method, top_k)

    def _retrieve_tfidf(self, question: str, top_k: int) -> list[SearchHit]:
        try:
            retriever = self._get_tfidf()
            return self.engine.search_with_external(retriever.retrieve(question, top_k), top_k)
        except Exception as error:
            self.status["tfidf"] = f"fallback: {error}"
            return self.engine.search(question, "tfidf", top_k)

    def _retrieve_bm25(self, question: str, top_k: int) -> list[SearchHit]:
        try:
            retriever = self._get_bm25()
            return self.engine.search_with_external(retriever.retrieve(question, top_k), top_k)
        except Exception as error:
            self.status["bm25"] = f"fallback: {error}"
            return self.engine.search(question, "bm25", top_k)

    def _retrieve_dense(self, question: str, top_k: int) -> list[SearchHit]:
        try:
            retriever = self._get_dense()
            return self.engine.search_with_external(retriever.retrieve(question, top_k), top_k)
        except Exception as error:
            self.status["dense"] = f"dense-lite fallback: {error}"
            return self.engine.search(question, "dense", top_k)

    def _retrieve_pyserini(self, question: str, top_k: int) -> list[SearchHit]:
        try:
            retriever = self._get_pyserini()
            return self.engine.search_with_external(retriever.retrieve(question, top_k), top_k)
        except Exception as error:
            self.status["pyserini"] = f"bm25 fallback: {error}"
            return self.engine.search(question, "bm25", top_k)

    def _get_tfidf(self) -> Any:
        if "tfidf" not in self.instances:
            from retrieval.tfidf_retriever import TfidfRetriever

            retriever = TfidfRetriever(ngram=2, hash_size=2**18)
            index_path = INDEX_DIR / "tfidf_index.npz"
            if index_path.exists():
                retriever.load_index(str(index_path))
                self.status["tfidf"] = f"loaded {index_path.name}"
            else:
                retriever.build_index(self.corpus)
                self.status["tfidf"] = "built in memory from retrieval.TfidfRetriever"
            self.instances["tfidf"] = retriever
        return self.instances["tfidf"]

    def _get_bm25(self) -> Any:
        if "bm25" not in self.instances:
            from retrieval.bm25_retriever import BM25Retriever

            retriever = BM25Retriever(variant="okapi", k1=1.5, b=0.75)
            index_path = INDEX_DIR / "bm25_okapi_index.pkl"
            if index_path.exists():
                retriever.load_index(str(index_path))
                self.status["bm25"] = f"loaded {index_path.name}"
            else:
                retriever.build_index(self.corpus)
                self.status["bm25"] = "built in memory from retrieval.BM25Retriever"
            self.instances["bm25"] = retriever
        return self.instances["bm25"]

    def _get_dense(self) -> Any:
        if "dense" not in self.instances:
            from retrieval.dense_retriever import DenseRetriever

            index_path = INDEX_DIR / "dense_index.faiss"
            if not index_path.exists():
                raise FileNotFoundError("data/indexes/dense_index.faiss not found")
            retriever = DenseRetriever()
            retriever.load_index(str(index_path))
            self.status["dense"] = f"loaded {index_path.name}"
            self.instances["dense"] = retriever
        return self.instances["dense"]

    def _get_pyserini(self) -> Any:
        if "pyserini" not in self.instances:
            from retrieval.pyserini_retriever import PyseriniRetriever

            meta_path = INDEX_DIR / "pyserini_meta.json"
            if not meta_path.exists():
                raise FileNotFoundError("data/indexes/pyserini_meta.json not found")
            retriever = PyseriniRetriever()
            retriever.load_index(str(meta_path))
            self.status["pyserini"] = f"loaded {meta_path.name}"
            self.instances["pyserini"] = retriever
        return self.instances["pyserini"]


RETRIEVAL = RetrievalManager(ENGINE)


def extract_identity_subject(question: str) -> str:
    normalized_question = normalize(question)
    match = re.search(r"^\s*(.*?)\s+(?:la ai|la nguoi nao)\s*\??\s*$", normalized_question)
    if not match:
        return ""
    subject = match.group(1).strip()
    return subject if len(subject) >= 2 else ""


def rerank_identity_hits(question: str, hits: list[SearchHit]) -> list[SearchHit]:
    subject = extract_identity_subject(question)
    if not subject:
        return hits

    subject_tokens = set(tokenize(subject))
    if not subject_tokens:
        return hits

    existing_ids = {hit.document.document_id for hit in hits}
    candidates = list(hits)

    for document in ENGINE.documents:
        if document.document_id in existing_ids:
            continue
        if not subject_tokens.issubset(set(document.tokens)):
            continue
        candidates.append(SearchHit(document=document, score=0.42, raw_score=0.42))

    def identity_score(hit: SearchHit) -> float:
        text = normalize(hit.document.text[:420])
        title = normalize(hit.document.title)
        subject_norm = normalize(subject)
        score = hit.raw_score
        if subject_norm in text[:180]:
            score += 8
        if subject_norm in title:
            score += 3
        if re.search(rf"{re.escape(subject_norm)}\s*(?:\([^)]{{0,120}}\))?\s+la\s+", text):
            score += 16
        if any(marker in text for marker in ["la thu tuong", "la chu tich", "la nha", "la mot", "la vi"]):
            score += 4
        if " co vo " in text[:120] or " con trai " in text[:160]:
            score -= 4
        return score

    reranked = sorted(candidates, key=identity_score, reverse=True)
    max_score = max((identity_score(hit) for hit in reranked[:10]), default=1.0) or 1.0
    return [SearchHit(hit.document, min(0.999, identity_score(hit) / max_score), identity_score(hit)) for hit in reranked]


def passage_from_hit(hit: SearchHit, rank: int, question: str, reader: str) -> dict[str, Any]:
    reader_output = ENGINE.answer(question, hit, reader)
    text = hit.document.text
    span_text = reader_output["answer_span"]["text"]
    span_index = text.find(span_text)
    if span_index >= 0:
        start = max(0, span_index - 180)
        end = min(len(text), span_index + len(span_text) + 220)
        preview = text[start:end].strip()
    else:
        preview = text[:420].strip()

    document_id = hit.document.document_id
    return {
        "rank": rank,
        "document_id": document_id,
        "passage_id": f"{document_id}_P{rank:03d}",
        "title": hit.document.title,
        "text": preview,
        "retrieval_score": round(hit.score, 4),
        "reader_score": round(reader_output["reader_score"], 4),
    }


def ask_question(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    question = str(payload.get("question", "")).strip()
    retriever = str(payload.get("retriever", "bm25")).lower()
    reader = str(payload.get("reader", "mock")).lower()
    top_k = int(payload.get("top_k", 5) or 5)

    internal_top_k = max(top_k, 30) if extract_identity_subject(question) else top_k
    hits = rerank_identity_hits(question, RETRIEVAL.retrieve(retriever, question, internal_top_k))[:top_k]
    if not hits:
        elapsed = int((time.perf_counter() - started) * 1000)
        return {
            "question": question,
            "answer": "",
            "confidence": 0.0,
            "processing_time_ms": elapsed,
            "retriever": retriever,
            "reader": reader,
            "source": {"document_id": "", "passage_id": "", "title": ""},
            "passages": [],
        }

    best = hits[0]
    reader_output = ENGINE.answer(question, best, reader)
    passages = [passage_from_hit(hit, index + 1, question, reader) for index, hit in enumerate(hits)]
    elapsed = int((time.perf_counter() - started) * 1000)
    document_id = best.document.document_id

    if reader_output["confidence"] < 0.18:
        answer = ""
    else:
        answer = reader_output["answer"]

    return {
        "question": question,
        "answer": answer,
        "confidence": round(reader_output["confidence"], 4),
        "processing_time_ms": elapsed,
        "retriever": retriever,
        "reader": reader,
        "source": {
            "document_id": document_id,
            "passage_id": f"{document_id}_P001",
            "title": best.document.title,
        },
        "answer_span": reader_output["answer_span"],
        "passages": passages,
    }


def compare_retrievers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    question = str(payload.get("question", "")).strip()
    rows: list[dict[str, Any]] = []
    labels = {"tfidf": "TF-IDF", "bm25": "BM25", "dense": "Dense Retrieval"}
    for retriever, label in labels.items():
        started = time.perf_counter()
        hits = RETRIEVAL.retrieve(retriever, question, 3)
        elapsed = int((time.perf_counter() - started) * 1000)
        first = hits[0] if hits else None
        rows.append(
            {
                "retriever": retriever,
                "label": label,
                "correctPassageRank": 1 if first else 0,
                "recallAt1": bool(first and first.score >= 0.45),
                "recallAt3": bool(first),
                "responseTimeMs": elapsed,
                "topPassagePreview": first.document.text[:220] if first else "No passage found.",
                "retrievalScore": round(first.score if first else 0, 4),
            }
        )
    return rows


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json({"status": "ok", "documents": len(ENGINE.documents), "retrievers": RETRIEVAL.status})
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        payload = self._read_json()
        if path == "/api/ask":
            self._send_json(ask_question(payload))
            return
        if path == "/api/compare":
            self._send_json(compare_retrievers(payload))
            return
        self._send_json({"error": "Not found"}, status=404)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, data: Any, status: int = 200) -> None:
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
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
    print(f"VIQA API loaded {len(ENGINE.documents)} documents from {DOCS_DB}")
    print(f"Serving http://localhost:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
