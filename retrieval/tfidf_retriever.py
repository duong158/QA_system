# -*- coding: utf-8 -*-
"""
TF-IDF Document Retriever — Reimplementation theo DrQA gốc (Facebook Research).

Kiến trúc:
  1. Tokenize văn bản tiếng Việt bằng pyvi (thay CoreNLP)
  2. Trích xuất n-grams (default: bigrams)
  3. Feature hashing bằng MurmurHash3 (sklearn)
  4. Xây dựng TF-IDF sparse matrix (scipy CSR)
  5. Scoring bằng sparse dot product: query_vec · doc_matrix

Reference: https://github.com/facebookresearch/DrQA/blob/main/drqa/retriever/tfidf_doc_ranker.py
"""

import os
import re
import logging
import pickle
import unicodedata
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32

from .base import BaseRetriever

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Vietnamese Tokenizer wrapper (pyvi)
# ──────────────────────────────────────────────────────────────────────────────

_TOKENIZER = None


def _get_tokenizer():
    """Lazy-load pyvi tokenizer."""
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from pyvi import ViTokenizer
            _TOKENIZER = ViTokenizer
        except ImportError:
            raise ImportError(
                "pyvi chưa được cài. Chạy: pip install pyvi"
            )
    return _TOKENIZER


def tokenize_vi(text: str) -> List[str]:
    """Tokenize văn bản tiếng Việt → list of tokens (từ đơn & từ ghép).

    pyvi tách từ ghép bằng dấu '_':  "Hà Nội" → "Hà_Nội"
    """
    tok = _get_tokenizer()
    segmented = tok.tokenize(text)
    return segmented.split()


# ──────────────────────────────────────────────────────────────────────────────
# Text normalization (theo DrQA retriever/utils.py)
# ──────────────────────────────────────────────────────────────────────────────

# Stopwords tiếng Việt phổ biến (cho retriever filtering)
VIETNAMESE_STOPWORDS = {
    "và", "của", "là", "có", "được", "trong", "để", "các", "cho", "với",
    "này", "một", "những", "đã", "từ", "không", "cũng", "như", "về",
    "theo", "khi", "trên", "tại", "hay", "đến", "do", "bị", "vào",
    "ra", "lên", "mà", "nên", "nhưng", "còn", "hoặc", "thì", "rất",
    "sẽ", "đang", "bởi", "nếu", "vì", "hơn", "chỉ", "sau", "qua",
    "lại", "nhiều", "nào", "đó", "giữa", "dưới", "trước", "phải",
}

PUNCT_RE = re.compile(r"^[\W]+$", re.UNICODE)


def normalize_text(text: str) -> str:
    """Chuẩn hóa text giống DrQA: NFC, lowercase (tùy chọn), strip."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_ngram(gram: str) -> bool:
    """Lọc n-gram: bỏ stopwords, punctuation, chuỗi quá ngắn.

    Giống DrQA retriever.utils.filter_ngram.
    """
    parts = gram.split()
    if any(PUNCT_RE.match(p) for p in parts):
        return True
    if all(p.lower() in VIETNAMESE_STOPWORDS for p in parts):
        return True
    if len(gram) < 2:
        return True
    return False


def hash_token(token: str, num_buckets: int) -> int:
    """MurmurHash3 feature hashing (unsigned 32-bit) — giống DrQA."""
    return murmurhash3_32(token, positive=True) % num_buckets


def extract_ngrams(tokens: List[str], n: int = 2, uncased: bool = True) -> List[str]:
    """Trích xuất n-grams từ list of tokens.

    Bao gồm unigrams + bigrams (nếu n=2), filtered by stopword/punct.
    """
    if uncased:
        tokens = [t.lower() for t in tokens]

    ngrams = []
    # Unigrams
    for t in tokens:
        if not filter_ngram(t):
            ngrams.append(t)

    # Higher-order n-grams
    for i in range(2, n + 1):
        for j in range(len(tokens) - i + 1):
            gram = " ".join(tokens[j : j + i])
            if not filter_ngram(gram):
                ngrams.append(gram)

    return ngrams


# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF Retriever Class
# ──────────────────────────────────────────────────────────────────────────────


class TfidfRetriever(BaseRetriever):
    """TF-IDF Document Retriever theo kiến trúc DrQA.

    Xây dựng sparse TF-IDF matrix bằng hashed n-grams,
    scoring bằng sparse dot product.

    Args:
        ngram: Bậc n-gram tối đa (default=2 → unigram + bigram).
        hash_size: Kích thước hash table (default=2^24).
        tokenizer_fn: Hàm tokenize text → list[str]. Default: pyvi tokenizer.
    """

    def __init__(
        self,
        ngram: int = 2,
        hash_size: int = 2**24,
        tokenizer_fn=None,
    ):
        super().__init__(name="TF-IDF (DrQA)")
        self.ngram = ngram
        self.hash_size = hash_size
        self.tokenizer_fn = tokenizer_fn or tokenize_vi

        # Index state
        self.doc_mat: Optional[sp.csr_matrix] = None  # (num_docs, hash_size) TF-IDF
        self.doc_freqs: Optional[np.ndarray] = None    # Document frequencies per hash bucket
        self.doc_dict: Dict[str, int] = {}             # doc_id → column index
        self.idx2doc: Dict[int, str] = {}              # column index → doc_id

    def build_index(self, corpus: List[Dict[str, str]]) -> None:
        """Xây dựng TF-IDF sparse matrix từ corpus.

        Args:
            corpus: List[{'id': str, 'text': str, 'title': str (optional)}]
        """
        logger.info(f"[TF-IDF] Building index for {len(corpus)} documents...")
        self.num_docs = len(corpus)
        self.doc_ids = [doc["id"] for doc in corpus]
        self.doc_dict = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        self.idx2doc = {idx: doc_id for doc_id, idx in self.doc_dict.items()}

        # ── Bước 1: Tokenize & count hashed n-grams cho mỗi document ──
        row_indices = []
        col_indices = []
        data_values = []

        doc_freq_counter = Counter()  # Đếm document frequency cho mỗi hash bucket

        for doc_idx, doc in enumerate(corpus):
            text = normalize_text(doc.get("text", ""))
            title = normalize_text(doc.get("title", ""))
            full_text = f"{title} {text}" if title else text

            tokens = self.tokenizer_fn(full_text)
            ngrams = extract_ngrams(tokens, n=self.ngram, uncased=True)

            # Hash n-grams → count
            hashed_counts = Counter(
                [hash_token(gram, self.hash_size) for gram in ngrams]
            )

            # Track document frequency (mỗi hash chỉ đếm 1 lần per doc)
            doc_freq_counter.update(hashed_counts.keys())

            # Sparse matrix entries (doc_idx = row, hash_bucket = column)
            for h, cnt in hashed_counts.items():
                row_indices.append(doc_idx)
                col_indices.append(h)
                data_values.append(cnt)

            if (doc_idx + 1) % 500 == 0:
                logger.info(f"  Processed {doc_idx + 1}/{self.num_docs} docs")

        # ── Bước 2: Xây dựng TF-IDF matrix ──
        # Raw count matrix
        count_matrix = sp.csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(self.num_docs, self.hash_size),
            dtype=np.float64,
        )

        # TF = log(1 + count)
        count_matrix.data = np.log1p(count_matrix.data)

        # IDF = log((N - df + 0.5) / (df + 0.5))  (DrQA formula)
        self.doc_freqs = np.zeros(self.hash_size, dtype=np.float64)
        for h, df in doc_freq_counter.items():
            self.doc_freqs[h] = df

        N = self.num_docs
        idf = np.log((N - self.doc_freqs + 0.5) / (self.doc_freqs + 0.5))
        idf[idf < 0] = 0  # Clip negative IDF

        # TF-IDF = TF * IDF  (element-wise trên mỗi hàng)
        # Nhân mỗi cột của count_matrix với IDF tương ứng
        idf_diag = sp.diags(idf, 0, shape=(self.hash_size, self.hash_size))
        self.doc_mat = count_matrix.dot(idf_diag).tocsr()

        self.is_indexed = True
        logger.info(
            f"[TF-IDF] Index built: {self.num_docs} docs, "
            f"matrix shape={self.doc_mat.shape}, "
            f"nnz={self.doc_mat.nnz}"
        )

    def _query_to_vector(self, query: str) -> sp.csr_matrix:
        """Chuyển query text thành sparse TF-IDF vector.

        Returns:
            Sparse vector shape (1, hash_size).
        """
        tokens = self.tokenizer_fn(normalize_text(query))
        ngrams = extract_ngrams(tokens, n=self.ngram, uncased=True)
        hashed_counts = Counter(
            [hash_token(gram, self.hash_size) for gram in ngrams]
        )

        if not hashed_counts:
            return sp.csr_matrix((1, self.hash_size), dtype=np.float64)

        cols = list(hashed_counts.keys())
        data = [np.log1p(cnt) for cnt in hashed_counts.values()]

        # Apply IDF
        N = self.num_docs
        for i, col in enumerate(cols):
            df = self.doc_freqs[col]
            idf = np.log((N - df + 0.5) / (df + 0.5))
            if idf < 0:
                idf = 0
            data[i] *= idf

        rows = [0] * len(cols)
        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(1, self.hash_size),
            dtype=np.float64,
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k documents bằng sparse dot product.

        Score = query_vector · doc_matrix.T
        """
        if not self.is_indexed:
            raise RuntimeError("Index chưa được build. Gọi build_index() trước.")

        query_vec = self._query_to_vector(query)
        if query_vec.nnz == 0:
            return []

        # Dot product: (1, H) · (N, H).T = (1, N)
        scores = query_vec.dot(self.doc_mat.T).toarray().flatten()

        # Top-k indices
        if top_k >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.idx2doc[idx], float(scores[idx])))

        return results[:top_k]

    def batch_retrieve(
        self, queries: List[str], top_k: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """Batch retrieve sử dụng matrix multiplication."""
        if not self.is_indexed:
            raise RuntimeError("Index chưa được build. Gọi build_index() trước.")

        # Build query matrix
        all_rows, all_cols, all_data = [], [], []
        for q_idx, query in enumerate(queries):
            tokens = self.tokenizer_fn(normalize_text(query))
            ngrams = extract_ngrams(tokens, n=self.ngram, uncased=True)
            hashed_counts = Counter(
                [hash_token(gram, self.hash_size) for gram in ngrams]
            )
            N = self.num_docs
            for h, cnt in hashed_counts.items():
                tf = np.log1p(cnt)
                df = self.doc_freqs[h]
                idf = max(0, np.log((N - df + 0.5) / (df + 0.5)))
                all_rows.append(q_idx)
                all_cols.append(h)
                all_data.append(tf * idf)

        query_mat = sp.csr_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(len(queries), self.hash_size),
            dtype=np.float64,
        )

        # Batch dot product: (Q, H) · (N, H).T = (Q, N)
        score_mat = query_mat.dot(self.doc_mat.T).toarray()

        results = []
        for q_idx in range(len(queries)):
            scores = score_mat[q_idx]
            if top_k >= len(scores):
                top_indices = np.argsort(-scores)
            else:
                top_indices = np.argpartition(-scores, top_k)[:top_k]
                top_indices = top_indices[np.argsort(-scores[top_indices])]

            r = []
            for idx in top_indices:
                if scores[idx] > 0:
                    r.append((self.idx2doc[idx], float(scores[idx])))
            results.append(r[:top_k])

        return results

    def save_index(self, path: str) -> None:
        """Lưu TF-IDF index (sparse matrix + metadata) ra file .npz."""
        if not self.is_indexed:
            raise RuntimeError("Không có index để lưu.")

        data = {
            "data": self.doc_mat.data,
            "indices": self.doc_mat.indices,
            "indptr": self.doc_mat.indptr,
            "shape": self.doc_mat.shape,
        }
        metadata = {
            "ngram": self.ngram,
            "hash_size": self.hash_size,
            "doc_freqs": self.doc_freqs,
            "doc_ids": self.doc_ids,
            "doc_dict": self.doc_dict,
            "num_docs": self.num_docs,
        }
        np.savez(path, **data)

        meta_path = path.replace(".npz", "_meta.pkl")
        if not meta_path.endswith(".pkl"):
            meta_path = path + "_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"[TF-IDF] Index saved to {path}")

    def load_index(self, path: str) -> None:
        """Tải TF-IDF index từ file .npz."""
        loader = np.load(path, allow_pickle=True)
        self.doc_mat = sp.csr_matrix(
            (loader["data"], loader["indices"], loader["indptr"]),
            shape=tuple(loader["shape"]),
        )

        meta_path = path.replace(".npz", "_meta.pkl")
        if not meta_path.endswith(".pkl"):
            meta_path = path + "_meta.pkl"
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        self.ngram = metadata["ngram"]
        self.hash_size = metadata["hash_size"]
        self.doc_freqs = metadata["doc_freqs"]
        self.doc_ids = metadata["doc_ids"]
        self.doc_dict = metadata["doc_dict"]
        self.num_docs = metadata["num_docs"]
        self.idx2doc = {idx: doc_id for doc_id, idx in self.doc_dict.items()}
        self.is_indexed = True

        logger.info(
            f"[TF-IDF] Index loaded: {self.num_docs} docs, "
            f"matrix shape={self.doc_mat.shape}"
        )
