# -*- coding: utf-8 -*-
"""
BM25 Document Retriever — Okapi BM25 cho tiếng Việt.

Sử dụng thư viện `rank_bm25` với tokenization bằng pyvi.
Hỗ trợ 3 variants: BM25Okapi, BM25L, BM25Plus.

Reference: Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
"""

import os
import logging
import pickle
from typing import List, Dict, Tuple, Optional

import numpy as np

from .base import BaseRetriever

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Vietnamese tokenizer (shared with TF-IDF module)
# ──────────────────────────────────────────────────────────────────────────────

_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from pyvi import ViTokenizer
            _TOKENIZER = ViTokenizer
        except ImportError:
            raise ImportError("pyvi chưa được cài. Chạy: pip install pyvi")
    return _TOKENIZER


def tokenize_vi(text: str) -> List[str]:
    """Tokenize tiếng Việt bằng pyvi → list of lowercased tokens."""
    tok = _get_tokenizer()
    segmented = tok.tokenize(text)
    return [t.lower() for t in segmented.split()]


# ──────────────────────────────────────────────────────────────────────────────
# BM25 Retriever
# ──────────────────────────────────────────────────────────────────────────────


class BM25Retriever(BaseRetriever):
    """BM25 Document Retriever sử dụng rank_bm25.

    Args:
        variant: Loại BM25 — 'okapi' (default), 'l', hoặc 'plus'.
        k1: BM25 parameter k1 (term frequency saturation). Default=1.5.
        b: BM25 parameter b (document length normalization). Default=0.75.
        tokenizer_fn: Hàm tokenize text → list[str]. Default: pyvi tokenizer.
    """

    def __init__(
        self,
        variant: str = "okapi",
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer_fn=None,
    ):
        name_map = {"okapi": "BM25-Okapi", "l": "BM25L", "plus": "BM25Plus"}
        super().__init__(name=name_map.get(variant, f"BM25-{variant}"))
        self.variant = variant
        self.k1 = k1
        self.b = b
        self.tokenizer_fn = tokenizer_fn or tokenize_vi

        self.bm25 = None
        self._tokenized_corpus: Optional[List[List[str]]] = None

    def _get_bm25_class(self):
        """Import & return BM25 class tương ứng variant."""
        try:
            import rank_bm25
        except ImportError:
            raise ImportError(
                "rank_bm25 chưa được cài. Chạy: pip install rank-bm25"
            )

        if self.variant == "okapi":
            return rank_bm25.BM25Okapi
        elif self.variant == "l":
            return rank_bm25.BM25L
        elif self.variant == "plus":
            return rank_bm25.BM25Plus
        else:
            raise ValueError(f"Unknown BM25 variant: {self.variant}")

    def build_index(self, corpus: List[Dict[str, str]]) -> None:
        """Xây dựng BM25 index từ corpus.

        Args:
            corpus: List[{'id': str, 'text': str, 'title': str (optional)}]
        """
        logger.info(
            f"[{self.name}] Building index for {len(corpus)} documents..."
        )
        self.num_docs = len(corpus)
        self.doc_ids = [doc["id"] for doc in corpus]

        # Tokenize corpus
        logger.info(f"[{self.name}] Tokenizing corpus...")
        self._tokenized_corpus = []
        for i, doc in enumerate(corpus):
            text = doc.get("text", "")
            title = doc.get("title", "")
            full_text = f"{title} {text}" if title else text
            tokens = self.tokenizer_fn(full_text)
            self._tokenized_corpus.append(tokens)

            if (i + 1) % 500 == 0:
                logger.info(f"  Tokenized {i + 1}/{self.num_docs} docs")

        # Build BM25
        BM25Class = self._get_bm25_class()
        if self.variant == "okapi":
            self.bm25 = BM25Class(self._tokenized_corpus, k1=self.k1, b=self.b)
        elif self.variant == "l":
            self.bm25 = BM25Class(self._tokenized_corpus, k1=self.k1, b=self.b)
        elif self.variant == "plus":
            self.bm25 = BM25Class(self._tokenized_corpus, k1=self.k1, b=self.b)

        self.is_indexed = True
        logger.info(
            f"[{self.name}] Index built: {self.num_docs} docs, "
            f"avg_doc_len={self.bm25.avgdl:.1f}"
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k documents bằng BM25 scoring."""
        if not self.is_indexed:
            raise RuntimeError("Index chưa được build. Gọi build_index() trước.")

        query_tokens = self.tokenizer_fn(query)
        scores = self.bm25.get_scores(query_tokens)

        # Top-k
        if top_k >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))

        return results[:top_k]

    def save_index(self, path: str) -> None:
        """Lưu BM25 index qua pickle."""
        if not self.is_indexed:
            raise RuntimeError("Không có index để lưu.")

        save_data = {
            "bm25": self.bm25,
            "doc_ids": self.doc_ids,
            "num_docs": self.num_docs,
            "variant": self.variant,
            "k1": self.k1,
            "b": self.b,
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"[{self.name}] Index saved to {path}")

    def load_index(self, path: str) -> None:
        """Tải BM25 index từ pickle file."""
        with open(path, "rb") as f:
            save_data = pickle.load(f)

        self.bm25 = save_data["bm25"]
        self.doc_ids = save_data["doc_ids"]
        self.num_docs = save_data["num_docs"]
        self.variant = save_data["variant"]
        self.k1 = save_data["k1"]
        self.b = save_data["b"]
        self.is_indexed = True

        name_map = {"okapi": "BM25-Okapi", "l": "BM25L", "plus": "BM25Plus"}
        self.name = name_map.get(self.variant, f"BM25-{self.variant}")

        logger.info(
            f"[{self.name}] Index loaded: {self.num_docs} docs from {path}"
        )
