# -*- coding: utf-8 -*-
"""
Dense Document Retriever — Sentence-Transformers + FAISS cho tiếng Việt.

Sử dụng pre-trained Vietnamese sentence embedding models để encode
documents & queries thành dense vectors, sau đó tìm kiếm bằng FAISS.

Supported models:
  - keepitreal/vietnamese-sbert (default, lightweight)
  - bkai-foundation-models/vietnamese-bi-encoder
  - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

from .base import BaseRetriever

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """Dense Document Retriever sử dụng sentence-transformers + FAISS.

    Args:
        model_name: Tên model trên HuggingFace Hub.
        batch_size: Batch size khi encode documents/queries.
        device: 'cuda', 'cpu', hoặc None (auto-detect).
        normalize: Normalize embeddings trước khi index (dùng cosine similarity).
    """

    def __init__(
        self,
        model_name: str = "keepitreal/vietnamese-sbert",
        batch_size: int = 64,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        super().__init__(name=f"Dense ({model_name.split('/')[-1]})")
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize

        self.model = None
        self.index = None  # FAISS index
        self.embedding_dim: int = 0

    def _load_model(self):
        """Lazy-load sentence-transformers model."""
        if self.model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers chưa được cài. "
                "Chạy: pip install sentence-transformers"
            )

        logger.info(f"[Dense] Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        try:
            self.embedding_dim = self.model.get_embedding_dimension()
        except AttributeError:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"[Dense] Model loaded: dim={self.embedding_dim}, "
            f"device={self.model.device}"
        )

    def _get_faiss(self):
        """Import FAISS."""
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError(
                "faiss chưa được cài. "
                "Chạy: pip install faiss-cpu  (hoặc faiss-gpu nếu có CUDA)"
            )

    def build_index(self, corpus: List[Dict[str, str]]) -> None:
        """Xây dựng FAISS index từ corpus.

        Args:
            corpus: List[{'id': str, 'text': str, 'title': str (optional)}]
        """
        self._load_model()
        faiss = self._get_faiss()

        self.num_docs = len(corpus)
        self.doc_ids = [doc["id"] for doc in corpus]

        logger.info(f"[Dense] Encoding {self.num_docs} documents...")

        # Prepare texts: combine title + text
        texts = []
        for doc in corpus:
            title = doc.get("title", "")
            text = doc.get("text", "")
            full_text = f"{title}: {text}" if title else text
            texts.append(full_text)

        # Encode in batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        # Build FAISS index
        if self.normalize:
            # Inner product = cosine similarity khi vectors đã normalize
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        self.index.add(embeddings.astype(np.float32))
        self.is_indexed = True

        logger.info(
            f"[Dense] Index built: {self.index.ntotal} vectors, "
            f"dim={self.embedding_dim}"
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k documents bằng FAISS similarity search."""
        if not self.is_indexed:
            raise RuntimeError("Index chưa được build. Gọi build_index() trước.")

        self._load_model()

        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, min(top_k, self.num_docs))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                results.append((self.doc_ids[idx], float(score)))

        return results

    def batch_retrieve(
        self, queries: List[str], top_k: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """Batch retrieve sử dụng FAISS batch search."""
        if not self.is_indexed:
            raise RuntimeError("Index chưa được build. Gọi build_index() trước.")

        self._load_model()

        query_vecs = self.model.encode(
            queries,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        ).astype(np.float32)

        scores_batch, indices_batch = self.index.search(
            query_vecs, min(top_k, self.num_docs)
        )

        results = []
        for scores, indices in zip(scores_batch, indices_batch):
            r = []
            for score, idx in zip(scores, indices):
                if idx >= 0:
                    r.append((self.doc_ids[idx], float(score)))
            results.append(r)

        return results

    def save_index(self, path: str) -> None:
        """Lưu FAISS index + metadata ra file."""
        if not self.is_indexed:
            raise RuntimeError("Không có index để lưu.")

        faiss = self._get_faiss()

        # Save FAISS index
        faiss_path = path if path.endswith(".faiss") else path + ".faiss"
        faiss.write_index(self.index, faiss_path)

        # Save metadata
        meta_path = faiss_path.replace(".faiss", "_meta.json")
        metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "num_docs": self.num_docs,
            "doc_ids": self.doc_ids,
            "normalize": self.normalize,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"[Dense] Index saved to {faiss_path}")

    def load_index(self, path: str) -> None:
        """Tải FAISS index + metadata từ file."""
        faiss = self._get_faiss()

        faiss_path = path if path.endswith(".faiss") else path + ".faiss"
        self.index = faiss.read_index(faiss_path)

        meta_path = faiss_path.replace(".faiss", "_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.model_name = metadata["model_name"]
        self.embedding_dim = metadata["embedding_dim"]
        self.num_docs = metadata["num_docs"]
        self.doc_ids = metadata["doc_ids"]
        self.normalize = metadata["normalize"]
        self.name = f"Dense ({self.model_name.split('/')[-1]})"
        self.is_indexed = True

        logger.info(
            f"[Dense] Index loaded: {self.num_docs} docs, "
            f"dim={self.embedding_dim}"
        )
