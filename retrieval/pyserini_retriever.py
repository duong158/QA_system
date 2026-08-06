# -*- coding: utf-8 -*-
"""
Pyserini Document Retriever — Lucene-based BM25 via Anserini/Pyserini.

Pyserini cung cấp BM25 retrieval tối ưu dựa trên Lucene inverted index,
hiệu suất cao hơn rank_bm25 trên corpus lớn.

Yêu cầu:
  - Java 11+ JDK
  - pip install pyserini

Reference: https://github.com/castorini/pyserini
"""

import os
import json
import shutil
import logging
import tempfile
from typing import List, Dict, Tuple, Optional

from .base import BaseRetriever

logger = logging.getLogger(__name__)


def _check_java() -> bool:
    """Kiểm tra Java có sẵn trên hệ thống không."""
    import subprocess
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class PyseriniRetriever(BaseRetriever):
    """Pyserini Document Retriever (Lucene BM25).

    Args:
        index_dir: Thư mục lưu Lucene index. Nếu None, tạo tạm.
        k1: BM25 k1 parameter. Default=0.9.
        b: BM25 b parameter. Default=0.4.
        language: Ngôn ngữ cho Lucene analyzer. Default='vi' (Vietnamese).
    """

    def __init__(
        self,
        index_dir: Optional[str] = None,
        k1: float = 0.9,
        b: float = 0.4,
        language: str = "vi",
    ):
        super().__init__(name="Pyserini (Lucene BM25)")
        self.index_dir = index_dir
        self.k1 = k1
        self.b = b
        self.language = language

        self.searcher = None
        self._temp_dir = None
        self._jsonl_dir = None

    def _check_available(self):
        """Kiểm tra pyserini và Java có sẵn không."""
        if not _check_java():
            raise RuntimeError(
                "Java không tìm thấy. Pyserini yêu cầu Java 11+ JDK. "
                "Cài Java tại: https://adoptium.net/"
            )
        try:
            import pyserini
        except ImportError:
            raise ImportError(
                "pyserini chưa được cài. Chạy: pip install pyserini"
            )

    def build_index(self, corpus: List[Dict[str, str]]) -> None:
        """Xây dựng Lucene index từ corpus.

        Workflow:
        1. Xuất corpus ra JSONL (format Pyserini)
        2. Gọi Pyserini indexer để build Lucene index
        3. Khởi tạo searcher

        Args:
            corpus: List[{'id': str, 'text': str, 'title': str (optional)}]
        """
        self._check_available()
        from pyserini.index.lucene import LuceneIndexer

        self.num_docs = len(corpus)
        self.doc_ids = [doc["id"] for doc in corpus]

        logger.info(f"[Pyserini] Building Lucene index for {self.num_docs} docs...")

        # Tạo thư mục JSONL tạm
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="pyserini_")

        jsonl_dir = os.path.join(self._temp_dir, "jsonl")
        os.makedirs(jsonl_dir, exist_ok=True)

        # Xuất corpus ra JSONL
        jsonl_path = os.path.join(jsonl_dir, "corpus.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for doc in corpus:
                entry = {
                    "id": doc["id"],
                    "contents": doc.get("text", ""),
                }
                if "title" in doc and doc["title"]:
                    entry["contents"] = f"{doc['title']}. {doc['text']}"
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"[Pyserini] Exported {self.num_docs} docs to {jsonl_path}")

        # Build Lucene index
        if self.index_dir is None:
            self.index_dir = os.path.join(self._temp_dir, "lucene_index")

        os.makedirs(self.index_dir, exist_ok=True)

        # Use pyserini command-line indexer
        import subprocess
        import sys

        cmd = [
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", jsonl_dir,
            "--index", self.index_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions", "--storeDocvectors", "--storeRaw",
            "--language", self.language,
        ]

        logger.info(f"[Pyserini] Running indexer...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Pyserini indexing failed: {result.stderr}")
            raise RuntimeError(f"Pyserini indexing failed: {result.stderr}")

        # Initialize searcher
        self._init_searcher()
        self.is_indexed = True
        logger.info(f"[Pyserini] Index built at {self.index_dir}")

    def _init_searcher(self):
        """Khởi tạo Pyserini searcher."""
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_dir)
        self.searcher.set_bm25(self.k1, self.b)
        if self.language != "en":
            self.searcher.set_language(self.language)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k documents bằng Lucene BM25."""
        if not self.is_indexed or self.searcher is None:
            raise RuntimeError("Index chưa được build. Gọi build_index() trước.")

        hits = self.searcher.search(query, k=top_k)

        results = []
        for hit in hits:
            results.append((hit.docid, float(hit.score)))

        return results

    def batch_retrieve(
        self, queries: List[str], top_k: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """Batch retrieve sử dụng Pyserini batch search."""
        if not self.is_indexed or self.searcher is None:
            raise RuntimeError("Index chưa được build. Gọi build_index() trước.")

        try:
            from pyserini.search.lucene import LuceneSearcher
            # Pyserini supports batch search
            q_ids = [str(i) for i in range(len(queries))]
            hits_dict = self.searcher.batch_search(
                queries, q_ids, k=top_k, threads=1
            )

            results = []
            for q_id in q_ids:
                r = []
                for hit in hits_dict.get(q_id, []):
                    r.append((hit.docid, float(hit.score)))
                results.append(r)
            return results
        except Exception:
            # Fallback to sequential
            return super().batch_retrieve(queries, top_k)

    def save_index(self, path: str) -> None:
        """Lưu metadata (index_dir đã chứa Lucene index)."""
        if not self.is_indexed:
            raise RuntimeError("Không có index để lưu.")

        metadata = {
            "index_dir": self.index_dir,
            "k1": self.k1,
            "b": self.b,
            "language": self.language,
            "num_docs": self.num_docs,
            "doc_ids": self.doc_ids,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"[Pyserini] Metadata saved to {path}")

    def load_index(self, path: str) -> None:
        """Tải Pyserini index từ metadata file."""
        self._check_available()

        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.index_dir = metadata["index_dir"]
        self.k1 = metadata["k1"]
        self.b = metadata["b"]
        self.language = metadata["language"]
        self.num_docs = metadata["num_docs"]
        self.doc_ids = metadata["doc_ids"]

        self._init_searcher()
        self.is_indexed = True
        logger.info(f"[Pyserini] Index loaded from {self.index_dir}")

    def __del__(self):
        """Cleanup temp directory."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
