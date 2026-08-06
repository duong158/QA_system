# -*- coding: utf-8 -*-
"""
Abstract Base Retriever — Interface chung cho tất cả retrieval methods.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class cho Document Retriever.

    Mọi retriever (TF-IDF, BM25, Dense, Pyserini) đều kế thừa class này
    và implement 3 phương thức chính: build_index, retrieve, batch_retrieve.
    """

    def __init__(self, name: str = "BaseRetriever"):
        self.name = name
        self.is_indexed = False
        self.doc_ids: List[str] = []
        self.num_docs: int = 0

    @abstractmethod
    def build_index(self, corpus: List[Dict[str, str]]) -> None:
        """Xây dựng index từ corpus.

        Args:
            corpus: List of dicts, mỗi dict chứa ít nhất {'id': str, 'text': str}.
                    Có thể chứa thêm 'title'.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Truy vấn top-k documents cho một câu hỏi.

        Args:
            query: Câu hỏi dạng text.
            top_k: Số documents trả về.

        Returns:
            List of (doc_id, score) tuples, sắp xếp giảm dần theo score.
        """
        pass

    def batch_retrieve(
        self, queries: List[str], top_k: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """Truy vấn top-k documents cho một batch câu hỏi.

        Default implementation: gọi retrieve() lần lượt.
        Subclass có thể override để tối ưu batch processing.

        Args:
            queries: List câu hỏi.
            top_k: Số documents trả về cho mỗi query.

        Returns:
            List of results, mỗi result là List[(doc_id, score)].
        """
        results = []
        for query in queries:
            results.append(self.retrieve(query, top_k=top_k))
        return results

    @abstractmethod
    def save_index(self, path: str) -> None:
        """Lưu index ra file."""
        pass

    @abstractmethod
    def load_index(self, path: str) -> None:
        """Tải index từ file."""
        pass

    def __repr__(self):
        return f"{self.name}(num_docs={self.num_docs}, indexed={self.is_indexed})"
