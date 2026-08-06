# -*- coding: utf-8 -*-
"""
Build Index Script — Xây dựng index cho tất cả retrieval methods.

Usage:
    python retrieval/build_index.py --method all
    python retrieval/build_index.py --method tfidf
    python retrieval/build_index.py --method bm25
    python retrieval/build_index.py --method dense
    python retrieval/build_index.py --method pyserini
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

import pandas as pd

# ── Setup paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR / "data" / "processed"
INDEX_DIR = PROJECT_DIR / "data" / "indexes"

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_corpus() -> list:
    """Load corpus từ corpus_clean.parquet."""
    corpus_path = DATA_DIR / "corpus_clean.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus không tìm thấy: {corpus_path}\n"
            "Chạy preprocessing notebook trước."
        )

    df = pd.read_parquet(corpus_path)
    corpus = []
    for _, row in df.iterrows():
        corpus.append({
            "id": str(row["id"]),
            "title": str(row.get("title", "")),
            "text": str(row["text"]),
        })

    logger.info(f"Loaded corpus: {len(corpus)} documents")
    return corpus


def build_tfidf(corpus: list):
    """Build TF-IDF index (DrQA style)."""
    from retrieval.tfidf_retriever import TfidfRetriever

    retriever = TfidfRetriever(ngram=2, hash_size=2**24)

    start = time.time()
    retriever.build_index(corpus)
    elapsed = time.time() - start

    index_path = INDEX_DIR / "tfidf_index.npz"
    retriever.save_index(str(index_path))

    logger.info(f"TF-IDF index built in {elapsed:.1f}s → {index_path}")
    return retriever


def build_bm25(corpus: list, variant: str = "okapi"):
    """Build BM25 index."""
    from retrieval.bm25_retriever import BM25Retriever

    retriever = BM25Retriever(variant=variant, k1=1.5, b=0.75)

    start = time.time()
    retriever.build_index(corpus)
    elapsed = time.time() - start

    index_path = INDEX_DIR / f"bm25_{variant}_index.pkl"
    retriever.save_index(str(index_path))

    logger.info(f"BM25 ({variant}) index built in {elapsed:.1f}s → {index_path}")
    return retriever


def build_dense(corpus: list, model_name: str = "keepitreal/vietnamese-sbert"):
    """Build Dense retrieval index (FAISS)."""
    from retrieval.dense_retriever import DenseRetriever

    retriever = DenseRetriever(model_name=model_name, batch_size=64)

    start = time.time()
    retriever.build_index(corpus)
    elapsed = time.time() - start

    index_path = INDEX_DIR / "dense_index.faiss"
    retriever.save_index(str(index_path))

    logger.info(f"Dense index built in {elapsed:.1f}s → {index_path}")
    return retriever


def build_pyserini(corpus: list):
    """Build Pyserini (Lucene) index."""
    from retrieval.pyserini_retriever import PyseriniRetriever

    index_dir = str(INDEX_DIR / "pyserini_lucene")
    retriever = PyseriniRetriever(index_dir=index_dir)

    start = time.time()
    retriever.build_index(corpus)
    elapsed = time.time() - start

    meta_path = INDEX_DIR / "pyserini_meta.json"
    retriever.save_index(str(meta_path))

    logger.info(f"Pyserini index built in {elapsed:.1f}s → {index_dir}")
    return retriever


def main():
    parser = argparse.ArgumentParser(
        description="Build retrieval indexes cho QA tiếng Việt"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "tfidf", "bm25", "dense", "pyserini",
                 "sparse"],  # sparse = tfidf + bm25
        help="Phương pháp retrieval cần build index. "
             "'sparse' = tfidf + bm25, 'all' = tất cả."
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default="keepitreal/vietnamese-sbert",
        help="Model name cho Dense retrieval."
    )
    parser.add_argument(
        "--bm25-variant",
        type=str,
        default="okapi",
        choices=["okapi", "l", "plus"],
        help="BM25 variant."
    )
    args = parser.parse_args()

    # Tạo thư mục index
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load corpus
    corpus = load_corpus()

    method = args.method.lower()

    if method in ("all", "sparse", "tfidf"):
        try:
            build_tfidf(corpus)
        except Exception as e:
            logger.error(f"TF-IDF build failed: {e}")

    if method in ("all", "sparse", "bm25"):
        try:
            build_bm25(corpus, variant=args.bm25_variant)
        except Exception as e:
            logger.error(f"BM25 build failed: {e}")

    if method in ("all", "dense"):
        try:
            build_dense(corpus, model_name=args.dense_model)
        except Exception as e:
            logger.error(f"Dense build failed: {e}")

    if method in ("all", "pyserini"):
        try:
            build_pyserini(corpus)
        except Exception as e:
            logger.warning(f"Pyserini build skipped: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
