# -*- coding: utf-8 -*-
"""
Evaluation Framework — Đánh giá Recall@k và MRR cho Document Retrieval.

Metrics:
  - Recall@k: Tỷ lệ câu hỏi mà gold document nằm trong top-k results.
  - MRR (Mean Reciprocal Rank): Trung bình 1/rank của gold document.
  - Hit@k: Tương đương Recall@k (binary per-query).

Logic matching gold documents:
  Mỗi câu hỏi QA trong UIT-ViQuAD có trường 'context'. Chúng ta cần map
  context → doc_id trong corpus. Matching bằng exact string hoặc overlap.
"""

import logging
import time
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

from .base import BaseRetriever

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Gold Document Mapping
# ──────────────────────────────────────────────────────────────────────────────


def build_gold_mapping(
    qa_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    match_threshold: float = 0.8,
) -> Dict[str, List[str]]:
    """Tạo mapping: question_id → list of gold doc_ids.

    Matching logic:
    1. Exact match: qa.context == corpus.text
    2. Substring match: qa.context in corpus.text hoặc ngược lại
    3. Overlap ratio: Jaccard similarity > threshold

    Args:
        qa_df: DataFrame với columns ['id', 'context', 'question', ...]
        corpus_df: DataFrame với columns ['id', 'text', 'title']
        match_threshold: Ngưỡng overlap ratio cho fuzzy matching.

    Returns:
        Dict mapping question_id → list of matching doc_ids.
    """
    logger.info("Building gold document mapping...")

    # Index corpus texts cho tra cứu nhanh
    corpus_text_to_id: Dict[str, str] = {}
    corpus_records: List[Tuple[str, str]] = []

    for _, row in corpus_df.iterrows():
        doc_id = str(row["id"])
        text = str(row["text"]).strip()
        corpus_text_to_id[text] = doc_id
        corpus_records.append((doc_id, text))

    gold_map: Dict[str, List[str]] = {}
    matched = 0
    unmatched = 0

    for _, qa_row in qa_df.iterrows():
        q_id = str(qa_row["id"])
        context = str(qa_row["context"]).strip()

        # ── Strategy 1: Exact match ──
        if context in corpus_text_to_id:
            gold_map[q_id] = [corpus_text_to_id[context]]
            matched += 1
            continue

        # ── Strategy 2: Substring match (context chứa trong doc hoặc ngược lại) ──
        found = False
        for doc_id, doc_text in corpus_records:
            if context in doc_text or doc_text in context:
                gold_map[q_id] = [doc_id]
                matched += 1
                found = True
                break

        if found:
            continue

        # ── Strategy 3: Jaccard word overlap ──
        context_words = set(context.lower().split())
        best_doc_id = None
        best_overlap = 0.0

        for doc_id, doc_text in corpus_records:
            doc_words = set(doc_text.lower().split())
            if not context_words or not doc_words:
                continue
            intersection = len(context_words & doc_words)
            union = len(context_words | doc_words)
            overlap = intersection / union if union > 0 else 0

            if overlap > best_overlap:
                best_overlap = overlap
                best_doc_id = doc_id

        if best_overlap >= match_threshold and best_doc_id is not None:
            gold_map[q_id] = [best_doc_id]
            matched += 1
        else:
            gold_map[q_id] = []
            unmatched += 1

    logger.info(
        f"Gold mapping: {matched} matched, {unmatched} unmatched "
        f"(out of {len(qa_df)} questions)"
    )

    return gold_map


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ──────────────────────────────────────────────────────────────────────────────


def compute_recall_at_k(
    retrieved: List[List[Tuple[str, float]]],
    gold_map: Dict[str, List[str]],
    question_ids: List[str],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Tính Recall@k cho các giá trị k.

    Recall@k = Tỷ lệ câu hỏi mà ít nhất 1 gold doc nằm trong top-k.

    Args:
        retrieved: List of retrieval results per query. Mỗi phần tử là
                   List[(doc_id, score)].
        gold_map: Mapping question_id → list of gold doc_ids.
        question_ids: List question IDs tương ứng với retrieved.
        k_values: Các giá trị k cần tính. Default: [1, 3, 5, 10, 20, 50].

    Returns:
        Dict['Recall@k'] = value (0-1).
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 50]

    results = {}
    num_valid = 0  # Số câu hỏi có gold document

    for k in k_values:
        hits = 0
        valid = 0

        for q_idx, q_id in enumerate(question_ids):
            gold_docs = gold_map.get(q_id, [])
            if not gold_docs:
                continue  # Bỏ qua câu hỏi không có gold mapping

            valid += 1
            top_k_docs = [doc_id for doc_id, _ in retrieved[q_idx][:k]]

            if any(g in top_k_docs for g in gold_docs):
                hits += 1

        recall = hits / valid if valid > 0 else 0.0
        results[f"Recall@{k}"] = recall
        num_valid = valid

    results["num_evaluated"] = num_valid
    return results


def compute_mrr(
    retrieved: List[List[Tuple[str, float]]],
    gold_map: Dict[str, List[str]],
    question_ids: List[str],
    max_k: int = 50,
) -> float:
    """Tính Mean Reciprocal Rank (MRR).

    MRR = (1/|Q|) * Σ (1/rank_i)
    Trong đó rank_i là vị trí đầu tiên của gold doc trong top-k results.

    Args:
        retrieved: List of retrieval results per query.
        gold_map: Mapping question_id → list of gold doc_ids.
        question_ids: List question IDs.
        max_k: Chỉ xét trong top max_k results.

    Returns:
        MRR score (0-1).
    """
    reciprocal_ranks = []

    for q_idx, q_id in enumerate(question_ids):
        gold_docs = gold_map.get(q_id, [])
        if not gold_docs:
            continue

        rr = 0.0
        top_docs = [doc_id for doc_id, _ in retrieved[q_idx][:max_k]]

        for rank, doc_id in enumerate(top_docs, start=1):
            if doc_id in gold_docs:
                rr = 1.0 / rank
                break

        reciprocal_ranks.append(rr)

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Full Evaluation Pipeline
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_retriever(
    retriever: BaseRetriever,
    qa_df: pd.DataFrame,
    gold_map: Dict[str, List[str]],
    k_values: List[int] = None,
    batch_size: int = 100,
) -> Dict:
    """Đánh giá toàn diện một retriever trên tập QA.

    Args:
        retriever: Instance của BaseRetriever (đã build index).
        qa_df: DataFrame câu hỏi (columns: 'id', 'question').
        gold_map: Gold document mapping.
        k_values: Các giá trị k cho Recall@k.
        batch_size: Batch size cho batch_retrieve.

    Returns:
        Dict chứa tất cả metrics + timing info.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 50]

    max_k = max(k_values)
    questions = qa_df["question"].tolist()
    question_ids = qa_df["id"].astype(str).tolist()

    logger.info(
        f"[Eval] Evaluating {retriever.name} on {len(questions)} questions..."
    )

    # ── Retrieve ──
    start_time = time.time()

    all_results = []
    for i in range(0, len(questions), batch_size):
        batch_q = questions[i : i + batch_size]
        batch_results = retriever.batch_retrieve(batch_q, top_k=max_k)
        all_results.extend(batch_results)

        if (i + batch_size) % 500 == 0 or i + batch_size >= len(questions):
            logger.info(
                f"  Retrieved {min(i + batch_size, len(questions))}/{len(questions)}"
            )

    retrieval_time = time.time() - start_time

    # ── Compute metrics ──
    recall_scores = compute_recall_at_k(
        all_results, gold_map, question_ids, k_values
    )
    mrr = compute_mrr(all_results, gold_map, question_ids, max_k=max_k)

    result = {
        "method": retriever.name,
        "num_questions": len(questions),
        "num_evaluated": recall_scores.pop("num_evaluated", len(questions)),
        "retrieval_time_sec": round(retrieval_time, 2),
        "queries_per_sec": round(len(questions) / retrieval_time, 1) if retrieval_time > 0 else 0,
        "MRR": round(mrr, 4),
    }
    for k_name, score in recall_scores.items():
        result[k_name] = round(score, 4)

    logger.info(f"[Eval] {retriever.name}: MRR={mrr:.4f}, {recall_scores}")
    return result


def format_results_table(results: List[Dict]) -> str:
    """Format kết quả evaluation thành bảng text đẹp.

    Args:
        results: List of result dicts từ evaluate_retriever().

    Returns:
        Formatted table string.
    """
    if not results:
        return "No results."

    # Collect all metric columns
    metric_cols = ["MRR"]
    for r in results:
        for key in r:
            if key.startswith("Recall@") and key not in metric_cols:
                metric_cols.append(key)

    # Sort Recall@k columns
    recall_cols = sorted(
        [c for c in metric_cols if c.startswith("Recall@")],
        key=lambda x: int(x.split("@")[1]),
    )
    metric_cols = ["MRR"] + recall_cols

    # Build table
    header = f"{'Method':<25} " + " ".join(f"{c:>10}" for c in metric_cols) + f" {'Time(s)':>8} {'Q/s':>8}"
    separator = "-" * len(header)

    lines = [separator, header, separator]
    for r in results:
        method = r.get("method", "?")
        vals = []
        for c in metric_cols:
            v = r.get(c, 0.0)
            vals.append(f"{v*100:>9.1f}%" if c.startswith("Recall") else f"{v:>10.4f}")

        line = (
            f"{method:<25} "
            + " ".join(vals)
            + f" {r.get('retrieval_time_sec', 0):>8.1f}"
            + f" {r.get('queries_per_sec', 0):>8.1f}"
        )
        lines.append(line)

    lines.append(separator)
    return "\n".join(lines)


def save_results(results: List[Dict], output_dir: str, prefix: str = "benchmark"):
    """Lưu kết quả ra JSON và CSV.

    Args:
        results: List of result dicts.
        output_dir: Thư mục output.
        prefix: Prefix cho tên file.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, f"{prefix}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV
    csv_path = os.path.join(output_dir, f"{prefix}_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    logger.info(f"Results saved to {json_path} and {csv_path}")
    return json_path, csv_path
