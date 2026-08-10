# -*- coding: utf-8 -*-
"""
Benchmark Runner — Chạy đánh giá tổng hợp tất cả retrievers & xuất báo cáo.

Usage:
    python retrieval/run_benchmark.py --eval-set dev
    python retrieval/run_benchmark.py --eval-set test --methods tfidf bm25 dense
    python retrieval/run_benchmark.py --eval-set dev --top-k 1 3 5 10 20 50
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np

# ── Setup paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR / "data" / "processed"
INDEX_DIR = PROJECT_DIR / "data" / "indexes"
OUTPUT_DIR = PROJECT_DIR / "results"

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_eval_data(eval_set: str):
    """Load evaluation QA dataset và corpus.

    Returns:
        (qa_df, corpus_df)
    """
    if eval_set in ("val", "dev"):
        qa_path = DATA_DIR / "viquad_val_clean.parquet"
    elif eval_set == "test":
        qa_path = DATA_DIR / "viquad_test_clean.parquet"
    elif eval_set == "train":
        qa_path = DATA_DIR / "viquad_train_clean.parquet"
    else:
        raise ValueError(f"Unknown eval_set: {eval_set}. Use 'val', 'test', or 'train'.")

    if not qa_path.exists():
        raise FileNotFoundError(f"QA data not found: {qa_path}")

    qa_df = pd.read_parquet(qa_path)
    logger.info(f"Loaded {eval_set} QA set: {len(qa_df)} questions")

    # Corpus
    corpus_path = DATA_DIR / "corpus_clean.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    corpus_df = pd.read_parquet(corpus_path)
    logger.info(f"Loaded corpus: {len(corpus_df)} documents")

    return qa_df, corpus_df


def load_retriever(method: str, **kwargs):
    """Load retriever với pre-built index.

    Returns:
        BaseRetriever instance (hoặc None nếu không load được).
    """
    if method == "tfidf":
        from retrieval.tfidf_retriever import TfidfRetriever

        index_path = INDEX_DIR / "tfidf_index.npz"
        if not index_path.exists():
            logger.warning(f"TF-IDF index not found: {index_path}. Skipping.")
            return None

        retriever = TfidfRetriever()
        retriever.load_index(str(index_path))
        return retriever

    elif method == "bm25":
        from retrieval.bm25_retriever import BM25Retriever

        variant = kwargs.get("bm25_variant", "okapi")
        index_path = INDEX_DIR / f"bm25_{variant}_index.pkl"
        if not index_path.exists():
            logger.warning(f"BM25 index not found: {index_path}. Skipping.")
            return None

        retriever = BM25Retriever(variant=variant)
        retriever.load_index(str(index_path))
        return retriever

    elif method == "dense":
        from retrieval.dense_retriever import DenseRetriever

        index_path = INDEX_DIR / "dense_index.faiss"
        if not index_path.exists():
            logger.warning(f"Dense index not found: {index_path}. Skipping.")
            return None

        retriever = DenseRetriever()
        retriever.load_index(str(index_path))
        return retriever

    elif method == "pyserini":
        from retrieval.pyserini_retriever import PyseriniRetriever

        meta_path = INDEX_DIR / "pyserini_meta.json"
        if not meta_path.exists():
            logger.warning(f"Pyserini meta not found: {meta_path}. Skipping.")
            return None

        retriever = PyseriniRetriever()
        retriever.load_index(str(meta_path))
        return retriever

    else:
        logger.error(f"Unknown method: {method}")
        return None


def plot_results(results: List[Dict], output_dir: str, prefix: str = "benchmark"):
    """Vẽ biểu đồ so sánh Recall@k giữa các phương pháp.

    Xuất ra file PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        logger.warning("matplotlib not available. Skipping plot.")
        return

    if not results:
        return

    # ── Collect Recall@k columns ──
    recall_cols = []
    for r in results:
        for key in r:
            if key.startswith("Recall@") and key not in recall_cols:
                recall_cols.append(key)

    recall_cols = sorted(recall_cols, key=lambda x: int(x.split("@")[1]))
    k_values = [int(c.split("@")[1]) for c in recall_cols]

    # ── Color palette ──
    colors = ["#4361EE", "#F72585", "#4CC9F0", "#7209B7", "#3A0CA3", "#560BAD"]

    # ── Figure 1: Recall@k curves ──
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    for i, r in enumerate(results):
        method = r["method"]
        recalls = [r.get(c, 0) * 100 for c in recall_cols]
        color = colors[i % len(colors)]
        ax.plot(
            k_values, recalls,
            marker="o", linewidth=2.5, markersize=8,
            label=method, color=color, zorder=3,
        )
        # Annotate last point
        ax.annotate(
            f"{recalls[-1]:.1f}%",
            (k_values[-1], recalls[-1]),
            textcoords="offset points",
            xytext=(10, 0),
            fontsize=9, color=color, fontweight="bold",
        )

    ax.set_xlabel("k (Top-k documents)", fontsize=12, color="white")
    ax.set_ylabel("Recall@k (%)", fontsize=12, color="white")
    ax.set_title(
        f"Retrieval Performance — Recall@k (UIT-ViQuAD2.0 {prefix})",
        fontsize=14, color="white", fontweight="bold", pad=15,
    )
    ax.set_xticks(k_values)
    ax.tick_params(colors="white")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.legend(
        fontsize=10, loc="lower right",
        facecolor="#21262D", edgecolor="#30363D",
        labelcolor="white",
    )
    ax.grid(True, alpha=0.15, color="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#30363D")
    ax.spines["bottom"].set_color("#30363D")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{prefix}_recall_curves.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Recall@k curve saved to {plot_path}")

    # ── Figure 2: Bar chart MRR + Recall@5 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0D1117")

    methods = [r["method"] for r in results]
    x = np.arange(len(methods))

    for ax_idx, (metric, title) in enumerate([
        ("MRR", "MRR (Mean Reciprocal Rank)"),
        ("Recall@5", "Recall@5"),
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor("#161B22")

        values = [r.get(metric, 0) for r in results]
        if metric.startswith("Recall"):
            values = [v * 100 for v in values]

        bar_colors = [colors[i % len(colors)] for i in range(len(methods))]
        bars = ax.bar(x, values, color=bar_colors, width=0.6, zorder=3)

        # Value labels on bars
        for bar, val in zip(bars, values):
            fmt = f"{val:.1f}%" if metric.startswith("Recall") else f"{val:.4f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                fmt, ha="center", va="bottom",
                fontsize=9, color="white", fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace(" ", "\n") for m in methods],
            fontsize=9, color="white",
        )
        ax.set_title(title, fontsize=12, color="white", fontweight="bold")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.1, axis="y", color="white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#30363D")
        ax.spines["bottom"].set_color("#30363D")

    plt.tight_layout()
    bar_path = os.path.join(output_dir, f"{prefix}_metrics_bar.png")
    fig.savefig(bar_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Metrics bar chart saved to {bar_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval methods cho QA tiếng Việt"
    )
    parser.add_argument(
        "--eval-set",
        type=str,
        default="val",
        choices=["val", "dev", "test", "train"],
        help="Tập dữ liệu đánh giá."
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["tfidf", "bm25", "dense", "pyserini"],
        help="Các phương pháp retrieval cần đánh giá."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20, 50],
        help="Các giá trị k cho Recall@k."
    )
    parser.add_argument(
        "--bm25-variant",
        type=str,
        default="okapi",
        choices=["okapi", "l", "plus"],
        help="BM25 variant."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size cho retrieval."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Thư mục output. Default: results/"
    )
    args = parser.parse_args()

    output_dir = args.output_dir or str(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──
    qa_df, corpus_df = load_eval_data(args.eval_set)

    # ── Build gold mapping ──
    from retrieval.evaluate import (
        build_gold_mapping,
        evaluate_retriever,
        format_results_table,
        save_results,
    )

    gold_map = build_gold_mapping(qa_df, corpus_df)

    # ── Evaluate each method ──
    all_results = []

    for method in args.methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {method.upper()}")
        logger.info(f"{'='*60}")

        try:
            retriever = load_retriever(
                method, bm25_variant=args.bm25_variant
            )
            if retriever is None:
                logger.warning(f"Skipping {method}: index not found.")
                continue

            result = evaluate_retriever(
                retriever,
                qa_df,
                gold_map,
                k_values=args.top_k,
                batch_size=args.batch_size,
            )
            result["eval_set"] = args.eval_set
            all_results.append(result)

        except Exception as e:
            logger.error(f"Error evaluating {method}: {e}")
            import traceback
            traceback.print_exc()

    # ── Output results ──
    if all_results:
        # Console table
        table = format_results_table(all_results)
        print("\n" + table + "\n")

        # Save files
        prefix = f"benchmark_{args.eval_set}"
        save_results(all_results, output_dir, prefix=prefix)

        # Plot
        plot_results(all_results, output_dir, prefix=prefix)

        logger.info(f"\nAll results saved to: {output_dir}/")
    else:
        logger.warning("No results to report. Build indexes first:")
        logger.warning("  python retrieval/build_index.py --method all")


if __name__ == "__main__":
    main()
