from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "IMG"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": "#94A3B8",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
        }
    )


def retriever_chart() -> None:
    payload = json.loads((ROOT / "results" / "retriever_eval_all.json").read_text(encoding="utf-8-sig"))
    methods = ["bm25", "tfidf", "dense", "hybrid"]
    labels = ["BM25", "TF-IDF", "Dense", "Hybrid"]
    colors = ["#2563EB", "#0EA5E9", "#8B5CF6", "#F59E0B"]
    ks = [1, 3, 5, 10]
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    for method, label, color in zip(methods, labels, colors):
        values = [payload[method]["test"][f"Recall@{k}"] * 100 for k in ks]
        ax.plot(ks, values, marker="o", linewidth=2.5, markersize=6, color=color, label=label)
        ax.annotate(f"{values[-1]:.1f}%", (ks[-1], values[-1]), xytext=(5, 0), textcoords="offset points", color=color, va="center")
    ax.set_title("Retriever benchmark on 7,301 UIT-ViQuAD2.0 test questions")
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k (%)")
    ax.set_xticks(ks)
    ax.set_ylim(50, 95)
    ax.legend(frameon=False, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(IMG / "report_retriever_recall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def reader_chart() -> None:
    payload = json.loads(
        (ROOT / "results" / "reader" / "heidiie_phobert_finetuned_viquad_full" / "validation_metrics.json").read_text(encoding="utf-8")
    )
    labels = ["Overall EM", "Overall F1", "Answerable EM", "Answerable F1", "Unanswerable acc."]
    values = [
        payload["overall"]["em"],
        payload["overall"]["f1"],
        payload["answerable"]["em"],
        payload["answerable"]["f1"],
        payload["unanswerable"]["accuracy"],
    ]
    colors = ["#2563EB", "#0EA5E9", "#F97316", "#F59E0B", "#10B981"]
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    bars = ax.barh(labels, values, color=colors)
    ax.set_title("PhoBERT Reader on the full 3,814-example validation split")
    ax.set_xlabel("Score (%)")
    ax.set_xlim(0, 50)
    ax.invert_yaxis()
    for bar, value in zip(bars, values):
        ax.text(value + 0.7, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center")
    fig.tight_layout()
    fig.savefig(IMG / "report_reader_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def semantic_chart() -> None:
    payload = json.loads((ROOT / "results" / "semantic_holdout_v1_report_refactored_v2_final.json").read_text(encoding="utf-8"))
    stages = ["reader_only", "reader_plus_refinement", "reader_refinement_subject", "reader_refinement_subject_relation", "full_pipeline"]
    labels = ["Reader only", "+ refinement", "+ subject", "+ relation", "Full pipeline"]
    f1 = [payload["ablation"][stage]["overall_f1"] for stage in stages]
    unans = [payload["ablation"][stage]["unanswerable_accuracy"] for stage in stages]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(x - width / 2, f1, width, label="Overall F1", color="#2563EB")
    ax.bar(x + width / 2, unans, width, label="Unanswerable accuracy", color="#F59E0B")
    ax.set_title("Semantic holdout ablation (provisional: lock checksum currently mismatches)")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "report_semantic_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    style()
    retriever_chart()
    reader_chart()
    semantic_chart()
