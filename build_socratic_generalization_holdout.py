from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from backend.chunking import split_sentences


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "evaluation" / "semantic_holdout_v1.jsonl"
OUTPUT = ROOT / "tests" / "data" / "socratic_generalization_holdout_v1.json"
LOCK = ROOT / "tests" / "data" / "socratic_generalization_holdout_v1.lock.json"
SEED = "socratic-generalization-holdout-v1-2026-08-21"

QUOTAS = {
    "IDENTITY_DEFINITION": 10,
    "TIME": 10,
    "LOCATION": 10,
    "CAUSE": 10,
    "PURPOSE": 10,
    "ENTITY": 10,
    "ATTRIBUTE_GENERAL": 10,
    "NO_ANSWER_SPARSE": 10,
}

# These are exclusion-only evaluation literals. Production must never import this file.
EXCLUDED_BENCHMARK_LITERALS = (
    "Phạm Văn Đồng",
    "Voltaire",
    "Roosevelt",
    "Madame du Barry",
    "Paris",
    "Saint-Pierre",
    "Baibars",
    "Cách mạng Pháp",
    "chủ nghĩa nô lệ",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fold(text: Any) -> str:
    import unicodedata

    value = unicodedata.normalize("NFD", str(text or "").casefold().replace("đ", "d"))
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def _bucket(row: dict[str, Any]) -> str:
    stratum = str(row.get("holdout_stratum") or "GENERAL").upper()
    if stratum == "DEFINITION":
        return "IDENTITY_DEFINITION"
    if stratum == "GENERAL":
        return "ATTRIBUTE_GENERAL"
    if stratum == "NO_ANSWER":
        return "NO_ANSWER_SPARSE"
    return stratum


def _contains_excluded_literal(row: dict[str, Any]) -> bool:
    haystack = "\n".join(
        str(row.get(key) or "") for key in ("title", "question", "context", "gold_answer")
    ).casefold()
    return any(literal.casefold() in haystack for literal in EXCLUDED_BENCHMARK_LITERALS)


def _weak_available_relations(context: str) -> list[str]:
    """Independent annotation proxy; deliberately does not import Socratic production rules."""

    folded = _fold(context)
    relations: set[str] = set()
    if re.search(r"\b(?:1\d{3}|20\d{2}|ngay|thang|nam|the ky|thap nien)\b", folded):
        relations.add("TIME")
    if re.search(r"\b(?:tai|o|nam tai|toa lac|thuoc|den tu|noi)\b\s+\w", folded):
        relations.add("LOCATION")
    if re.search(r"\b(?:boi vi|vi|do|boi|nguyen nhan|khiến|dan den|bat nguon tu)\b", folded):
        relations.add("CAUSE_OR_CONSEQUENCE")
    if re.search(r"\b(?:nham|voi muc dich|voi muc tieu|de)\b\s+\w", folded):
        relations.add("PURPOSE")
    if re.search(
        r"\b(?:giu chuc|dam nhiem|duoc bau|duoc bo nhiem|lanh dao|chu tich|"
        r"thu tuong|tong thong|giao su|giam doc|bi thu|bo truong)\b",
        folded,
    ):
        relations.add("ROLE")
    if re.search(r"\b(?:tham gia|thanh lap|sang lap|to chuc|ky ket|chi huy|thuc hien)\b", folded):
        relations.add("EVENT")
    if re.search(r"\b(?:co|dat|cao|dai|rong|chiem)\b.{0,35}\b\d+(?:[,.]\d+)?\b", folded):
        relations.add("ATTRIBUTE")
    if re.search(r"(?:^|[.!?]\s+)\w[^.!?]{0,90}\b(?:la|duoc xem la)\b", folded):
        relations.add("IDENTITY_DEFINITION")
    return sorted(relations)


def _current_family(stratum: str) -> str:
    return {
        "DEFINITION": "IDENTITY_DEFINITION",
        "TIME": "TIME",
        "LOCATION": "LOCATION",
        "CAUSE": "CAUSE_OR_CONSEQUENCE",
        "PURPOSE": "PURPOSE",
        "ENTITY": "ENTITY",
        "GENERAL": "ATTRIBUTE_GENERAL",
        "NO_ANSWER": "NO_ANSWER_SPARSE",
    }.get(stratum, stratum)


def _chunks(values: list[str], size: int = 3) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _passages(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str, list[str]]:
    context = str(row["context"]).strip()
    sentences = [sentence.strip() for sentence in split_sentences(context) if sentence.strip()]
    if not sentences:
        sentences = [context]
    answer = str(row.get("gold_answer") or "").strip()
    answer_folded = _fold(answer)
    selected_index = 0
    if answer_folded:
        selected_index = next(
            (index for index, sentence in enumerate(sentences) if answer_folded in _fold(sentence)),
            0,
        )
    case_id = str(row["id"]).replace("-", "_")
    selected_id = f"holdout_{case_id}_P0001"
    passages = [
        {
            "passage_id": selected_id,
            "title": row.get("title"),
            "text": sentences[selected_index],
            "relevance_score": 1.0,
            "provenance": "validation_corpus_answer_sentence",
        }
    ]
    remaining = sentences[:selected_index] + sentences[selected_index + 1 :]
    retrieved_ids: list[str] = []
    for offset, group in enumerate(_chunks(remaining), start=2):
        passage_id = f"holdout_{case_id}_P{offset:04d}"
        retrieved_ids.append(passage_id)
        passages.append(
            {
                "passage_id": passage_id,
                "title": row.get("title"),
                "text": " ".join(group),
                "relevance_score": max(0.25, 0.90 - (offset - 2) * 0.06),
                "provenance": "validation_corpus_related_sentences",
            }
        )
    return passages, selected_id, retrieved_ids


def _select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = _bucket(row)
        if bucket in QUOTAS and not _contains_excluded_literal(row):
            grouped[bucket].append(row)

    selected: list[dict[str, Any]] = []
    for bucket, quota in QUOTAS.items():
        candidates = list(grouped[bucket])
        rng.shuffle(candidates)
        # Prefer topic diversity before taking another row from the same article.
        candidates.sort(key=lambda row: str(row.get("title") or ""))
        by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            by_title[str(row.get("title") or "")].append(row)
        chosen: list[dict[str, Any]] = []
        while len(chosen) < quota and any(by_title.values()):
            for title in sorted(by_title):
                if by_title[title] and len(chosen) < quota:
                    chosen.append(by_title[title].pop())
        if len(chosen) != quota:
            raise RuntimeError(f"Could only select {len(chosen)}/{quota} rows for {bucket}")
        selected.extend(chosen)
    return selected


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise SystemExit(
            "Holdout v1 already exists. Do not overwrite a locked holdout; create a new version."
        )
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line]
    selected = _select(rows)
    cases: list[dict[str, Any]] = []
    for row in selected:
        passages, selected_id, retrieved_ids = _passages(row)
        stratum = str(row.get("holdout_stratum") or "GENERAL").upper()
        weak_relations = _weak_available_relations(str(row.get("context") or ""))
        current_family = _current_family(stratum)
        novel_relations = [relation for relation in weak_relations if relation != current_family]
        answerable = bool(row.get("is_answerable")) and bool(str(row.get("gold_answer") or "").strip())
        cases.append(
            {
                "id": f"SGH1_{row['id']}",
                "source_row_id": row["id"],
                "stratum": _bucket(row),
                "question": row["question"],
                "answer": row.get("gold_answer") if answerable else None,
                "title": row.get("title"),
                "selected_passage_id": selected_id,
                "retrieved_passage_ids": retrieved_ids,
                "passages": passages,
                "is_main_answer_available": answerable,
                "weak_available_followup_relations": novel_relations if answerable else [],
                "opportunity_available": bool(novel_relations) if answerable else False,
                "annotation_method": "weak_linguistic_v1_not_human_gold",
            }
        )

    payload = {
        "name": "socratic_generalization_holdout_v1",
        "status": "LOCKED_DO_NOT_TUNE",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_status": "LOCKED_DO_NOT_TUNE",
        "seed": SEED,
        "rows": len(cases),
        "policy": [
            "Production code must not import this file or its companion lock.",
            "Do not patch production from individual failures in this holdout.",
            "If failures are used for development, mark v1 CONSUMED_FOR_DEVELOPMENT and create v2.",
            "Opportunity labels are weak diagnostic annotations, not human gold.",
        ],
        "excluded_benchmark_literals": list(EXCLUDED_BENCHMARK_LITERALS),
        "quotas": QUOTAS,
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lock = {
        "name": payload["name"],
        "status": payload["status"],
        "source_sha256": _sha256(SOURCE),
        "dataset_file": str(OUTPUT.relative_to(ROOT)),
        "dataset_sha256": _sha256(OUTPUT),
        "rows": len(cases),
        "actual_strata": dict(sorted(Counter(case["stratum"] for case in cases).items())),
        "unique_titles": len({case["title"] for case in cases}),
        "seed": SEED,
        "created_from_git_commit": "3b5bddf",
        "holdout_was_inspected_for_production_tuning": False,
    }
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
