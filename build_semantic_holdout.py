from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import polars as pl


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data" / "processed" / "viquad_val_clean.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "evaluation"
DEFAULT_QUOTAS = {
    "CAUSE": 40,
    "TIME": 40,
    "LOCATION": 30,
    "PURPOSE": 20,
    "ENTITY": 30,
    "DEFINITION": 20,
    "GENERAL": 40,
    "NO_ANSWER": 40,
}
SEMANTIC_TUNING_RESULT_PATTERNS = (
    "cause_semantic_replay*.json",
    "fallback_phrase_*.json",
    "location_diagnostic_*.json",
    "location_pipeline_*.json",
    "manual_live_semantic_*.json",
    "reranking_candidate_cache*.json",
    "reranking_refinement_validation*.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_text(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text or "").casefold().replace("đ", "d"))
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return " ".join(re.findall(r"[\w]+", value, flags=re.UNICODE))


def weak_relation_bucket(question: str) -> str:
    """Frozen, selection-only weak labels; never used by the production API."""

    text = fold_text(question)
    if re.search(r"\b(?:vi sao|tai sao|do dau|boi dau|nguyen nhan|ly do)\b", text):
        return "CAUSE"
    if re.search(r"\b(?:dieu gi|yeu to nao)\b.+\b(?:khien|lam|dan toi|dan den)\b", text):
        return "CAUSE"
    if re.search(r"\b(?:muc dich|de lam gi|nham|nham vao|nham muc tieu)\b", text):
        return "PURPOSE"
    if re.search(
        r"\b(?:khi nao|bao gio|nam nao|ngay nao|thang nao|thoi gian nao|"
        r"nam bao nhieu|nam sinh|sinh nam|sinh vao nam|the ky nao)\b",
        text,
    ):
        return "TIME"
    if re.search(
        r"\b(?:o dau|tai dau|noi nao|dia diem nao|dia danh nao|khu vuc nao|"
        r"thanh pho nao|tinh nao|quoc gia nao|nuoc nao)\b",
        text,
    ):
        return "LOCATION"
    if re.search(r"\b(?:la gi|co nghia la gi|dinh nghia|duoc hieu nhu the nao)\b", text):
        return "DEFINITION"
    if re.search(
        r"\b(?:ai|nguoi nao|nhan vat nao|cai gi|dieu gi|cong trinh nao|"
        r"tac pham nao|to chuc nao|ten goi nao|bi danh nao|ten nao|loai nao)\b",
        text,
    ):
        return "ENTITY"
    return "GENERAL"


def iter_json_values(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from iter_json_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_json_values(item)


def prior_evaluation_ids() -> tuple[set[str], list[str]]:
    excluded: set[str] = set()
    sources: list[str] = []
    results_dir = ROOT / "results"
    paths = sorted(
        {
            path
            for pattern in SEMANTIC_TUNING_RESULT_PATTERNS
            for path in results_dir.glob(pattern)
        }
    )
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        found_before = len(excluded)
        for item in iter_json_values(payload):
            identifier = item.get("id")
            if identifier is not None:
                excluded.add(str(identifier))
        if len(excluded) > found_before:
            sources.append(str(path.relative_to(ROOT)))
    return excluded, sources


def full_validation_evaluation_ids() -> tuple[int, list[str]]:
    """Report broad prior evaluation without treating every row as semantic tuning."""

    paths = [ROOT / "results" / "reader_eval_results.json"]
    identifiers: set[str] = set()
    sources: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        for item in iter_json_values(payload):
            identifier = item.get("id")
            if identifier is not None:
                identifiers.add(str(identifier))
        sources.append(str(path.relative_to(ROOT)))
    return len(identifiers), sources


def regression_questions() -> tuple[set[str], list[str]]:
    excluded: set[str] = set()
    sources: list[str] = []
    for name in ("qa_semantic_regressions.json", "manual_live_semantic_set_v1.json"):
        path = ROOT / "tests" / "data" / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("cases", []) if isinstance(payload, dict) else payload
        for row in rows:
            question = str(row.get("question") or "").strip()
            if question:
                excluded.add(fold_text(question))
        sources.append(str(path.relative_to(ROOT)))
    return excluded, sources


def stable_key(seed: str, row: dict[str, Any]) -> str:
    identity = f"{seed}\0{row.get('id')}\0{row.get('question')}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def select_holdout(rows: list[dict[str, Any]], seed: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_ids, result_sources = prior_evaluation_ids()
    broadly_evaluated_count, broad_evaluation_sources = full_validation_evaluation_ids()
    excluded_questions, regression_sources = regression_questions()
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_counts: Counter[str] = Counter()

    for row in rows:
        identifier = str(row.get("id") or "")
        question = str(row.get("question") or "").strip()
        if identifier in excluded_ids:
            excluded_counts["prior_evaluation_id"] += 1
            continue
        if fold_text(question) in excluded_questions:
            excluded_counts["regression_question"] += 1
            continue
        answer_start_raw = row.get("answer_start")
        answer_start = int(answer_start_raw) if answer_start_raw is not None else -1
        answer = str(row.get("answer_text") or "")
        answerable = answer_start >= 0 and bool(answer.strip())
        bucket = weak_relation_bucket(question)
        prepared = {
            "id": identifier,
            "title": str(row.get("title") or ""),
            "context": str(row.get("context") or ""),
            "question": question,
            "gold_answer": answer if answerable else "",
            "answer_start": answer_start if answerable else -1,
            "is_answerable": answerable,
            "expected_relation_bucket": bucket,
        }
        (buckets if answerable else no_answer)[bucket].append(prepared)

    for grouped in (buckets, no_answer):
        for values in grouped.values():
            values.sort(key=lambda row: stable_key(seed, row))

    selected: list[dict[str, Any]] = []
    actual: Counter[str] = Counter()
    for bucket, quota in DEFAULT_QUOTAS.items():
        if bucket == "NO_ANSWER":
            continue
        available = buckets[bucket]
        if len(available) < quota:
            raise RuntimeError(f"Not enough unseen {bucket} rows: {len(available)} < {quota}")
        for row in available[:quota]:
            copied = dict(row)
            copied["holdout_stratum"] = bucket
            selected.append(copied)
            actual[bucket] += 1

    no_answer_quota = DEFAULT_QUOTAS["NO_ANSWER"]
    no_answer_keys = sorted(no_answer)
    cursor = Counter()
    while actual["NO_ANSWER"] < no_answer_quota:
        progressed = False
        for bucket in no_answer_keys:
            index = cursor[bucket]
            if index >= len(no_answer[bucket]):
                continue
            row = dict(no_answer[bucket][index])
            cursor[bucket] += 1
            row["holdout_stratum"] = "NO_ANSWER"
            selected.append(row)
            actual["NO_ANSWER"] += 1
            progressed = True
            if actual["NO_ANSWER"] >= no_answer_quota:
                break
        if not progressed:
            raise RuntimeError("Not enough unseen no-answer rows")

    selected.sort(key=lambda row: stable_key(seed + ":final", row))
    metadata = {
        "actual_quotas": dict(sorted(actual.items())),
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "excluded_prior_id_count": len(excluded_ids),
        "exclusion_sources": result_sources + regression_sources,
        "prior_broad_evaluation_id_count": broadly_evaluated_count,
        "prior_broad_evaluation_sources": broad_evaluation_sources,
        "blind_scope": (
            "Blind to explicit semantic regression questions and semantic/reranking candidate pools, "
            "but not model-blind: the validation split was previously used for Reader evaluation."
        ),
        "no_answer_relation_distribution": dict(
            sorted(Counter(row["expected_relation_bucket"] for row in selected if not row["is_answerable"]).items())
        ),
    }
    return selected, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an immutable semantic holdout from ViQuAD validation")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--seed", default="semantic-holdout-v1-2026-08-18")
    args = parser.parse_args()

    dataset_path = args.output_dir / f"semantic_holdout_{args.version}.jsonl"
    lock_path = args.output_dir / f"semantic_holdout_{args.version}.lock.json"
    if dataset_path.exists() or lock_path.exists():
        if not (dataset_path.is_file() and lock_path.is_file()):
            raise RuntimeError("Holdout is partially present; do not overwrite it")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        actual_hash = sha256_file(dataset_path)
        if actual_hash != lock.get("dataset_sha256"):
            raise RuntimeError("Locked holdout checksum mismatch")
        print(json.dumps({"status": "already_locked", "rows": lock["rows"], "sha256": actual_hash}, indent=2))
        return

    rows = pl.read_parquet(args.source).to_dicts()
    selected, selection = select_holdout(rows, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    dataset_hash = sha256_file(dataset_path)
    lock = {
        "name": f"semantic_holdout_{args.version}",
        "status": "LOCKED_DO_NOT_TUNE",
        "created_date": "2026-08-18",
        "source": str(args.source.relative_to(ROOT)),
        "source_sha256": sha256_file(args.source),
        "dataset_file": str(dataset_path.relative_to(ROOT)),
        "dataset_sha256": dataset_hash,
        "rows": len(selected),
        "seed": args.seed,
        "requested_quotas": DEFAULT_QUOTAS,
        **selection,
        "policy": [
            "Never add a failing holdout item to production rules and rerun under the v1 name.",
            "Any production change after inspecting v1 requires a newly sampled v2 holdout.",
            "The weak relation labels are for stratification, not human gold semantic annotations.",
            "v1 is semantic-rule holdout only; it is not a model-blind test split.",
        ],
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "created_and_locked", "rows": len(selected), "sha256": dataset_hash, **selection}, indent=2))


if __name__ == "__main__":
    main()
