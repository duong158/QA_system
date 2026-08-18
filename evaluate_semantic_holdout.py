from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from reader.metrics import evaluate_predictions
from reader.question_semantics import QuestionSemantics, parse_question_semantics
from reader.semantic_policy import SEMANTIC_POLICIES


ROOT = Path(__file__).resolve().parent
DEFAULT_HOLDOUT = ROOT / "data" / "evaluation" / "semantic_holdout_v1.jsonl"
DEFAULT_LOCK = ROOT / "data" / "evaluation" / "semantic_holdout_v1.lock.json"
DEFAULT_CACHE = ROOT / "results" / "semantic_holdout_v1_candidate_cache.jsonl"
DEFAULT_REPORT = ROOT / "results" / "semantic_holdout_v1_report.json"
STRICT_SUBJECT_BUCKETS = {"CAUSE", "TIME", "PURPOSE"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_holdout(path: Path, lock_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(path)
    if lock.get("status") != "LOCKED_DO_NOT_TUNE":
        raise RuntimeError("Holdout is not marked LOCKED_DO_NOT_TUNE")
    if actual_hash != lock.get("dataset_sha256"):
        raise RuntimeError("Holdout checksum does not match its lock manifest")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != int(lock.get("rows", -1)):
        raise RuntimeError("Holdout row count does not match its lock manifest")
    return rows, lock


def relation_bucket(semantics: QuestionSemantics) -> str:
    relation = semantics.relation
    if relation == "CAUSE":
        return "CAUSE"
    if relation in {"BIRTH_TIME", "DEATH_TIME", "EVENT_TIME"}:
        return "TIME"
    if relation.endswith("_LOCATION"):
        return "LOCATION"
    if relation == "PURPOSE":
        return "PURPOSE"
    if relation == "DEFINITION":
        return "DEFINITION"
    if relation in {"IDENTITY", "ATTRIBUTE", "CONTRAST"} or any(
        item in {"ENTITY", "PERSON"} for item in semantics.question_type
    ):
        return "ENTITY"
    return "GENERAL"


def _fold_text(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text or "").casefold().replace("đ", "d"))
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return " ".join(re.findall(r"[\w]+", value, flags=re.UNICODE))


def plausible_subject(semantics: QuestionSemantics, question: str, expected: str) -> bool:
    subject = _fold_text(semantics.subject or "")
    if not subject or subject == _fold_text(question):
        return False
    interrogative_leak = re.search(
        r"\b(?:vi sao|tai sao|nguyen nhan|ly do|dieu gi|yeu to nao|o dau|tai dau|"
        r"noi nao|dia diem nao|nam nao|bao nhieu|muc dich gi|la gi|la ai|ai|nao)\b",
        subject,
    )
    if interrogative_leak:
        return False
    # When a typed clause requires a predicate, a missing predicate usually
    # means the parser swallowed SUBJECT + PREDICATE into the subject field.
    if expected in {"CAUSE", "TIME", "LOCATION", "PURPOSE", "DEFINITION"} and not semantics.predicate:
        return False
    return True


def parser_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    relation_ok: Counter[str] = Counter()
    subject_ok: Counter[str] = Counter()
    predicate_ok: Counter[str] = Counter()
    fully_ok: Counter[str] = Counter()
    unknown: Counter[str] = Counter()

    for row in rows:
        expected = str(row["expected_relation_bucket"])
        parsed = parse_question_semantics(row["question"])
        parsed_bucket = relation_bucket(parsed)
        totals[expected] += 1
        matches_relation = parsed_bucket == expected
        has_subject = plausible_subject(parsed, row["question"], expected)
        predicate_required = expected in {"CAUSE", "TIME", "LOCATION", "PURPOSE", "DEFINITION"}
        has_predicate = bool(parsed.predicate) if predicate_required else True
        relation_ok[expected] += int(matches_relation)
        subject_ok[expected] += int(has_subject)
        predicate_ok[expected] += int(has_predicate)
        fully_ok[expected] += int(matches_relation and has_subject and has_predicate)
        unknown[expected] += int(expected != "GENERAL" and parsed_bucket == "GENERAL")

    def rates(counter: Counter[str]) -> dict[str, dict[str, float | int]]:
        return {
            bucket: {
                "passed": counter[bucket],
                "total": total,
                "rate": round(100.0 * counter[bucket] / total, 3) if total else 0.0,
            }
            for bucket, total in sorted(totals.items())
        }

    total = sum(totals.values())
    return {
        "rows": total,
        "question_semantics_parse_rate": round(100.0 * sum(fully_ok.values()) / total, 3),
        "subject_parse_rate": round(100.0 * sum(subject_ok.values()) / total, 3),
        "relation_parse_rate": round(100.0 * sum(relation_ok.values()) / total, 3),
        "predicate_parse_rate": round(100.0 * sum(predicate_ok.values()) / total, 3),
        "relation_UNKNOWN_rate": round(100.0 * sum(unknown.values()) / total, 3),
        "by_relation": {
            "relation": rates(relation_ok),
            "subject": rates(subject_ok),
            "predicate": rates(predicate_ok),
            "fully_parsed": rates(fully_ok),
            "unknown": rates(unknown),
        },
    }


PARAPHRASE_CASES = (
    ("CAUSE", "Vì sao hệ thống Z phát triển vượt bậc?"),
    ("CAUSE", "Điều gì làm thị trường Z gia tăng nhanh chóng?"),
    ("CAUSE", "Yếu tố nào khiến ngành Z bùng nổ?"),
    ("CAUSE", "Nguyên nhân nào khiến dịch vụ Z trở nên phổ biến?"),
    ("CAUSE", "Vì đâu doanh nghiệp Z suy giảm?"),
    ("CAUSE", "Điều gì dẫn tới việc tổ chức Z sụp đổ?"),
    ("TIME", "Nguyễn Văn An chào đời vào năm nào?"),
    ("TIME", "Nguyễn Văn An được sinh ra năm bao nhiêu?"),
    ("TIME", "Năm sinh của Nguyễn Văn An là bao nhiêu?"),
    ("LOCATION", "Hội nghị Z được tổ chức tại địa điểm nào?"),
    ("PURPOSE", "Đoàn nghiên cứu Z được thành lập nhằm mục tiêu gì?"),
    ("ENTITY", "Tên gọi khác của tổ chức Z là gì?"),
)


def paraphrase_diagnostic() -> dict[str, Any]:
    rows = []
    passed = 0
    for expected, question in PARAPHRASE_CASES:
        semantics = parse_question_semantics(question)
        actual = relation_bucket(semantics)
        predicate_required = expected in {"CAUSE", "TIME", "LOCATION", "PURPOSE"}
        complete = actual == expected and bool(semantics.subject) and (
            bool(semantics.predicate) if predicate_required else True
        )
        passed += int(complete)
        rows.append(
            {
                "question": question,
                "expected_relation_bucket": expected,
                "actual_relation": semantics.relation,
                "actual_relation_bucket": actual,
                "subject": semantics.subject,
                "predicate": semantics.predicate,
                "passed": complete,
            }
        )
    return {"passed": passed, "total": len(rows), "rate": round(100.0 * passed / len(rows), 3), "cases": rows}


def post_question(endpoint: str, question: str, top_k: int) -> tuple[dict[str, Any], float]:
    payload = json.dumps(
        {"question": question, "retriever": "bm25", "reader": "phobert", "top_k": top_k},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result, (time.perf_counter() - started) * 1000


def ask_in_process(question: str, top_k: int) -> tuple[dict[str, Any], float]:
    from backend.viqa_api import ask_question

    started = time.perf_counter()
    result = ask_question(
        {"question": question, "retriever": "bm25", "reader": "phobert", "top_k": top_k}
    )
    return result, (time.perf_counter() - started) * 1000


def reduce_response(record: dict[str, Any], response: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for passage in response.get("passages", []):
        for candidate in passage.get("candidates", []):
            candidates.append(
                {
                    key: candidate.get(key)
                    for key in (
                        "text", "raw_text", "display_text", "method", "reader_score", "ranking_score",
                        "valid_span", "passes_reader_threshold", "passes_evidence_gate", "passes_type_gate",
                        "passes_relation_gate", "passes_completeness_gate", "subject_match_score",
                        "semantic_status", "semantic_relation", "relation_type", "rejection_reason",
                        "semantic_policy", "gate_results", "gates",
                    )
                }
            )
    return {
        "id": record["id"],
        "question": record["question"],
        "gold_answer": record["gold_answer"],
        "is_answerable": record["is_answerable"],
        "expected_relation_bucket": record["expected_relation_bucket"],
        "latency_ms": latency_ms,
        "production_answer": str(response.get("answer") or ""),
        "production_has_answer": bool(response.get("has_answer", response.get("answer"))),
        "production_rejection_reason": response.get("rejection_reason"),
        "candidates": candidates,
    }


def load_cache(cache_path: Path, holdout_hash: str, top_k: int) -> tuple[list[dict[str, Any]], set[str]]:
    if not cache_path.is_file():
        return [], set()
    lines = [line for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return [], set()
    metadata = json.loads(lines[0])
    if metadata.get("kind") != "semantic_holdout_candidate_cache_meta":
        raise RuntimeError("Candidate cache metadata is missing")
    if metadata.get("holdout_sha256") != holdout_hash or int(metadata.get("top_k", -1)) != top_k:
        raise RuntimeError("Candidate cache belongs to a different holdout or top_k")
    rows = [json.loads(line) for line in lines[1:]]
    return rows, {str(row["id"]) for row in rows}


def generate_cache(
    rows: list[dict[str, Any]],
    cache_path: Path,
    holdout_hash: str,
    endpoint: str,
    top_k: int,
    in_process: bool = False,
    max_new_rows: int | None = None,
) -> list[dict[str, Any]]:
    cached, completed = load_cache(cache_path, holdout_hash, top_k)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        cache_path.write_text(
            json.dumps(
                {
                    "kind": "semantic_holdout_candidate_cache_meta",
                    "holdout_sha256": holdout_hash,
                    "top_k": top_k,
                    "endpoint": endpoint,
                    "execution_mode": "in_process" if in_process else "http",
                    "semantic_policy_version": SEMANTIC_POLICIES.version,
                    "production_rules_frozen": True,
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
    remaining = [row for row in rows if str(row["id"]) not in completed]
    if max_new_rows is not None:
        remaining = remaining[:max_new_rows]
    for index, record in enumerate(remaining, start=1):
        response, latency_ms = (
            ask_in_process(record["question"], top_k)
            if in_process
            else post_question(endpoint, record["question"], top_k)
        )
        reduced = reduce_response(record, response, latency_ms)
        with cache_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(reduced, ensure_ascii=False, separators=(",", ":")) + "\n")
        cached.append(reduced)
        if index % 5 == 0 or index == len(remaining):
            print(f"Generated holdout candidates {len(cached)}/{len(rows)}", flush=True)
    return cached


def best_candidate(row: dict[str, Any], gate: Callable[[dict[str, Any]], bool], refined: bool) -> str:
    eligible = [candidate for candidate in row["candidates"] if gate(candidate)]
    if not eligible:
        return ""
    selected = max(
        eligible,
        key=lambda item: (float(item.get("reader_score") or 0.0), float(item.get("ranking_score") or 0.0)),
    )
    if refined:
        return str(selected.get("display_text") or selected.get("text") or "")
    return str(selected.get("raw_text") or selected.get("text") or "")


def ablation_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def reader_gate(candidate: dict[str, Any]) -> bool:
        return bool(
            candidate.get("method") == "neural_span"
            and candidate.get("valid_span")
            and candidate.get("passes_reader_threshold")
        )

    def refinement_gate(candidate: dict[str, Any]) -> bool:
        return reader_gate(candidate) and bool(candidate.get("passes_completeness_gate", True))

    def subject_gate(candidate: dict[str, Any], bucket: str) -> bool:
        if not refinement_gate(candidate):
            return False
        policy = SEMANTIC_POLICIES.get(str(candidate.get("semantic_relation") or "GENERAL"))
        return (
            not policy.require_subject_match
            or float(candidate.get("subject_match_score") or 0.0) >= policy.min_subject_score
        )

    def relation_gate(candidate: dict[str, Any], bucket: str) -> bool:
        if not subject_gate(candidate, bucket):
            return False
        return bucket not in STRICT_SUBJECT_BUCKETS or bool(candidate.get("passes_relation_gate"))

    variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        base = {
            "id": row["id"],
            "gold_answer": row["gold_answer"],
            "is_answerable": row["is_answerable"],
            "expected_relation_bucket": row["expected_relation_bucket"],
        }
        bucket = row["expected_relation_bucket"]
        predictions = {
            "reader_only": best_candidate(row, reader_gate, refined=False),
            "reader_plus_refinement": best_candidate(row, refinement_gate, refined=True),
            "reader_refinement_subject": best_candidate(row, lambda item: subject_gate(item, bucket), refined=True),
            "reader_refinement_subject_relation": best_candidate(
                row, lambda item: relation_gate(item, bucket), refined=True
            ),
            "full_pipeline": row["production_answer"],
        }
        for name, prediction in predictions.items():
            variants[name].append({**base, "predicted_answer": prediction})

    report = {}
    for name, predictions in variants.items():
        metrics = evaluate_predictions(predictions)
        by_relation: dict[str, Any] = {}
        buckets = sorted({str(row["expected_relation_bucket"]) for row in predictions})
        for bucket in buckets:
            subset = [row for row in predictions if row["expected_relation_bucket"] == bucket]
            subset_metrics = evaluate_predictions(subset)
            by_relation[bucket] = {
                "rows": len(subset),
                "em": subset_metrics["overall"]["em"],
                "f1": subset_metrics["overall"]["f1"],
                "answerable_em": subset_metrics["answerable"]["em"],
                "answerable_f1": subset_metrics["answerable"]["f1"],
                "unanswerable_accuracy": subset_metrics["unanswerable"]["accuracy"],
                "false_positive": sum(
                    1 for row in subset if not row["is_answerable"] and bool(row["predicted_answer"])
                ),
                "false_negative": sum(
                    1 for row in subset if row["is_answerable"] and not row["predicted_answer"]
                ),
            }
        report[name] = {
            "em": metrics["overall"]["em"],
            "f1": metrics["overall"]["f1"],
            "answerable_em": metrics["answerable"]["em"],
            "answerable_f1": metrics["answerable"]["f1"],
            "overall_f1": metrics["overall"]["f1"],
            "unanswerable_accuracy": metrics["unanswerable"]["accuracy"],
            "false_positive": sum(
                1 for row in predictions if not row["is_answerable"] and bool(row["predicted_answer"])
            ),
            "false_negative": sum(
                1 for row in predictions if row["is_answerable"] and not row["predicted_answer"]
            ),
            "false_positive_rate": 100.0
            * sum(1 for row in predictions if not row["is_answerable"] and bool(row["predicted_answer"]))
            / max(1, sum(1 for row in predictions if not row["is_answerable"])),
            "false_negative_rate": 100.0
            * sum(1 for row in predictions if row["is_answerable"] and not row["predicted_answer"])
            / max(1, sum(1 for row in predictions if row["is_answerable"])),
            "by_relation": by_relation,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate locked semantic_holdout_v1 without tuning it")
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/api/ask")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--generate-candidates", action="store_true")
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument("--max-new-rows", type=int)
    args = parser.parse_args()

    rows, lock = load_locked_holdout(args.holdout, args.lock)
    coverage = parser_coverage(rows)
    paraphrases = paraphrase_diagnostic()
    cache_rows, _ = load_cache(args.cache, lock["dataset_sha256"], args.top_k)
    if args.generate_candidates:
        cache_rows = generate_cache(
            rows,
            args.cache,
            lock["dataset_sha256"],
            args.endpoint,
            args.top_k,
            args.in_process,
            args.max_new_rows,
        )

    payload: dict[str, Any] = {
        "semantic_policy_version": SEMANTIC_POLICIES.version,
        "holdout": lock["name"],
        "holdout_status": lock["status"],
        "holdout_sha256": lock["dataset_sha256"],
        "rows": len(rows),
        "parser_coverage": coverage,
        "unseen_paraphrase_diagnostic": paraphrases,
        "candidate_rows": len(cache_rows),
        "methodology_notes": [
            "No threshold, retriever, checkpoint, or production semantic rule is tuned on this holdout.",
            "Relation strata are frozen weak labels and are not equivalent to human semantic annotation.",
            "Ablations rescore/select from one fixed production-generated candidate pool.",
            "Reader-only still shares retrieval and Reader span generation with the full pipeline.",
            "In-process latency excludes localhost HTTP serialization when --in-process is used.",
        ],
    }
    if len(cache_rows) == len(rows):
        payload["ablation"] = ablation_report(cache_rows)
        latencies = [float(row["latency_ms"]) for row in cache_rows]
        payload["latency_ms"] = {
            "average": statistics.fmean(latencies),
            "p50": statistics.median(latencies),
            "maximum": max(latencies),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": payload["rows"],
        "parser_coverage": coverage,
        "paraphrase_passed": f"{paraphrases['passed']}/{paraphrases['total']}",
        "candidate_rows": len(cache_rows),
        "ablation_ready": "ablation" in payload,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
