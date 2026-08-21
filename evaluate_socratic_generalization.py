from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from backend.socratic import (
    SocraticConfig,
    generate_followup_response,
    normalize_question,
    question_similarity,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "tests" / "data" / "socratic_generalization_holdout_v1.json"
DEFAULT_LOCK = ROOT / "tests" / "data" / "socratic_generalization_holdout_v1.lock.json"

RULE_DESCRIPTIONS = {
    "additional_grounded_fact": "additional subject-bound evidence sentence",
    "evidenced_activity": "subject + generic activity predicate",
    "evidenced_cause": "effect clause + causal connector + cause clause",
    "evidenced_comparison": "subject-bound comparison connector",
    "evidenced_consequence": "cause clause + consequence connector",
    "evidenced_context": "subject event/activity + context phrase",
    "evidenced_event_location": "event subject + explicit locative phrase",
    "evidenced_event_time": "subject-bound clause + explicit temporal value",
    "evidenced_height": "subject-bound height measure",
    "evidenced_identity": "subject + copular definition",
    "evidenced_object_location": "subject + static location predicate",
    "evidenced_purpose": "action clause + purpose connector + purpose clause",
    "evidenced_role": "subject + appointment/office grammar",
    "explicit_birth_location": "person subject + birth predicate + location",
    "explicit_birth_time": "person subject + birth predicate + time",
    "explicit_death_location": "person subject + death predicate + location",
    "explicit_death_time": "person subject + death predicate + time",
    "structural_numeric_attribute": "subject clause + measure predicate + numeric unit",
    "structural_process_location": "subject + unseen predicate + locative connector",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _relation_family(value: Any) -> str:
    relation = str(value or "GENERAL").upper()
    if relation in {"IDENTITY", "DEFINITION", "ENTITY"}:
        return "IDENTITY_DEFINITION"
    if relation.endswith("_TIME") or relation == "TIME":
        return "TIME"
    if relation.endswith("_LOCATION") or relation in {"LOCATION", "GENERIC_LOCATION"}:
        return "LOCATION"
    if relation in {"CAUSE", "CONSEQUENCE"}:
        return "CAUSE_OR_CONSEQUENCE"
    return relation


def _evaluation_semantics(case: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the Socratic benchmark independent from the main QA classifier.

    The holdout measures opportunity discovery rather than Reader/question-parser
    inference. The validation article title is the stable topical entity, while
    the source stratum supplies only the already-asked relation family.
    """

    relation = {
        "IDENTITY_DEFINITION": "DEFINITION",
        "TIME": "TIME",
        "LOCATION": "LOCATION",
        "CAUSE": "CAUSE",
        "PURPOSE": "PURPOSE",
        "ENTITY": "ENTITY",
        "ATTRIBUTE_GENERAL": "GENERAL",
        "NO_ANSWER_SPARSE": "GENERAL",
    }.get(str(case.get("stratum") or "GENERAL"), "GENERAL")
    return {
        "subject": str(case.get("title") or "").strip() or None,
        "relation": relation,
        "predicate": None,
        "target": None,
    }


def _mode_config(mode: str) -> SocraticConfig:
    from backend.socratic import SOCRATIC_CONFIG

    payload = dict(SOCRATIC_CONFIG.__dict__)
    payload["allow_bm25_probe"] = mode == "bm25_fallback"
    return SocraticConfig(**payload)


def _evaluate_mode(payload: Mapping[str, Any], mode: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rejection_distribution: Counter[str] = Counter()
    empty_statuses: Counter[str] = Counter()
    current_relation_total: Counter[str] = Counter()
    current_relation_covered: Counter[str] = Counter()
    opportunity_relation_total: Counter[str] = Counter()
    opportunity_relation_discovered: Counter[str] = Counter()
    opportunity_relation_accepted_cases: dict[str, set[str]] = defaultdict(set)
    rule_matches: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"cases": set(), "subjects": set(), "predicates": set(), "relations": set()}
    )
    latencies: list[float] = []
    generated = 0
    accepted = 0
    duplicate_count = 0
    off_topic_count = 0
    grounded_count = 0
    answerable_count = 0
    opportunity_cases = 0
    opportunity_cases_covered = 0
    human_review_queue: list[dict[str, Any]] = []

    for case in payload["cases"]:
        passage_map = {passage["passage_id"]: passage for passage in case["passages"]}
        semantics = _evaluation_semantics(case)
        selected_id = case["selected_passage_id"]
        retrieved_ids = case["retrieved_passage_ids"] if mode != "selected_only" else []

        def lookup(passage_id: str) -> Mapping[str, Any] | None:
            return passage_map.get(passage_id)

        def probe(_question: str, top_k: int) -> list[Mapping[str, Any]]:
            return list(passage_map.values())[:top_k]

        result = generate_followup_response(
            {
                "question": case["question"],
                "answer": case.get("answer"),
                "subject": semantics["subject"],
                "relation": semantics["relation"],
                "target": semantics["target"],
                "predicate": semantics["predicate"],
                "selected_passage_id": selected_id,
                "retrieved_passage_ids": retrieved_ids,
                "visited_relations": [semantics["relation"]],
                "asked_questions": [case["question"]],
                "limit": 3,
                "debug": True,
            },
            passage_lookup=lookup,
            probe=probe if mode == "bm25_fallback" else None,
            config=_mode_config(mode),
        )
        debug = result.get("debug", {})
        generation = debug.get("candidate_generation", {})
        semantic_opportunities = debug.get("semantic_opportunities", {})
        generated += int(generation.get("generated", 0))
        rejection_distribution.update(debug.get("rejection_distribution", {}))
        empty_statuses.update([str(debug.get("status") or "UNKNOWN")])
        latency = float(debug.get("latency", {}).get("total_ms", result.get("processing_time_ms", 0)))
        latencies.append(latency)

        current_bucket = str(case["stratum"])
        current_relation_total[current_bucket] += 1
        expected_relations = set(case.get("weak_available_followup_relations") or [])
        if case.get("opportunity_available"):
            opportunity_cases += 1
            opportunity_relation_total.update(expected_relations)
        discovered_relations = {
            _relation_family(relation)
            for relation, count in semantic_opportunities.get("by_relation", {}).items()
            if count
        }
        for relation in expected_relations & discovered_relations:
            opportunity_relation_discovered[relation] += 1

        traces = debug.get("candidates", [])
        accepted_traces = [trace for trace in traces if trace.get("accepted")]
        trace_by_question = {
            normalize_question(trace.get("question", "")): trace for trace in accepted_traces
        }
        followups: list[dict[str, Any]] = []
        seen_questions: list[str] = []
        accepted_relations: set[str] = set()
        for followup in result.get("followups", []):
            accepted += 1
            trace = trace_by_question.get(normalize_question(followup.get("question", "")), {})
            source = passage_map.get(str(followup.get("source_passage_id") or ""))
            evidence = str(trace.get("evidence_sentence") or "")
            grounded = bool(source and evidence and normalize_question(evidence) in normalize_question(source["text"]))
            grounded_count += int(grounded)
            answerable = grounded and float(followup.get("answerability_score") or 0.0) >= 0.62
            answerable_count += int(answerable)
            duplicate = question_similarity(followup["question"], case["question"]) >= 0.93 or any(
                question_similarity(followup["question"], previous) >= 0.93
                for previous in seen_questions
            )
            duplicate_count += int(duplicate)
            seen_questions.append(followup["question"])
            expected_subject = normalize_question(semantics["subject"] or "")
            actual_subject = normalize_question(followup.get("subject") or "")
            off_topic = bool(expected_subject and actual_subject and expected_subject != actual_subject)
            off_topic_count += int(off_topic)
            relation_family = _relation_family(followup.get("relation"))
            accepted_relations.add(relation_family)
            if relation_family in expected_relations:
                opportunity_relation_accepted_cases[relation_family].add(case["id"])
            generated_by = str(trace.get("generated_by") or "UNKNOWN")
            rule_id = generated_by.split(":", 1)[0]
            rule_matches[rule_id]["cases"].add(case["id"])
            rule_matches[rule_id]["subjects"].add(str(followup.get("subject") or ""))
            rule_matches[rule_id]["predicates"].add(str(trace.get("predicate") or ""))
            rule_matches[rule_id]["relations"].add(str(followup.get("relation") or ""))
            graded = {
                **followup,
                "grounded_proxy": grounded,
                "answerable_proxy": answerable,
                "duplicate_proxy": duplicate,
                "off_topic_proxy": off_topic,
                "expected_relation_proxy": relation_family in expected_relations,
            }
            followups.append(graded)
            human_review_queue.append(
                {
                    "case_id": case["id"],
                    "original_question": case["question"],
                    "followup": followup["question"],
                    "relation": followup.get("relation"),
                    "source_passage_id": followup.get("source_passage_id"),
                    "evidence_sentence": evidence,
                    "source_passage": source.get("text") if source else None,
                    "review": {
                        "status": "NOT_REVIEWED_BY_HUMAN",
                        "grounded": None,
                        "relevant": None,
                        "novel": None,
                        "answerable": None,
                        "natural": None,
                    },
                }
            )
        if followups:
            current_relation_covered[current_bucket] += 1
        if case.get("opportunity_available") and followups:
            opportunity_cases_covered += 1
        rows.append(
            {
                "id": case["id"],
                "stratum": case["stratum"],
                "question": case["question"],
                "parsed_semantics": {
                    "subject": semantics["subject"],
                    "relation": semantics["relation"],
                    "predicate": semantics["predicate"],
                    "target": semantics["target"],
                },
                "expected_opportunity_relations_weak": sorted(expected_relations),
                "discovered_relations": sorted(discovered_relations),
                "accepted_relations": sorted(accepted_relations),
                "status": debug.get("status"),
                "followups": followups,
                "rejections": debug.get("rejection_distribution", {}),
                "candidate_generation": generation,
                "latency_ms": round(latency, 3),
            }
        )

    total_cases = len(rows)
    cases_with_followup = sum(bool(row["followups"]) for row in rows)
    final_denominator = max(1, accepted)
    known_opportunities = sum(opportunity_relation_total.values())
    discovered_known = sum(opportunity_relation_discovered.values())
    relation_coverage = {
        relation: {
                "known_available_cases": opportunity_relation_total[relation],
                "discovered_cases": opportunity_relation_discovered[relation],
                "final_accepted_cases": len(opportunity_relation_accepted_cases[relation]),
            "opportunity_recall": round(
                opportunity_relation_discovered[relation] / max(1, opportunity_relation_total[relation]), 4
            ),
        }
        for relation in sorted(opportunity_relation_total)
    }
    coverage_by_stratum = {
        relation: {
            "total_cases": current_relation_total[relation],
            "cases_with_followup": current_relation_covered[relation],
            "coverage": round(current_relation_covered[relation] / max(1, current_relation_total[relation]), 4),
        }
        for relation in sorted(current_relation_total)
    }
    rule_report = []
    for rule_id, values in sorted(rule_matches.items()):
        matched_cases = len(values["cases"])
        unique_subjects = len({value for value in values["subjects"] if value})
        rule_report.append(
            {
                "rule_id": rule_id,
                "pattern": RULE_DESCRIPTIONS.get(rule_id, "unregistered diagnostic origin"),
                "matched_cases": matched_cases,
                "unique_subjects": unique_subjects,
                "unique_predicates": len({value for value in values["predicates"] if value}),
                "relations": sorted(values["relations"]),
                "flag": "POSSIBLE_CASE_OVERFIT" if matched_cases == 1 and unique_subjects == 1 else None,
            }
        )
    rng = random.Random("socratic-human-review-queue-v1")
    rng.shuffle(human_review_queue)
    return {
        "mode": mode,
        "total_cases": total_cases,
        "cases_with_followup": cases_with_followup,
        "coverage_rate": round(cases_with_followup / max(1, total_cases), 4),
        "empty_rate": round((total_cases - cases_with_followup) / max(1, total_cases), 4),
        "opportunity_cases": opportunity_cases,
        "opportunity_cases_covered": opportunity_cases_covered,
        "opportunity_aware_coverage": round(opportunity_cases_covered / max(1, opportunity_cases), 4),
        "known_available_relation_opportunities": known_opportunities,
        "discovered_known_relation_opportunities": discovered_known,
        "opportunity_recall": round(discovered_known / max(1, known_opportunities), 4),
        "total_candidates_generated": generated,
        "total_followups": accepted,
        "avg_followups_per_case": round(accepted / max(1, total_cases), 4),
        "grounding_rate_proxy": round(grounded_count / final_denominator, 4),
        "answerability_rate_proxy": round(answerable_count / final_denominator, 4),
        "duplicate_rate": round(duplicate_count / final_denominator, 4),
        "off_topic_rate": round(off_topic_count / final_denominator, 4),
        "coverage_by_stratum": coverage_by_stratum,
        "relation_opportunity_coverage": relation_coverage,
        "empty_statuses": dict(empty_statuses),
        "rejection_distribution": dict(rejection_distribution.most_common()),
        "latency_ms": {
            "mean": round(mean(latencies), 3) if latencies else 0.0,
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
        "rule_coverage": rule_report,
        "human_review_queue": human_review_queue[:50],
        "human_review_status": "NOT_PERFORMED_REQUIRES_INDEPENDENT_REVIEWER",
        "cases": rows,
    }


def evaluate(input_path: Path, lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    actual = _sha256(input_path)
    if actual != lock["dataset_sha256"]:
        raise RuntimeError(f"Locked holdout checksum mismatch: expected {lock['dataset_sha256']}, got {actual}")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("status") != "LOCKED_DO_NOT_TUNE":
        raise RuntimeError("Generalization holdout is not locked")
    return {
        "dataset": payload["name"],
        "dataset_sha256": actual,
        "status": payload["status"],
        "annotation_warning": "Opportunity labels and quality metrics are automated proxies, not human gold.",
        "modes": {
            mode: _evaluate_mode(payload, mode)
            for mode in ("selected_only", "selected_plus_retrieved", "bm25_fallback")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate locked Socratic generalization holdout")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--write-supporting-artifacts",
        action="store_true",
        help="Write rule coverage and rejection analysis beside the main report.",
    )
    args = parser.parse_args()
    report = evaluate(args.input, args.lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.write_supporting_artifacts:
        rule_payload = {
            "dataset": report["dataset"],
            "dataset_sha256": report["dataset_sha256"],
            "note": "Low-coverage linguistic rules are flagged for review, not automatically removed.",
            "modes": {
                mode: metrics["rule_coverage"] for mode, metrics in report["modes"].items()
            },
        }
        rejection_payload = {
            "dataset": report["dataset"],
            "dataset_sha256": report["dataset_sha256"],
            "modes": {
                mode: {
                    "empty_statuses": metrics["empty_statuses"],
                    "rejection_distribution": metrics["rejection_distribution"],
                    "cases": [
                        {
                            "id": row["id"],
                            "status": row["status"],
                            "rejections": row["rejections"],
                            "candidate_generation": row["candidate_generation"],
                        }
                        for row in metrics["cases"]
                    ],
                }
                for mode, metrics in report["modes"].items()
            },
        }
        (args.output.parent / "socratic_rule_coverage.json").write_text(
            json.dumps(rule_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output.parent / "socratic_rejection_analysis.json").write_text(
            json.dumps(rejection_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {
        mode: {
            key: value
            for key, value in metrics.items()
            if key in {
                "coverage_rate", "opportunity_aware_coverage", "opportunity_recall",
                "avg_followups_per_case", "empty_rate", "grounding_rate_proxy",
                "answerability_rate_proxy", "duplicate_rate", "off_topic_rate", "latency_ms",
            }
        }
        for mode, metrics in report["modes"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
