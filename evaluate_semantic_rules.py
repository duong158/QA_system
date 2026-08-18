from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from reader.question_semantics import (
    MODIFIER_RULE,
    PREDICATE_BOUNDARY_RULES,
    QUESTION_RULES,
    Rule,
    parse_question_semantics,
)
from reader.semantic_policy import SEMANTIC_POLICIES


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "evaluation" / "semantic_holdout_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "semantic_rule_coverage_v2.json"
LOW_COVERAGE_THRESHOLD = 5


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    for key in ("examples", "items", "data", "rows"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"Cannot find evaluation rows in {path}")


def relation_bucket(relation: str) -> str:
    if relation in {"BIRTH_TIME", "DEATH_TIME", "EVENT_TIME"}:
        return "TIME"
    if relation.endswith("_LOCATION"):
        return "LOCATION"
    if relation in {"IDENTITY", "ATTRIBUTE", "CONTRAST"}:
        return "ENTITY"
    return relation


def all_rules() -> tuple[Rule, ...]:
    return (*QUESTION_RULES, *PREDICATE_BOUNDARY_RULES, MODIFIER_RULE)


def expected_bucket(row: dict[str, Any]) -> str | None:
    value = row.get("expected_relation_bucket") or row.get("relation")
    return str(value) if value else None


def matched_rule_ids(question: str, parsed) -> set[str]:
    return set(parsed.matched_rule_ids)


def rule_is_correct(rule: Rule, parsed, expected: str | None) -> bool:
    if rule.category == "question_relation":
        return expected is None or relation_bucket(str(rule.relation)) == expected
    if rule.category == "predicate_boundary":
        return bool(parsed.subject and parsed.predicate)
    if rule.category == "predicate_modifier":
        return bool(parsed.modifier)
    return False


def coverage_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counters: dict[str, dict[str, int]] = defaultdict(
        lambda: {"matched_count": 0, "correct_count": 0, "incorrect_count": 0}
    )
    rules = {rule.id: rule for rule in all_rules()}
    row_count = 0
    for row in rows:
        question = str(row.get("question") or "")
        if not question:
            continue
        row_count += 1
        parsed = parse_question_semantics(question)
        expected = expected_bucket(row)
        for rule_id in matched_rule_ids(question, parsed):
            rule = rules[rule_id]
            counters[rule_id]["matched_count"] += 1
            key = "correct_count" if rule_is_correct(rule, parsed, expected) else "incorrect_count"
            counters[rule_id][key] += 1

    results = []
    for rule in all_rules():
        counts = counters[rule.id]
        matched = counts["matched_count"]
        correct = counts["correct_count"]
        warnings = ["LOW_COVERAGE_RULE"] if matched < LOW_COVERAGE_THRESHOLD else []
        results.append(
            {
                "rule_id": rule.id,
                "category": rule.category,
                "description": rule.description,
                "general": rule.general,
                **counts,
                "precision": round(correct / matched, 6) if matched else None,
                "warnings": warnings,
            }
        )
    return {
        "semantic_policy_version": SEMANTIC_POLICIES.version,
        "rows": row_count,
        "low_coverage_threshold": LOW_COVERAGE_THRESHOLD,
        "rules": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit semantic linguistic rule coverage")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = coverage_report(load_rows(args.dataset))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
