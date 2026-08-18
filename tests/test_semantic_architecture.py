import ast
import json
import unittest
from pathlib import Path

from backend.viqa_api import _candidate_rejection, build_passage_candidates
from reader.candidate_validation import GateResult, first_gate_failure
from reader.candidates import AnswerCandidate
from reader.question_semantics import parse_question_semantics
from reader.question_type import QuestionType
from reader.relation_validator import RELATION_HANDLERS, validate_candidate_relation
from reader.semantic_policy import SEMANTIC_POLICIES


ROOT = Path(__file__).resolve().parents[1]


class SemanticArchitectureTests(unittest.TestCase):
    def test_candidate_rejection_is_generic_and_short(self):
        source = Path("backend/viqa_api.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_candidate_rejection"
        )
        segment = ast.get_source_segment(source, function) or ""
        self.assertLessEqual(function.end_lineno - function.lineno + 1, 3)
        self.assertNotIn("CAUSE", segment)
        self.assertNotIn("LOCATION", segment)
        self.assertNotIn("method", segment)
        self.assertNotRegex(segment, r"0\.\d+")

    def test_backend_candidate_decision_has_no_relation_specific_branch(self):
        source = Path("backend/viqa_api.py").read_text(encoding="utf-8")
        self.assertNotIn('candidate.relation_type == "CAUSE"', source)
        self.assertNotIn('"LOCATION" in candidate.relation_type', source)
        self.assertNotIn('candidate.method == "sentence_fallback"', source)

    def test_unknown_required_relation_does_not_pass(self):
        semantics = parse_question_semantics("Câu hỏi tổng quát là gì?")
        semantics = semantics.__class__(
            question_type=semantics.question_type,
            relation="UNREGISTERED_RELATION",
            subject="đối tượng",
            predicate="hành động",
            target=None,
            modifier=None,
            expected_answer_type=semantics.expected_answer_type,
        )
        result = validate_candidate_relation(
            semantics,
            "Câu hỏi",
            "Ngữ cảnh có câu trả lời.",
            "câu trả lời",
            12,
            23,
            "neural_span",
        )
        self.assertFalse(result.relation_evidence)
        self.assertEqual(result.status, "UNKNOWN")
        self.assertEqual(result.reason, "RELATION_UNSUPPORTED")
        self.assertNotIn("UNREGISTERED_RELATION", RELATION_HANDLERS)

    def test_first_gate_failure_propagates_validator_reason(self):
        gates = {
            "span": GateResult("span", True),
            "relation": GateResult("relation", False, 0.0, 0.5, "CAUSE_RELATION_NOT_FOUND"),
        }
        self.assertEqual(first_gate_failure(gates), "CAUSE_RELATION_NOT_FOUND")

    def test_candidate_serializes_structured_gate_trace(self):
        context = "Cây lúa phát triển nhanh nhờ nguồn nước dồi dào."
        answer = "nguồn nước dồi dào"
        start = context.index(answer)
        candidates = build_passage_candidates(
            "Vì sao cây lúa phát triển nhanh?",
            [QuestionType.GENERAL],
            "synthetic_P0001",
            context,
            1.0,
            {
                "confidence": 0.9,
                "span_candidates": [
                    {
                        "text": answer,
                        "start": start,
                        "end": start + len(answer),
                        "score_margin": 1.0,
                        "reader_threshold_score": 0.9,
                        "valid_span": True,
                    }
                ],
            },
            {"answer": "", "confidence": 0.0},
        )
        serialized = candidates[0].to_dict()
        self.assertEqual(serialized["semantic_policy"], "CAUSE")
        self.assertEqual(tuple(serialized["gate_results"]), SEMANTIC_POLICIES.gate_order)
        for gate in serialized["gate_results"].values():
            self.assertEqual(set(gate), {"name", "passed", "score", "threshold", "reason"})

    def test_gate_thresholds_are_not_numeric_comparisons_in_validators(self):
        modules = (
            "candidate_validation.py",
            "cause_relations.py",
            "subject_consistency.py",
            "relation_validator.py",
            "answer_completeness.py",
            "span_boundaries.py",
            "fallback_extractor.py",
        )
        for name in modules:
            source = (ROOT / "reader" / name).read_text(encoding="utf-8")
            module = ast.parse(source)
            numeric_comparators = [
                node
                for node in ast.walk(module)
                if isinstance(node, ast.Compare)
                and any(isinstance(operator, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for operator in node.ops)
                and any(
                    isinstance(item, ast.Constant) and isinstance(item.value, float)
                    for item in node.comparators
                )
            ]
            self.assertEqual(numeric_comparators, [], name)

    def test_general_predicate_extraction_has_no_benchmark_literal(self):
        semantics = parse_question_semantics("Vì sao cây lúa phát triển nhanh ở đồng bằng?")
        self.assertEqual(semantics.relation, "CAUSE")
        self.assertEqual(semantics.subject, "cây lúa")
        self.assertEqual(semantics.predicate, "phát triển nhanh")
        self.assertEqual(semantics.modifier, "ở đồng bằng")

        birth = parse_question_semantics("Nguyễn Văn An sinh năm nào?")
        self.assertEqual(birth.relation, "BIRTH_TIME")
        self.assertEqual(birth.subject, "Nguyễn Văn An")

    def test_semantic_config_is_versioned_and_contains_no_benchmark_entity(self):
        text = (ROOT / "config" / "semantic_policy.json").read_text(encoding="utf-8")
        payload = json.loads(text)
        self.assertEqual(payload["semantic_policy_version"], "v2")
        for forbidden in ("Voltaire", "Phạm Văn Đồng", "Roosevelt", "Paris", "Saint-Pierre"):
            self.assertNotIn(forbidden, text)

    def test_production_does_not_import_evaluation_fixtures(self):
        for directory in (ROOT / "backend", ROOT / "reader"):
            for path in directory.glob("*.py"):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("tests/data", text, path)
                self.assertNotIn("manual_live_semantic_set", text, path)
                self.assertNotIn("qa_semantic_regressions", text, path)


if __name__ == "__main__":
    unittest.main()
