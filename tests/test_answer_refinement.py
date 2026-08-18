import unittest

from reader.answer_refinement import (
    assess_relation_completeness,
    refine_answer,
)
from reader.config import max_answer_length_for_type
from reader.fallback_extractor import extract_fallback_answer
from reader.question_type import QuestionType
from evaluate_reranking import _evaluate_configuration


class AnswerRefinementTests(unittest.TestCase):
    def test_typed_answer_length_caps(self):
        self.assertEqual(max_answer_length_for_type(QuestionType.TIME), 12)
        self.assertEqual(max_answer_length_for_type(QuestionType.NUMBER), 10)
        self.assertEqual(max_answer_length_for_type(QuestionType.PERSON), 16)
        self.assertEqual(max_answer_length_for_type(QuestionType.LOCATION), 20)
        self.assertEqual(max_answer_length_for_type(QuestionType.ENTITY), 24)
        self.assertEqual(max_answer_length_for_type(QuestionType.DEFINITION), 48)
        self.assertEqual(max_answer_length_for_type(QuestionType.GENERAL), 64)

    def test_expands_missing_entity_designator(self):
        context = "Đỉnh là vị trí nhà thờ Saint-Pierre, được xây dựng từ lâu."
        raw = "Saint-Pierre"
        start = context.index(raw)

        result = refine_answer(
            "Công trình nào nằm trên đỉnh?",
            QuestionType.ENTITY,
            context,
            start,
            start + len(raw),
        )

        self.assertEqual(result.raw_answer, raw)
        self.assertEqual(result.refined_answer, "nhà thờ Saint-Pierre")
        self.assertEqual(context[result.final_start : result.final_end], result.refined_answer)
        self.assertIn("noun_phrase_expand_left", result.refinement_method)

    def test_compresses_scaffold_and_relative_clause(self):
        context = "Đáp án là nhà thờ Saint-Pierre, được xây dựng từ lâu."
        raw = "là nhà thờ Saint-Pierre, được xây dựng từ lâu"
        start = context.index(raw)

        result = refine_answer(
            "Công trình nào được nhắc đến?",
            QuestionType.ENTITY,
            context,
            start,
            start + len(raw),
        )

        self.assertEqual(result.refined_answer, "nhà thờ Saint-Pierre")
        self.assertIn("leading_scaffold_compression", result.refinement_method)
        self.assertIn("relative_clause_compression", result.refinement_method)

    def test_incomplete_contrast_is_rejected_structurally(self):
        score, complete, reasons = assess_relation_completeness(
            "A và B khác nhau như thế nào?",
            QuestionType.GENERAL,
            "A tập trung vào chức năng học thực sự",
        )

        self.assertFalse(complete)
        self.assertEqual(score, 0.25)
        self.assertIn("MISSING_CONTRAST_SIDE", reasons)

    def test_extracts_cause_clause(self):
        sentence = (
            "Ông phải rời Versailles vì xuất thân thấp kém "
            "(ông là con của một công chứng viên), khiến triều đình không vừa mắt."
        )
        candidate = extract_fallback_answer(
            "Vì sao ông phải rời Versailles?",
            QuestionType.GENERAL,
            sentence,
        )

        self.assertEqual(candidate.answer, "xuất thân thấp kém")
        self.assertEqual(candidate.method, "cause_clause_pattern")
        self.assertEqual(candidate.relation_type, "CAUSE")

    def test_rejects_cause_span_cut_after_abstract_noun(self):
        score, complete, reasons = assess_relation_completeness(
            "Vì sao ông phải rời Versailles?",
            QuestionType.GENERAL,
            "nhưng do xuất thân",
        )

        self.assertFalse(complete)
        self.assertEqual(score, 0.2)
        self.assertIn("INCOMPLETE_CAUSE_PHRASE", reasons)

    def test_rejects_span_cut_inside_parenthetical(self):
        score, complete, reasons = assess_relation_completeness(
            "Vì sao ông phải rời Versailles?",
            QuestionType.GENERAL,
            "do xuất thân thấp kém (ông là con của một công chứng viên",
        )

        self.assertFalse(complete)
        self.assertEqual(score, 0.15)
        self.assertIn("UNBALANCED_DELIMITER", reasons)

    def test_extracts_purpose_clause(self):
        sentence = "Đoàn sứ giả được gửi đi để ngăn một cuộc tấn công vào Aleppo."
        candidate = extract_fallback_answer(
            "Đoàn sứ giả được gửi đi với mục đích gì?",
            QuestionType.GENERAL,
            sentence,
        )

        self.assertEqual(candidate.answer, "ngăn một cuộc tấn công vào Aleppo")
        self.assertEqual(candidate.method, "purpose_clause_pattern")
        self.assertEqual(candidate.relation_type, "PURPOSE")

    def test_evaluator_reports_raw_vs_refined_improvement(self):
        rows = [
            {
                "id": "case-1",
                "gold_answer": "Lâm Bá Kiệt",
                "gold_normalized": "lâm bá kiệt",
                "is_answerable": True,
                "gold_available": True,
                "candidates": [
                    {
                        "text": "Lâm Bá Kiệt",
                        "display_text": "Lâm Bá Kiệt",
                        "raw_text": "là Lâm Bá",
                        "refinement_method": "leading_scaffold_compression+phrase_expand_right",
                        "refinement_changed": True,
                        "method": "neural_span",
                        "retrieval_score": 1.0,
                        "reader_score": 1.0,
                        "answer_type_score": 1.0,
                        "relation_score": 1.0,
                        "boundary_score": 1.0,
                        "completeness_score": 1.0,
                        "fallback_penalty": 1.0,
                        "valid_span": True,
                        "passes_evidence_gate": True,
                        "passes_type_gate": True,
                        "passes_relation_gate": True,
                        "passes_completeness_gate": True,
                        "passage_text": "Ông có tên gọi là Lâm Bá Kiệt.",
                    }
                ],
            }
        ]
        metrics, predictions = _evaluate_configuration(
            rows,
            {"retriever": 0.4, "reader": 0.3, "answer_type": 0.2, "relation": 0.1},
            0.625,
        )

        self.assertGreater(metrics["refinement"]["answerable_f1_delta"], 0.0)
        self.assertEqual(metrics["refinement"]["outcomes"]["IMPROVED"], 1)
        self.assertEqual(predictions[0]["raw_predicted_answer"], "là Lâm Bá")


if __name__ == "__main__":
    unittest.main()
