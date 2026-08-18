import unittest

from reader.answer_completeness import (
    assess_answer_completeness,
    refine_dangling_clause,
)
from reader.question_semantics import parse_question_semantics
from reader.relation_validator import validate_candidate_relation
from reader.subject_consistency import score_subject_consistency


class SharedQuestionSemanticsTests(unittest.TestCase):
    def test_separates_cause_subject_predicate_and_modifier(self):
        semantics = parse_question_semantics(
            "Vì sao chủ nghĩa nô lệ phát triển mạnh tại các tiểu bang miền Nam?"
        )
        self.assertEqual(semantics.relation, "CAUSE")
        self.assertEqual(semantics.subject, "chủ nghĩa nô lệ")
        self.assertEqual(semantics.predicate, "phát triển mạnh")
        self.assertEqual(semantics.modifier, "tại các tiểu bang miền Nam")

    def test_keeps_passive_predicate_together(self):
        semantics = parse_question_semantics("Vì sao Madame du Barry bị khinh miệt?")
        self.assertEqual(semantics.subject, "Madame du Barry")
        self.assertEqual(semantics.predicate, "bị khinh miệt")

    def test_birth_time_paraphrases_share_semantics(self):
        questions = (
            "Phạm Văn Đồng sinh năm nào?",
            "Phạm Văn Đồng sinh vào năm bao nhiêu?",
            "Năm sinh của Phạm Văn Đồng là bao nhiêu?",
        )
        parsed = [parse_question_semantics(question) for question in questions]
        self.assertEqual({item.relation for item in parsed}, {"BIRTH_TIME"})
        self.assertEqual({item.subject for item in parsed}, {"Phạm Văn Đồng"})
        self.assertEqual({item.predicate for item in parsed}, {"sinh"})


class SharedSubjectAndRelationTests(unittest.TestCase):
    def test_subject_consistency_rejects_unrelated_time_passage(self):
        semantics = parse_question_semantics("Phạm Văn Đồng sinh năm nào?")
        evidence = "Love Story, xuất bản năm 1970, là tác phẩm của Erich Segal."
        subject = score_subject_consistency(semantics, evidence, "năm 1970", evidence)
        self.assertEqual(subject.status, "INVALID")
        self.assertEqual(subject.score, 0.0)

        relation = validate_candidate_relation(
            semantics,
            "Phạm Văn Đồng sinh năm nào?",
            evidence,
            "năm 1970",
            evidence.index("năm 1970"),
            evidence.index("năm 1970") + len("năm 1970"),
            "phrase_fallback",
            {"sentence_answer": evidence},
        )
        self.assertFalse(relation.relation_evidence)
        self.assertEqual(relation.reason, "TIME_SUBJECT_MISMATCH")

    def test_birth_time_accepts_biography_parenthetical(self):
        question = "Phạm Văn Đồng sinh năm nào?"
        semantics = parse_question_semantics(question)
        evidence = (
            "Phạm Văn Đồng (1 tháng 3 năm 1906 – 29 tháng 4 năm 2000) "
            "là Thủ tướng đầu tiên của Việt Nam."
        )
        answer = "năm 1906"
        start = evidence.index(answer)
        relation = validate_candidate_relation(
            semantics,
            question,
            evidence,
            answer,
            start,
            start + len(answer),
            "phrase_fallback",
            {"sentence_answer": evidence},
        )
        self.assertTrue(relation.relation_evidence)
        self.assertEqual(relation.reason, "BIOGRAPHY_PARENTHETICAL_TIME")
        self.assertGreaterEqual(relation.subject_match_score, 0.9)

    def test_cause_without_cause_effect_evidence_is_unknown_not_neutral(self):
        question = "Vì sao Madame du Barry bị khinh miệt?"
        semantics = parse_question_semantics(question)
        evidence = "Nhiều quý tộc khinh miệt Madame du Barry."
        relation = validate_candidate_relation(
            semantics,
            question,
            evidence,
            evidence,
            0,
            len(evidence),
            "neural_span",
        )
        self.assertFalse(relation.relation_evidence)
        self.assertEqual(relation.relation_score, 0.0)
        self.assertIn(relation.reason, {"CAUSE_RELATION_NOT_FOUND", "CAUSE_EFFECT_REPETITION"})


class AnswerCompletenessTests(unittest.TestCase):
    def test_expands_dangling_connector_to_coordinated_clause(self):
        context = (
            "Hitler khinh thường Roosevelt vì cho rằng ông là một lãnh tụ yếu thế, "
            "hay dao động và vì thế đánh giá thấp Hoa Kỳ."
        )
        raw = "vì cho rằng ông là một lãnh tụ yếu thế, hay"
        start = context.index(raw)
        end = start + len(raw)
        refinement = refine_dangling_clause(context, start, end)
        refined = context[refinement.start : refinement.end]
        self.assertEqual(
            refined,
            "vì cho rằng ông là một lãnh tụ yếu thế, hay dao động",
        )
        self.assertEqual(refinement.method, "dangling_connector_expand_right")
        self.assertTrue(refinement.after.complete)

    def test_detects_all_required_dangling_connectors(self):
        for connector in ("và", "hay", "hoặc", "nhưng", "mà", "vì", "do", "bởi", "nên", "rằng", "nếu", "khi", "còn", "trong khi"):
            with self.subTest(connector=connector):
                result = assess_answer_completeness(f"một mệnh đề {connector}")
                self.assertFalse(result.complete)
                self.assertIn("DANGLING_CONNECTOR", result.reasons)


if __name__ == "__main__":
    unittest.main()
