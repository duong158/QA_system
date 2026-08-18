import json
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.chunking import Passage
from backend.viqa_api import IndexedPassage, SearchHit, _semantic_relation_assessment, ask_question
from reader.answer_refinement import (
    QuestionRelation,
    assess_relation_completeness,
    detect_question_relation,
)
from reader.cause_relations import (
    assess_cause_candidate,
    extract_cause_candidate,
    extract_cause_question,
)
from reader.question_type import QuestionType, detect_question_type


CORRECT_SENTENCE = (
    "Những năm 1740 Voltaire được triều đình chào đón với tư cách nhà viết kịch "
    "và nhà thơ, nhưng do xuất thân thấp kém (ông là con của một công chứng viên "
    "và cha ông cũng là người Jansen) khiến Vua và Hòang hậu thấy không vừa mắt, "
    "cuối cùng ông buộc phải rời khỏi Versailles."
)
WRONG_SENTENCE = (
    "Chủ nghĩa nô lệ bị bãi bỏ tại các tiểu bang miền Bắc, nhưng lại phát triển "
    "mạnh tại các tiểu bang miền Nam vì nhu cầu lớn về bông vải tại châu Âu."
)


def make_hit(passage_id: str, text: str, normalized: float, rank: int) -> SearchHit:
    metadata = Passage("DOC", passage_id, "Title", "DOC_PAR0001", 0, 0, text)
    return SearchHit(
        IndexedPassage(metadata, tuple(), {}),
        retrieval_score_raw=normalized * 10,
        retrieval_score_normalized=normalized,
        retrieval_rank=rank,
    )


class EmptyReader:
    def predict_many(self, question, contexts, **kwargs):
        return [
            {
                "answer": "",
                "candidate_answer": "",
                "candidate_start": -1,
                "candidate_end": -1,
                "confidence": 0.0,
                "reader_threshold_score": 0.0,
                "score_margin": -4.0,
                "passes_reader_threshold": False,
                "valid_span": False,
                "has_answer": False,
            }
            for _ in contexts
        ]


class CauseSemanticTests(unittest.TestCase):
    def test_cause_paraphrases_share_relation_subject_and_target(self):
        questions = (
            "Vì sao Voltaire bị khinh miệt?",
            "Tại sao Voltaire bị khinh miệt?",
            "Nguyên nhân nào khiến Voltaire bị khinh miệt?",
            "Điều gì khiến Voltaire bị khinh miệt?",
            "Lý do nào khiến Voltaire bị khinh miệt?",
        )
        frames = []
        for question in questions:
            self.assertEqual(detect_question_type(question), QuestionType.GENERAL)
            self.assertEqual(
                detect_question_relation(question, QuestionType.GENERAL),
                QuestionRelation.CAUSE,
            )
            frame = extract_cause_question(question)
            self.assertIsNotNone(frame)
            frames.append(frame)
        self.assertEqual({frame.subject for frame in frames}, {"Voltaire"})
        self.assertEqual({frame.target for frame in frames}, {"bị khinh miệt"})

    def test_extracts_directional_cause_for_requested_subject(self):
        question = "Vì sao Voltaire bị khinh miệt?"
        candidate = extract_cause_candidate(question, None, CORRECT_SENTENCE)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.answer, "xuất thân thấp kém")
        self.assertTrue(candidate.relation_evidence)
        self.assertGreaterEqual(candidate.subject_match_score, 0.9)
        self.assertGreaterEqual(candidate.target_relation_score, 0.9)

    def test_rejects_causal_sentence_about_another_subject(self):
        question = "Nguyên nhân nào khiến Voltaire bị khinh miệt?"
        assessed = assess_cause_candidate(
            question,
            WRONG_SENTENCE,
            "nhu cầu lớn về bông vải tại châu Âu",
            WRONG_SENTENCE,
        )
        self.assertFalse(assessed.relation_evidence)
        self.assertEqual(assessed.subject_match_score, 0.0)
        self.assertIn(
            assessed.rejection_reason,
            {"CAUSE_SUBJECT_MISMATCH", "CAUSE_PHRASE_NOT_FOUND"},
        )

    def test_untyped_neural_span_from_subjectless_passage_is_rejected(self):
        relation = _semantic_relation_assessment(
            "Nguyên nhân nào khiến Voltaire bị khinh miệt?",
            QuestionType.GENERAL,
            "Triều đình chia rẽ và nhiều quý tộc khinh miệt Madame du Barry.",
            "nhiều quý tộc",
            20,
            35,
            "neural_span",
        )
        self.assertFalse(relation["relation_evidence"])
        self.assertEqual(relation["subject_match_score"], 0.0)
        self.assertEqual(relation["relation_rejection_reason"], "CAUSE_SUBJECT_MISMATCH")

    def test_anaphoric_only_cause_is_incomplete(self):
        score, complete, reasons = assess_relation_completeness(
            "Vì sao sự kiện xảy ra?",
            QuestionType.GENERAL,
            "vì điều này",
        )
        self.assertFalse(complete)
        self.assertEqual(score, 0.0)
        self.assertIn("ANAPHORIC_CAUSE_PHRASE", reasons)

    def test_end_to_end_semantic_gate_selects_lower_ranked_grounded_cause(self):
        question = "Nguyên nhân nào khiến Voltaire bị khinh miệt?"
        hits = [
            make_hit("doc_02551_P0001", WRONG_SENTENCE, 1.0, 1),
            make_hit("doc_01436_P0001", CORRECT_SENTENCE, 0.15, 2),
        ]
        with patch("backend.viqa_api.INDEX.retrieve", return_value=hits), patch(
            "backend.viqa_api.READERS.get", return_value=EmptyReader()
        ):
            result = ask_question(
                {
                    "question": question,
                    "retriever": "bm25",
                    "reader": "phobert",
                    "top_k": 2,
                }
            )

        self.assertTrue(result["has_answer"])
        self.assertEqual(result["answer"], "xuất thân thấp kém")
        self.assertEqual(result["question_relation"], "CAUSE")
        self.assertEqual(result["question_subject"], "Voltaire")
        self.assertEqual(result["selected_passage_id"], "doc_01436_P0001")
        wrong = next(item for item in result["passages"] if item["passage_id"] == "doc_02551_P0001")
        self.assertIn(
            wrong["rejection_reason"],
            {"CAUSE_SUBJECT_MISMATCH", "CAUSE_PHRASE_NOT_FOUND"},
        )
        self.assertEqual(wrong["subject_match_score"], 0.0)

    def test_regression_fixture_is_semantic_not_exact_only(self):
        fixture = Path(__file__).parent / "data" / "qa_semantic_regressions.json"
        rows = json.loads(fixture.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 6)
        self.assertGreaterEqual(sum(row["expected_relation"] == "CAUSE" for row in rows), 5)
        self.assertTrue(all("forbidden_answer_contains" in row for row in rows))


if __name__ == "__main__":
    unittest.main()
