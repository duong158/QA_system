import unittest
from unittest.mock import patch

from backend.chunking import Passage
from backend.viqa_api import IndexedPassage, SearchHit, ask_question
from reader.answer_completeness import assess_answer_completeness


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


class FixedSpanReader:
    def __init__(self, answer: str):
        self.answer = answer

    def predict_many(self, question, contexts, **kwargs):
        rows = []
        for context in contexts:
            start = context.index(self.answer)
            rows.append(
                {
                    "answer": self.answer,
                    "candidate_answer": self.answer,
                    "candidate_start": start,
                    "candidate_end": start + len(self.answer),
                    "confidence": 0.80,
                    "reader_threshold_score": 0.80,
                    "score_margin": 2.0,
                    "passes_reader_threshold": True,
                    "valid_span": True,
                    "has_answer": True,
                }
            )
        return rows


class LiveSemanticRegressionTests(unittest.TestCase):
    def test_slavery_cause_uses_subject_predicate_and_returns_cotton_demand(self):
        question = "Vì sao chủ nghĩa nô lệ phát triển mạnh tại các tiểu bang miền Nam?"
        context = (
            "Chủ nghĩa nô lệ bị bãi bỏ tại các tiểu bang miền Bắc, nhưng lại phát triển "
            "mạnh tại các tiểu bang miền Nam vì nhu cầu lớn về bông vải tại châu Âu."
        )
        with patch(
            "backend.viqa_api.INDEX.retrieve",
            return_value=[make_hit("doc_02551_P0001", context, 1.0, 1)],
        ), patch("backend.viqa_api.READERS.get", return_value=EmptyReader()):
            result = ask_question({"question": question, "top_k": 1})

        self.assertTrue(result["has_answer"])
        self.assertIn("nhu cầu lớn về bông vải", result["answer"])
        self.assertEqual(result["question_subject"], "chủ nghĩa nô lệ")
        self.assertEqual(result["question_predicate"], "phát triển mạnh")
        self.assertEqual(result["semantic_status"], "VALID")

    def test_madame_effect_repetition_does_not_answer_cause(self):
        question = "Vì sao Madame du Barry bị khinh miệt?"
        answer = "Nhiều quý tộc khinh miệt Madame du Barry"
        context = answer + "."
        with patch(
            "backend.viqa_api.INDEX.retrieve",
            return_value=[make_hit("doc_01426_P0001", context, 1.0, 1)],
        ), patch("backend.viqa_api.READERS.get", return_value=FixedSpanReader(answer)):
            result = ask_question({"question": question, "top_k": 1})

        self.assertFalse(result["has_answer"])
        self.assertIn(
            result["rejection_reason"],
            {"CAUSE_RELATION_NOT_FOUND", "CAUSE_EFFECT_REPETITION"},
        )

    def test_birth_time_rejects_unrelated_year_and_selects_biography(self):
        question = "Phạm Văn Đồng sinh năm nào?"
        wrong = "Love Story, xuất bản năm 1970, là tác phẩm của Erich Segal."
        correct = (
            "Phạm Văn Đồng (1 tháng 3 năm 1906 – 29 tháng 4 năm 2000) "
            "là Thủ tướng đầu tiên của Việt Nam."
        )
        hits = [
            make_hit("doc_love_story", wrong, 1.0, 1),
            make_hit("doc_00001_P0001", correct, 0.15, 2),
        ]
        with patch("backend.viqa_api.INDEX.retrieve", return_value=hits), patch(
            "backend.viqa_api.READERS.get", return_value=EmptyReader()
        ):
            result = ask_question({"question": question, "top_k": 2})

        self.assertTrue(result["has_answer"])
        self.assertIn("1906", result["answer"])
        self.assertNotIn("1970", result["answer"])
        self.assertEqual(result["semantic_relation"], "BIRTH_TIME")
        wrong_passage = next(
            item for item in result["passages"] if item["passage_id"] == "doc_love_story"
        )
        self.assertEqual(wrong_passage["rejection_reason"], "TIME_SUBJECT_MISMATCH")
        self.assertEqual(wrong_passage["subject_match_score"], 0.0)

    def test_roosevelt_answer_never_ends_with_dangling_connector(self):
        question = "Vì sao Roosevelt bị Hitler khinh thường?"
        raw = "vì cho rằng ông là một lãnh tụ yếu thế, hay"
        context = (
            "Hitler khinh thường Roosevelt vì cho rằng ông là một lãnh tụ yếu thế, "
            "hay dao động và vì thế đánh giá thấp Hoa Kỳ."
        )
        with patch(
            "backend.viqa_api.INDEX.retrieve",
            return_value=[make_hit("doc_roosevelt", context, 1.0, 1)],
        ), patch("backend.viqa_api.READERS.get", return_value=FixedSpanReader(raw)):
            result = ask_question({"question": question, "top_k": 1})

        self.assertTrue(result["has_answer"])
        self.assertTrue(assess_answer_completeness(result["answer"]).complete)
        self.assertFalse(result["answer"].casefold().endswith((" hay", " và", " hoặc", " nhưng")))
        self.assertGreaterEqual(result["answer_refinement"]["completeness_after"], 0.5)


if __name__ == "__main__":
    unittest.main()
