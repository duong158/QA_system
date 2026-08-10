import unittest
from unittest.mock import patch

from backend.chunking import Passage
from backend.viqa_api import IndexedPassage, SearchHit, ask_question, expand_answer_to_sentence


def make_hit(passage_id: str, text: str, raw: float, normalized: float) -> SearchHit:
    metadata = Passage("DOC", passage_id, "Title", "DOC_PAR0001", 0, 0, text)
    return SearchHit(IndexedPassage(metadata, tuple(), {}), raw, normalized)


class FakePredictor:
    def __init__(self):
        self.calls = []

    def predict(self, question, context, no_answer_threshold):
        self.calls.append(context)
        if context == "retriever favorite":
            return {"answer": "weak", "confidence": 0.2, "score": 1.0, "null_score": 2.0, "score_margin": -1.0, "start": 0, "end": 4}
        return {"answer": "reader wins", "confidence": 0.9, "score": 5.0, "null_score": 1.0, "score_margin": 4.0, "start": 0, "end": 11}


class PipelineTests(unittest.TestCase):
    def test_expands_reader_span_to_containing_sentence(self):
        context = "Paris is the capital of France. The city sits on the Seine river."

        answer = expand_answer_to_sentence(
            context,
            "Seine river",
            context.index("Seine"),
            context.index("river") + len("river"),
        )

        self.assertEqual(answer, "The city sits on the Seine river.")

    def test_reader_runs_on_every_top_k_and_reranks(self):
        hits = [
            make_hit("DOC_P0001", "retriever favorite", 12.0, 1.0),
            make_hit("DOC_P0002", "reader favorite", 8.0, 0.0),
        ]
        predictor = FakePredictor()
        with patch("backend.viqa_api.INDEX.retrieve", return_value=hits), patch("backend.viqa_api.READERS.get", return_value=predictor):
            result = ask_question({"question": "test question", "retriever": "bm25", "reader": "phobert", "top_k": 2})

        self.assertEqual(len(predictor.calls), 2)
        self.assertEqual(result["selected_passage_id"], "DOC_P0002")
        self.assertEqual(result["answer"], "reader wins")
        self.assertGreater(result["passages"][1]["final_score"], result["passages"][0]["final_score"])

    def test_production_backend_contains_no_question_answer_mapping(self):
        source = __import__("pathlib").Path("backend/viqa_api.py").read_text(encoding="utf-8")
        self.assertNotIn("Phạm Văn Đồng", source)
        self.assertNotIn("Falling back to heuristic", source)
        self.assertNotIn("reader_bias", source)


if __name__ == "__main__":
    unittest.main()
